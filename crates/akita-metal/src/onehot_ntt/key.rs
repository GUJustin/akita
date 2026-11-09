//! Reusable field-key preparation; no witness dimensions enter this step.

use akita_algebra::ntt::butterfly::forward_ntt;
use akita_algebra::ntt::{NttTwiddles, I32_LAZY_DOT_BATCH};
use akita_algebra::tables::Q128_RAW_PRIMES;
use akita_algebra::{NttKernelPlan, NttPrime};
use akita_error::checked;
use jolt_field::Prime128OffsetA7F7;
use jolt_metal::runtime::DeviceBuffer;

use crate::onehot::OneHotCommitShape;
use crate::{AkitaMetal, AkitaMetalError};

pub(super) const D: usize = 512;
pub(super) const LIMBS: usize = 8;
pub(super) const LIMB_BITS: usize = 16;
pub(super) const ROWS: usize = D / 16;

/// All rings of a flat Fp128 key, split into 16-bit limbs and transformed.
///
/// Preparation depends only on the supplied key. A commitment may view any
/// validated prefix and matrix geometry, independently of its trace length.
/// The transformed key uses twice the coefficient storage of the field key.
/// Commitments cache span sums for the most recent matrix view; changing that
/// view replaces only the sums, leaving the transformed key intact.
pub struct DeviceOneHotNttKey {
    pub(super) coefficient_count: usize,
    pub(super) transformed: DeviceBuffer<u32>,
    pub(super) tables: DeviceBuffer<u32>,
    pub(super) prime: NttPrime<i32>,
    pub(super) reconstruction_positions: usize,
    pub(super) summed: Option<(OneHotCommitShape, DeviceBuffer<u32>)>,
}

impl DeviceOneHotNttKey {
    /// Prepares every complete degree-512 ring, with no source-length input.
    #[expect(
        clippy::arithmetic_side_effects,
        clippy::indexing_slicing,
        reason = "fixed-size limbs/rings and checked allocation dimensions bound every operation"
    )]
    pub fn new(
        metal: &AkitaMetal,
        coefficients: &[Prime128OffsetA7F7],
    ) -> Result<Self, AkitaMetalError> {
        if !coefficients.len().is_multiple_of(D) {
            return Err(AkitaMetalError::Shape(
                "the limb-NTT key must contain complete degree-512 rings".into(),
            ));
        }
        let modulus = Q128_RAW_PRIMES
            .first()
            .copied()
            .ok_or_else(|| AkitaMetalError::Shape("the Q128 profile has no NTT prime".into()))?;
        if modulus <= 0 || modulus >= 1 << 30 || (modulus - 1) % (2 * D as i32) != 0 {
            return Err(AkitaMetalError::Shape(
                "the limb kernel requires a sub-30-bit NTT prime admitting degree 512".into(),
            ));
        }
        let prime = NttPrime::new(modulus);
        let twiddles = NttTwiddles::<i32, D>::compute(prime);
        let plan = NttKernelPlan::detect::<i32>();
        // Centering must recover the integer convolution without wraparound.
        // Round down to complete lazy-dot batches; this is 252 positions for
        // the current Q128 prime and six-product batch.
        let max_positions = (modulus as usize / 2) / (ROWS * u16::MAX as usize);
        let reconstruction_positions = max_positions / I32_LAZY_DOT_BATCH * I32_LAZY_DOT_BATCH;
        if reconstruction_positions == 0 || I32_LAZY_DOT_BATCH > 8 {
            return Err(AkitaMetalError::Shape(
                "the NTT prime/dot batch cannot support the limb reconstruction bound".into(),
            ));
        }
        let len = checked::product([coefficients.len(), LIMBS]).ok_or_else(|| {
            AkitaMetalError::Shape("the transformed key length overflows usize".into())
        })?;
        let mut transformed = Vec::new();
        transformed.try_reserve_exact(len).map_err(|_| {
            AkitaMetalError::Shape(format!("cannot stage {len} transformed key words"))
        })?;
        transformed.resize(len, 0);
        let prepare_ring = |(out, ring): (&mut [u32], &[Prime128OffsetA7F7])| {
            for (limb, out) in out.chunks_exact_mut(D).enumerate() {
                let mut values = std::array::from_fn(|i| {
                    let value =
                        (ring[i].to_canonical_u128() >> (limb * LIMB_BITS)) & u128::from(u16::MAX);
                    prime.from_canonical(value as i32)
                });
                forward_ntt(&mut values, prime, &twiddles, plan);
                // Akita's DIF output is bit-reversed. The specialized K16
                // transform writes natural frequencies with coalesced stores.
                for (frequency, out) in out.iter_mut().enumerate() {
                    let reversed = frequency.reverse_bits() >> (usize::BITS - D.ilog2());
                    *out = prime.normalize(values[reversed]).raw() as u32;
                }
            }
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            transformed
                .par_chunks_exact_mut(D * LIMBS)
                .zip(coefficients.par_chunks_exact(D))
                .for_each(prepare_ring);
        }
        #[cfg(not(feature = "parallel"))]
        transformed
            .chunks_exact_mut(D * LIMBS)
            .zip(coefficients.chunks_exact(D))
            .for_each(prepare_ring);
        let tables = transform_tables(prime, &twiddles);
        Ok(Self {
            coefficient_count: coefficients.len(),
            transformed: DeviceBuffer::from_slice(metal.device(), &transformed)?,
            tables: DeviceBuffer::from_slice(metal.device(), &tables)?,
            prime,
            reconstruction_positions,
            summed: None,
        })
    }
}

#[expect(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    reason = "fixed-size roots and K16/D512 table indices are bounded by construction"
)]
fn transform_tables(prime: NttPrime<i32>, twiddles: &NttTwiddles<i32, D>) -> Vec<u32> {
    let [_, _, psi, untwist] = twiddles.negacyclic_tables();
    let modulus = prime.p as u32;
    let powers: [u32; 2 * D] = std::array::from_fn(|i| {
        let value = prime.to_canonical(psi[i % D]) as u32;
        if i < D {
            value
        } else {
            modulus - value
        }
    });
    let mut tables = Vec::with_capacity(16 * ROWS * ROWS + 16 + D + 2 * D + 16 * D);
    for digit in 0..16 {
        for row in 0..ROWS {
            for frequency in 0..ROWS {
                tables.push(powers[((2 * frequency + 1) * (16 * row + digit)) % (2 * D)]);
            }
        }
    }
    let small_twiddles: Vec<_> = (0..8).map(|j| powers[64 * j]).collect();
    let inverse: Vec<_> = (0..D / 2)
        .map(|j| powers[(2 * D - 2 * j) % (2 * D)])
        .collect();
    let untwist: Vec<_> = untwist
        .iter()
        .map(|&x| prime.to_canonical(x) as u32)
        .collect();
    for values in [&small_twiddles, &inverse, &untwist] {
        tables.extend(values);
        tables.extend(
            values
                .iter()
                .map(|&value| ((u64::from(value) << 32) / u64::from(modulus)) as u32),
        );
    }
    // A uniform ring has the same selected digit in every K16 row. Source
    // spectra are canonical residues in natural frequency order, whereas
    // the prepared key retains Montgomery residues for the dot product.
    let plan = NttKernelPlan::detect::<i32>();
    for digit in 0..16 {
        let mut values = std::array::from_fn(|i| prime.from_canonical(i32::from(i % 16 == digit)));
        forward_ntt(&mut values, prime, twiddles, plan);
        for frequency in 0..D {
            let reversed = frequency.reverse_bits() >> (usize::BITS - D.ilog2());
            tables.push(prime.to_canonical(values[reversed]) as u32);
        }
    }
    tables
}
