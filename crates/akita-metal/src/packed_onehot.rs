use akita_error::AkitaError;
use akita_prover::compute::{CommitInnerPlan, DecomposeFoldPlan, RootCommitKernel};
use akita_prover::{CommitInnerWitness, DecomposeFoldWitness, OneHotPoly, RootCommitSource};
use jolt_field::One;

use crate::backend::MetalBackend;
use crate::field::F;
use crate::prepared::MetalPreparedSetup;
use crate::runtime::PackedDecomposeFoldParams;
use crate::{MetalCommitError, MetalExecutionPolicy};

/// Borrowed row-major selectors for the packed Metal root-commit kernel.
///
/// Byte zero denotes an absent coefficient. Values in `1..onehot_k` select
/// that coefficient within the row's one-hot chunk. Logical chunks are
/// column-major through `column_capacity`; omitted suffix columns are zero.
#[derive(Clone, Copy, Debug)]
pub struct PackedOneHotCommitView<'a> {
    lanes: &'a [u8],
    active_zero_rows: &'a [u64],
    zero_column_mask: u64,
    num_rows: usize,
    num_columns: usize,
    column_capacity: usize,
    onehot_k: usize,
    hot_entries: usize,
    zero_suffix_start: usize,
}

impl<'a> PackedOneHotCommitView<'a> {
    /// Validate a borrowed packed selector matrix without copying it.
    pub fn new(
        onehot_k: usize,
        column_capacity: usize,
        num_columns: usize,
        lanes: &'a [u8],
    ) -> Result<Self, AkitaError> {
        Self::new_with_active_zero_rows(onehot_k, column_capacity, num_columns, lanes, &[], 0)
    }

    /// Validate selectors whose row-zero coefficients are described by a
    /// compact active-row bitset and a fixed live-column mask.
    pub fn new_with_active_zero_rows(
        onehot_k: usize,
        column_capacity: usize,
        num_columns: usize,
        lanes: &'a [u8],
        active_zero_rows: &'a [u64],
        zero_column_mask: u64,
    ) -> Result<Self, AkitaError> {
        Self::new_inner(
            onehot_k,
            column_capacity,
            num_columns,
            lanes,
            active_zero_rows,
            zero_column_mask,
            None,
            None,
        )
    }

    /// Validate structural K256 geometry while reusing an exact entry count
    /// accumulated when the selector bytes were produced.
    pub fn new_k256_with_precomputed_hot_entries(
        column_capacity: usize,
        num_columns: usize,
        lanes: &'a [u8],
        active_zero_rows: &'a [u64],
        zero_column_mask: u64,
        hot_entries: usize,
    ) -> Result<Self, AkitaError> {
        Self::new_inner(
            256,
            column_capacity,
            num_columns,
            lanes,
            active_zero_rows,
            zero_column_mask,
            Some(hot_entries),
            None,
        )
    }

    /// Reuse an exact producer-side entry count and a certified all-zero row suffix.
    ///
    /// The caller must include committed zeros in the count and certify that rows
    /// at or beyond `zero_suffix_start` have no nonzero or committed-zero entries.
    /// Selector bounds and structural geometry are still validated here.
    #[expect(
        clippy::too_many_arguments,
        reason = "borrowed packed selector geometry and producer metrics"
    )]
    pub fn new_with_precomputed_metrics(
        onehot_k: usize,
        column_capacity: usize,
        num_columns: usize,
        lanes: &'a [u8],
        active_zero_rows: &'a [u64],
        zero_column_mask: u64,
        hot_entries: usize,
        zero_suffix_start: usize,
    ) -> Result<Self, AkitaError> {
        Self::new_inner(
            onehot_k,
            column_capacity,
            num_columns,
            lanes,
            active_zero_rows,
            zero_column_mask,
            Some(hot_entries),
            Some(zero_suffix_start),
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the borrowed selector view validates each independent source dimension"
    )]
    fn new_inner(
        onehot_k: usize,
        column_capacity: usize,
        num_columns: usize,
        lanes: &'a [u8],
        active_zero_rows: &'a [u64],
        zero_column_mask: u64,
        precomputed_hot_entries: Option<usize>,
        zero_suffix_start: Option<usize>,
    ) -> Result<Self, AkitaError> {
        if onehot_k == 0 || onehot_k > usize::from(u8::MAX) + 1 || !onehot_k.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot K={onehot_k} must be a power of two at most 256"
            )));
        }
        if column_capacity == 0 || !column_capacity.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot column capacity {column_capacity} must be a nonzero power of two"
            )));
        }
        if num_columns == 0 || num_columns > column_capacity {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot live column count {num_columns} must be in 1..={column_capacity}"
            )));
        }
        if !lanes.len().is_multiple_of(num_columns) {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot lane count {} is not divisible by {num_columns} live columns",
                lanes.len()
            )));
        }
        let num_rows = lanes.len() / num_columns;
        if num_rows == 0 || !num_rows.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot row count {num_rows} must be a nonzero power of two"
            )));
        }
        let total_field_elements = num_rows
            .checked_mul(column_capacity)
            .and_then(|chunks| chunks.checked_mul(onehot_k))
            .ok_or_else(|| {
                AkitaError::InvalidInput("packed one-hot logical size overflow".into())
            })?;
        if !total_field_elements.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot logical field length {total_field_elements} is not a power of two"
            )));
        }
        let zero_suffix_start = zero_suffix_start.unwrap_or(num_rows);
        if zero_suffix_start > num_rows {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot zero suffix starts at row {zero_suffix_start}, beyond {num_rows} rows"
            )));
        }
        let live_column_mask = if num_columns == u64::BITS as usize {
            u64::MAX
        } else {
            (1u64 << num_columns) - 1
        };
        if zero_column_mask & !live_column_mask != 0 {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot committed-zero mask {zero_column_mask:#x} exceeds {num_columns} live columns"
            )));
        }
        let expected_active_words = num_rows.div_ceil(u64::BITS as usize);
        if zero_column_mask == 0 {
            if !active_zero_rows.is_empty() {
                return Err(AkitaError::InvalidInput(
                    "packed one-hot active-zero rows require a nonzero column mask".into(),
                ));
            }
        } else if active_zero_rows.len() != expected_active_words {
            return Err(AkitaError::InvalidSize {
                expected: expected_active_words,
                actual: active_zero_rows.len(),
            });
        }
        let hot_entries = if let Some(hot_entries) = precomputed_hot_entries {
            if hot_entries > lanes.len() {
                return Err(AkitaError::InvalidInput(
                    "precomputed packed entry counts cannot exceed the lane count".into(),
                ));
            }
            if onehot_k < 256
                && lanes
                    .iter()
                    .copied()
                    .max()
                    .is_some_and(|lane| usize::from(lane) >= onehot_k)
            {
                return Err(AkitaError::InvalidInput(format!(
                    "packed one-hot selector is outside K={onehot_k}"
                )));
            }
            hot_entries
        } else {
            let mut hot_entries = 0usize;
            for (position, &lane) in lanes.iter().enumerate() {
                if usize::from(lane) >= onehot_k {
                    return Err(AkitaError::InvalidInput(format!(
                        "packed one-hot lane {lane} at byte {position} is outside K={onehot_k}"
                    )));
                }
                let row = position / num_columns;
                let column = position % num_columns;
                let committed_zero = lane == 0
                    && zero_column_mask & (1u64 << column) != 0
                    && active_zero_rows
                        .get(row / u64::BITS as usize)
                        .is_some_and(|word| word & (1u64 << (row % u64::BITS as usize)) != 0);
                hot_entries += usize::from(lane != 0 || committed_zero);
            }
            hot_entries
        };
        Ok(Self {
            lanes,
            active_zero_rows,
            zero_column_mask,
            num_rows,
            num_columns,
            column_capacity,
            onehot_k,
            hot_entries,
            zero_suffix_start,
        })
    }

    pub(crate) fn lanes(self) -> &'a [u8] {
        self.lanes
    }

    pub(crate) fn active_zero_rows(self) -> &'a [u64] {
        self.active_zero_rows
    }

    pub(crate) fn zero_column_mask(self) -> u64 {
        self.zero_column_mask
    }

    pub(crate) fn commits_zero_at(self, row: usize, column: usize) -> bool {
        self.zero_column_mask & (1u64 << column) != 0
            && self
                .active_zero_rows
                .get(row / u64::BITS as usize)
                .is_some_and(|word| word & (1u64 << (row % u64::BITS as usize)) != 0)
    }

    pub(crate) fn num_rows(self) -> usize {
        self.num_rows
    }

    pub(crate) fn num_columns(self) -> usize {
        self.num_columns
    }

    pub(crate) fn column_capacity(self) -> usize {
        self.column_capacity
    }

    pub(crate) fn onehot_k(self) -> usize {
        self.onehot_k
    }

    pub(crate) fn hot_entries(self) -> usize {
        self.hot_entries
    }

    pub(crate) fn zero_suffix_start(self) -> usize {
        self.zero_suffix_start
    }
}

impl<const D: usize> RootCommitKernel<PackedOneHotCommitView<'_>, F, D> for MetalBackend {
    #[tracing::instrument(skip_all, name = "MetalBackend::packed_onehot_commit_inner")]
    fn commit_inner_group(
        &self,
        prepared: &Self::PreparedSetup,
        sources: Vec<PackedOneHotCommitView<'_>>,
        plan: CommitInnerPlan,
    ) -> Result<Vec<CommitInnerWitness<F>>, AkitaError> {
        let [source] = sources.as_slice() else {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot commitment requires exactly one physical source, got {}",
                sources.len()
            )));
        };
        self.commit_packed_onehot::<D>(prepared, *source, plan)
            .map(|witness| vec![witness])
    }
}

impl MetalBackend {
    /// Commit one resident packed source through the D512 rank-1 or the
    /// D128 rank-3 K256 Metal kernel, selected by the plan's ring geometry.
    pub fn commit_packed_onehot<const D: usize>(
        &self,
        prepared: &MetalPreparedSetup,
        source: PackedOneHotCommitView<'_>,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F>, AkitaError> {
        if D == crate::packed_onehot_fp128_d128_rank3::RING_D {
            return self.commit_packed_onehot_d128_rank3::<D>(prepared, source, plan);
        }
        let shape = match crate::packed_onehot_fp128_d512::validate_shape::<D>(source, plan) {
            Ok(shape) => shape,
            Err(error) if self.policy() == MetalExecutionPolicy::RequireMetal => return Err(error),
            Err(_) => return self.commit_packed_onehot_cpu::<D>(prepared, source, plan),
        };
        let Some(runtime) = self.runtime() else {
            return match self.policy() {
                MetalExecutionPolicy::RequireMetal => {
                    Err(MetalCommitError::DeviceUnavailable.into_akita())
                }
                MetalExecutionPolicy::PreferMetal => {
                    self.commit_packed_onehot_cpu::<D>(prepared, source, plan)
                }
            };
        };
        if !runtime.supports_packed_fp128_d512_panels() {
            return match self.policy() {
                MetalExecutionPolicy::RequireMetal => Err(MetalCommitError::UnsupportedShape(
                    "fp128 D512 panel pipeline is unavailable".into(),
                )
                .into_akita()),
                MetalExecutionPolicy::PreferMetal => {
                    self.commit_packed_onehot_cpu::<D>(prepared, source, plan)
                }
            };
        }
        crate::packed_onehot_fp128_d512::commit_validated::<D>(
            self, prepared, runtime, source, plan, shape,
        )
    }

    /// Build and cache the packed A-matrix prefix the D512 rank-1 or D128
    /// rank-3 K256 commit will use, so the commit itself does not pay `matrix_prepare` on the critical
    /// path. Meant to run concurrently with the caller's row assembly. Returns
    /// `Ok(true)` when the prefix was already resident, `Ok(false)` when this
    /// call built it or when no Metal runtime is available (a later commit then
    /// takes whatever path it would have taken anyway).
    pub fn prewarm_packed_onehot_matrix(
        &self,
        prepared: &MetalPreparedSetup,
        ring_d: usize,
        n_a: usize,
        active_a_cols: usize,
    ) -> Result<bool, AkitaError> {
        let Some(runtime) = self.runtime() else {
            return Ok(false);
        };
        let supported = match ring_d {
            512 => runtime.supports_packed_fp128_d512_panels(),
            128 => runtime.supports_packed_fp128_d128_rank3(),
            _ => false,
        };
        if !supported {
            return Ok(false);
        }
        let _span = tracing::info_span!(
            "MetalBackend::prewarm_packed_onehot_matrix",
            ring_d,
            n_a,
            active_a_cols
        )
        .entered();
        let matrix = prepared.matrix(runtime, ring_d, n_a, active_a_cols)?;
        Ok(matrix.cache_hit)
    }

    fn commit_packed_onehot_d128_rank3<const D: usize>(
        &self,
        prepared: &MetalPreparedSetup,
        source: PackedOneHotCommitView<'_>,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F>, AkitaError> {
        let shape = match crate::packed_onehot_fp128_d128_rank3::validate_shape::<D>(source, plan) {
            Ok(shape) => shape,
            Err(error) if self.policy() == MetalExecutionPolicy::RequireMetal => return Err(error),
            Err(_) => return self.commit_packed_onehot_cpu::<D>(prepared, source, plan),
        };
        let Some(runtime) = self.runtime() else {
            return match self.policy() {
                MetalExecutionPolicy::RequireMetal => {
                    Err(MetalCommitError::DeviceUnavailable.into_akita())
                }
                MetalExecutionPolicy::PreferMetal => {
                    self.commit_packed_onehot_cpu::<D>(prepared, source, plan)
                }
            };
        };
        if !runtime.supports_packed_fp128_d128_rank3() {
            return match self.policy() {
                MetalExecutionPolicy::RequireMetal => Err(MetalCommitError::UnsupportedShape(
                    "fp128 D128 rank-3 pipeline is unavailable".into(),
                )
                .into_akita()),
                MetalExecutionPolicy::PreferMetal => {
                    self.commit_packed_onehot_cpu::<D>(prepared, source, plan)
                }
            };
        }
        crate::packed_onehot_fp128_d128_rank3::commit_validated::<D>(
            self, prepared, runtime, source, plan, shape,
        )
    }

    fn commit_packed_onehot_cpu<const D: usize>(
        &self,
        prepared: &MetalPreparedSetup,
        source: PackedOneHotCommitView<'_>,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F>, AkitaError> {
        let indices = (0..source.column_capacity())
            .flat_map(|column| {
                (0..source.num_rows()).map(move |row| {
                    if column >= source.num_columns() {
                        None
                    } else {
                        let lane = source.lanes()[row * source.num_columns() + column];
                        (lane != 0 || source.commits_zero_at(row, column)).then_some(lane)
                    }
                })
            })
            .collect();
        let poly = OneHotPoly::<F, u8>::new(source.onehot_k(), indices)?;
        let view = <OneHotPoly<F, u8> as RootCommitSource<F, D>>::commit_view(&poly)?;
        let mut witnesses =
            self.cpu_backend()
                .commit_inner_group(&prepared.cpu, vec![view], plan)?;
        witnesses.pop().ok_or_else(|| {
            AkitaError::InvalidSetup("CPU packed fallback returned no commitment witness".into())
        })
    }

    /// Decompose and challenge-fold a resident K16/capacity64 source at D512,
    /// or a K256/capacity32 source at D512 or D128.
    pub fn decompose_fold_packed_onehot<const D: usize>(
        &self,
        source: PackedOneHotCommitView<'_>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness<F>, AkitaError> {
        let total_start = std::time::Instant::now();
        if !matches!(
            (D, source.onehot_k(), source.column_capacity()),
            (128 | 512, 256, 32) | (512, 16, 64)
        ) || plan.num_positions_per_block == 0
            || plan.num_digits == 0
            || !(source.num_rows() * source.onehot_k()).is_multiple_of(D)
        {
            return Err(MetalCommitError::UnsupportedShape(
                "packed opening requires K256/capacity32 at D128 or D512, or K16/capacity64 at D512, and nonempty output"
                    .into(),
            )
            .into_akita());
        }

        let segment_rings = source.num_rows() * source.onehot_k() / D;
        if !segment_rings.is_multiple_of(plan.num_positions_per_block) {
            return Err(MetalCommitError::UnsupportedShape(
                "packed opening segment is not aligned to the scheduled positions".into(),
            )
            .into_akita());
        }
        let blocks_per_column = segment_rings / plan.num_positions_per_block;
        let live_challenges = source
            .num_columns()
            .checked_mul(blocks_per_column)
            .ok_or_else(|| {
                MetalCommitError::ShapeOverflow("packed opening live challenges").into_akita()
            })?;
        let expected_challenges = source
            .column_capacity()
            .checked_mul(blocks_per_column)
            .ok_or_else(|| {
                MetalCommitError::ShapeOverflow("packed opening challenges").into_akita()
            })?;
        if plan.challenges.len() != expected_challenges {
            return Err(AkitaError::InvalidSize {
                expected: expected_challenges,
                actual: plan.challenges.len(),
            });
        }
        let challenge_weight = plan
            .challenges
            .first()
            .map_or(0, |challenge| challenge.positions.len());
        if challenge_weight == 0
            || plan.challenges[..live_challenges].iter().any(|challenge| {
                challenge.positions.len() != challenge_weight
                    || challenge.coeffs.len() != challenge_weight
            })
        {
            return Err(MetalCommitError::UnsupportedShape(
                "packed opening requires a fixed nonzero challenge weight".into(),
            )
            .into_akita());
        }

        let challenge_terms = live_challenges
            .checked_mul(challenge_weight)
            .ok_or_else(|| {
                MetalCommitError::ShapeOverflow("packed opening challenge terms").into_akita()
            })?;
        let mut positions = Vec::with_capacity(challenge_terms);
        let mut coefficients = Vec::with_capacity(challenge_terms);
        let dense_len = live_challenges.checked_mul(64).ok_or_else(|| {
            MetalCommitError::ShapeOverflow("packed opening dense challenges").into_akita()
        })?;
        let mut dense_subring64 = vec![0i8; dense_len];
        let mut has_subring64_embedding = true;
        let subring_stride = D / 64;
        for (challenge_index, challenge) in plan.challenges[..live_challenges].iter().enumerate() {
            challenge.validate::<D>()?;
            for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
                positions.push(u16::try_from(position).map_err(|_| {
                    MetalCommitError::UnsupportedShape(
                        "packed opening challenge position does not fit u16".into(),
                    )
                    .into_akita()
                })?);
                coefficients.push(coefficient);
                let dense_position = usize::try_from(position)
                    .ok()
                    .filter(|position| position.is_multiple_of(subring_stride))
                    .map(|position| position / subring_stride)
                    .filter(|&position| position < 64);
                if let Some(dense_position) = dense_position {
                    let slot = challenge_index * 64 + dense_position;
                    if dense_subring64[slot] == 0 {
                        dense_subring64[slot] = coefficient;
                    } else {
                        has_subring64_embedding = false;
                    }
                } else {
                    has_subring64_embedding = false;
                }
            }
        }
        let dense_subring64 = has_subring64_embedding.then_some(dense_subring64);
        let output_coefficients = plan.num_positions_per_block.checked_mul(D).ok_or_else(|| {
            MetalCommitError::ShapeOverflow("packed opening output coefficients").into_akita()
        })?;
        let runtime = self
            .runtime()
            .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?;
        let params = PackedDecomposeFoldParams {
            onehot_k: source.onehot_k() as u64,
            zero_suffix_start: source.zero_suffix_start() as u64,
            num_rows: source.num_rows() as u64,
            num_columns: source.num_columns() as u64,
            lane_stride: source.num_columns() as u64,
            num_positions: plan.num_positions_per_block as u64,
            position_start: 0,
            blocks_per_column: blocks_per_column as u64,
            challenge_weight: challenge_weight as u64,
            output_coefficients: output_coefficients as u64,
            zero_column_mask: source.zero_column_mask(),
        };
        let outcome = runtime
            .dispatch_packed_fp128_decompose_fold_streaming(
                D,
                source.lanes(),
                source.active_zero_rows(),
                &positions,
                &coefficients,
                dense_subring64.as_deref(),
                params,
                plan.num_positions_per_block,
                |_, _| {},
            )
            .map_err(MetalCommitError::into_akita)?;
        let timings = outcome.timings;
        let allocation_bytes = outcome.allocation_bytes;
        let mut compressed = Vec::with_capacity(plan.num_positions_per_block);
        let mut chunks = outcome.centered_coefficients.chunks_exact(D);
        for coefficients in chunks.by_ref() {
            let coefficients: &[i32; D] = coefficients.try_into().map_err(|_| {
                AkitaError::InvalidSetup("packed opening produced a short coefficient row".into())
            })?;
            compressed.push(*coefficients);
        }
        if !chunks.remainder().is_empty() {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: chunks.remainder().len(),
            });
        }
        let centered = if plan.num_digits == 1 {
            compressed
        } else {
            let expanded_len = compressed
                .len()
                .checked_mul(plan.num_digits)
                .ok_or_else(|| {
                    MetalCommitError::ShapeOverflow("packed opening digit expansion").into_akita()
                })?;
            let mut expanded = Vec::with_capacity(expanded_len);
            for coefficients in compressed {
                expanded.push(coefficients);
                expanded.extend((1..plan.num_digits).map(|_| [0i32; D]));
            }
            expanded
        };
        let modulus = (-F::one()).to_canonical_u128() + 1;
        let witness = akita_prover::backend::poly_helpers::build_decompose_fold_witness::<F, D>(
            centered, modulus,
        );
        self.update_opening_metrics(|metrics| {
            metrics.command_wall_time += timings.command_wall;
            metrics.gpu_active_time += timings.gpu.unwrap_or_default();
            metrics.buffer_setup_time += timings.buffer_setup;
            metrics.readback_time += timings.readback_copy;
            metrics.allocation_bytes = metrics.allocation_bytes.saturating_add(allocation_bytes);
        })
        .map_err(MetalCommitError::into_akita)?;
        tracing::debug!(
            elapsed_s = total_start.elapsed().as_secs_f64(),
            "completed packed Metal decompose-fold"
        );
        Ok(witness)
    }
}

#[cfg(test)]
mod tests {
    use akita_challenges::SparseChallenge;
    use akita_prover::compute::{OpeningFoldKernel, RootOpeningSource};
    use akita_prover::CpuBackend;

    use super::*;

    #[test]
    fn k16_decompose_fold_matches_cpu_with_committed_zeros() {
        const D: usize = 512;
        const ROWS: usize = 4096;
        const COLUMNS: usize = 59;
        const CAPACITY: usize = 64;
        const POSITIONS: usize = 32;
        const LIVE_ROWS: usize = ROWS / 2 + 1;
        const BLOCKS: usize = ROWS * 16 / D / POSITIONS;
        let lanes = (0..ROWS * COLUMNS)
            .map(|i| {
                if i / COLUMNS < LIVE_ROWS {
                    ((i * 7 + i / COLUMNS) % 16) as u8
                } else {
                    0
                }
            })
            .collect::<Vec<_>>();
        let mut active = vec![0u64; ROWS / 64];
        for row in (0..LIVE_ROWS).step_by(2) {
            active[row / 64] |= 1 << (row % 64);
        }
        let mask = 1 | (1 << 58);
        let source = PackedOneHotCommitView::new_with_active_zero_rows(
            16, CAPACITY, COLUMNS, &lanes, &active, mask,
        )
        .unwrap();
        let indices = (0..CAPACITY)
            .flat_map(|column| {
                let lanes = &lanes;
                (0..ROWS).map(move |row| {
                    if column >= COLUMNS {
                        return None;
                    }
                    let lane = lanes[row * COLUMNS + column];
                    (lane != 0
                        || (row < LIVE_ROWS && row % 2 == 0 && (column == 0 || column == 58)))
                        .then_some(lane)
                })
            })
            .collect();
        let with_suffix = PackedOneHotCommitView::new_with_precomputed_metrics(
            16,
            CAPACITY,
            COLUMNS,
            &lanes,
            &active,
            mask,
            source.hot_entries(),
            LIVE_ROWS,
        )
        .unwrap();
        let oracle = OneHotPoly::<F, u8>::new(16, indices).unwrap();
        let view = <OneHotPoly<F, u8> as RootOpeningSource<F, D>>::opening_view(&oracle).unwrap();
        let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        for stride in [1, 8] {
            let challenges = (0..CAPACITY * BLOCKS)
                .map(|i| SparseChallenge {
                    positions: (0..4)
                        .map(|j| (((i * 13 + j * 17) % (D / stride)) * stride) as u32)
                        .collect(),
                    coeffs: [1, -2, 3, -4].into_iter().collect(),
                })
                .collect::<Vec<_>>();
            for num_digits in [1, 3] {
                let plan = DecomposeFoldPlan {
                    challenges: &challenges,
                    num_positions_per_block: POSITIONS,
                    num_digits,
                    log_basis: 3,
                };
                let expected = CpuBackend::DEFAULT
                    .decompose_fold(None, view, plan)
                    .unwrap();
                for source in [source, with_suffix] {
                    let actual = backend
                        .decompose_fold_packed_onehot::<D>(source, plan)
                        .unwrap();
                    assert_eq!(
                        actual.centered_coeffs_flat(),
                        expected.centered_coeffs_flat()
                    );
                    assert_eq!(actual.z_folded_rings, expected.z_folded_rings);
                }
            }
        }
    }

    #[test]
    fn validates_borrowed_geometry_and_lanes() {
        let lanes = vec![0, 1, 255, 7, 0, 3, 2, 0];
        let view = PackedOneHotCommitView::new(256, 4, 2, &lanes).unwrap();
        assert_eq!(view.lanes().as_ptr(), lanes.as_ptr());
        assert_eq!(view.num_rows(), 4);
        assert_eq!(view.hot_entries(), 5);

        assert!(PackedOneHotCommitView::new(256, 4, 3, &lanes).is_err());
        assert!(PackedOneHotCommitView::new(128, 4, 2, &lanes).is_err());
    }

    #[test]
    fn precomputed_metrics_validate_small_k_selectors() {
        for k in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
            let lanes = [0, (k - 1) as u8, 0, 0];
            let count = if k == 1 { 1 } else { 2 };
            let view = PackedOneHotCommitView::new_with_precomputed_metrics(
                k,
                2,
                2,
                &lanes,
                &[1],
                1,
                count,
                1,
            )
            .unwrap();
            assert_eq!(view.hot_entries(), count);
            assert_eq!(view.zero_suffix_start(), 1);
            assert!(view.commits_zero_at(0, 0));
            if k < 256 {
                let invalid = [0, k as u8, 0, 0];
                assert!(PackedOneHotCommitView::new_with_precomputed_metrics(
                    k,
                    2,
                    2,
                    &invalid,
                    &[1],
                    1,
                    count,
                    1,
                )
                .is_err());
            }
            assert!(PackedOneHotCommitView::new_with_precomputed_metrics(
                k,
                2,
                2,
                &lanes,
                &[1],
                1,
                5,
                1,
            )
            .is_err());
        }
    }

    #[test]
    fn active_rows_distinguish_committed_zero_from_absence() {
        let lanes = vec![0, 0, 0, 7, 0, 0, 3, 0];
        let view =
            PackedOneHotCommitView::new_with_active_zero_rows(16, 4, 2, &lanes, &[0b0101], 0b01)
                .unwrap();
        assert!(view.commits_zero_at(0, 0));
        assert!(!view.commits_zero_at(1, 0));
        assert!(view.commits_zero_at(2, 0));
        assert!(!view.commits_zero_at(0, 1));
        assert_eq!(view.hot_entries(), 4);

        let lanes = vec![0, 0, 0, 7, 0, 0, 3, 0];
        let scanned =
            PackedOneHotCommitView::new_with_active_zero_rows(256, 4, 2, &lanes, &[0b0101], 0b01)
                .unwrap();
        let precomputed = PackedOneHotCommitView::new_k256_with_precomputed_hot_entries(
            4,
            2,
            &lanes,
            &[0b0101],
            0b01,
            scanned.hot_entries(),
        )
        .unwrap();
        assert_eq!(precomputed.hot_entries(), scanned.hot_entries());
        let with_suffix = PackedOneHotCommitView::new_with_precomputed_metrics(
            256,
            4,
            2,
            &lanes,
            &[0b0101],
            0b01,
            scanned.hot_entries(),
            3,
        )
        .unwrap();
        assert_eq!(with_suffix.zero_suffix_start(), 3);
        assert!(PackedOneHotCommitView::new_with_precomputed_metrics(
            256,
            4,
            2,
            &lanes,
            &[0b0101],
            0b01,
            scanned.hot_entries(),
            5,
        )
        .is_err());
    }
}
