//! Compare compact source commitments with multiplication by ring monomials.
#![cfg(target_os = "macos")]

use akita_algebra::CyclotomicRing;
use akita_metal::onehot::{DeviceK16Sources, OneHotCommitShape};
use akita_metal::onehot_ntt::{DeviceOneHotNttKey, OneHotNttWorkspace, SharedKeyNttWorkspace};
use akita_metal::{AkitaMetal, AkitaMetalError};
use jolt_field::{Prime128OffsetA7F7 as F, Ring, Zero};

const D: usize = 512;
const COLUMNS: usize = 9;

#[test]
fn row_major_commitments_preserve_zeros_prefixes_and_reconstruction_boundaries() {
    let metal = AkitaMetal::new().expect("Metal device and shaders");
    let mut workspace = OneHotNttWorkspace::new(&metal);
    let mut shared = SharedKeyNttWorkspace::new(&metal);
    for (positions, rank, digits) in [(4usize, 2, 3), (512, 1, 1)] {
        let shape = OneHotCommitShape {
            n_a: rank,
            active_a_cols: positions * digits,
            num_digits_inner: digits,
        };
        let mut state = 901u128;
        let key: Vec<F> = (0..rank * shape.active_a_cols * D)
            .map(|i| {
                state = state.wrapping_mul(0xda942042e4dd58b5).wrapping_add(1);
                if i % 7 == 0 {
                    -F::from_u64(1)
                } else {
                    F::from_u128(state)
                }
            })
            .collect();
        let device_key = DeviceOneHotNttKey::new(&metal, &key).expect("key");
        let logical_rows = positions * 32 * 2;
        let prefix = positions * 32 + 7;
        let mut selectors = vec![0; prefix * COLUMNS];
        let mut masks = vec![0; prefix];
        for row in 0..prefix {
            // The larger case is sparse so the independent ring oracle stays
            // cheap while crossing both 252-position and block boundaries.
            if positions == 4 || [0, 1, 251 * 32, 252 * 32, 511 * 32, prefix - 1].contains(&row) {
                selectors[row * COLUMNS] = if row < 32 { 0 } else { (row % 16) as u8 };
                masks[row] = 1; // column zero commits digit zero too
                selectors[row * COLUMNS + 1] = 15;
                // The last column reaches the second sixteen-output tile.
                selectors[row * COLUMNS + COLUMNS - 1] = 7;
                // Columns two through seven remain absent despite zero selectors.
            }
        }
        let mut expected = Vec::new();
        for column in 0..COLUMNS {
            let mut rings = vec![CyclotomicRing::<F, D>::zero(); 2 * rank];
            for row in 0..prefix {
                let hot = selectors[row * COLUMNS + column];
                if hot == 0 && masks[row] & (1 << column) == 0 {
                    continue;
                }
                let exponent = row * 16 + usize::from(hot);
                let ring = exponent / D;
                for a_row in 0..rank {
                    let offset = (a_row * shape.active_a_cols + (ring % positions) * digits) * D;
                    let term = CyclotomicRing::<F, D>::from_slice(&key[offset..offset + D])
                        .negacyclic_shift(exponent % D);
                    rings[(ring / positions) * rank + a_row] += term;
                }
            }
            expected.extend(rings.iter().flat_map(|ring| *ring.coefficients()));
        }
        for direct in [false, true] {
            let source = if direct {
                DeviceK16Sources::from_row_major_producer(
                    &metal,
                    COLUMNS,
                    prefix,
                    logical_rows,
                    |first, output, output_masks| {
                        let start = first * COLUMNS;
                        output.copy_from_slice(&selectors[start..start + output.len()]);
                        output_masks.copy_from_slice(&masks[first..first + output_masks.len()]);
                        Ok(())
                    },
                )
            } else {
                DeviceK16Sources::from_row_major(
                    &metal,
                    selectors.clone(),
                    &masks,
                    COLUMNS,
                    logical_rows,
                )
            }
            .expect("row-major source");
            let (out, _) = workspace
                .commit(&device_key, &source, shape)
                .expect("commit");
            assert_eq!(out.read().expect("read output"), expected.as_slice());
            if rank == 1 && digits == 1 {
                let (out, _) = shared
                    .commit(&device_key, &source, shape)
                    .expect("shared-key commit");
                assert_eq!(out.read().expect("shared-key output"), expected.as_slice());
            }
        }
        let empty =
            DeviceK16Sources::from_row_major(&metal, Vec::new(), &[], COLUMNS, logical_rows)
                .expect("implicit zero suffix");
        let (out, _) = workspace
            .commit(&device_key, &empty, shape)
            .expect("zero commit");
        assert!(out.read().expect("zero output").iter().all(Zero::is_zero));
        if rank == 1 && digits == 1 {
            let (out, _) = shared
                .commit(&device_key, &empty, shape)
                .expect("shared-key empty reuse");
            assert!(out
                .read()
                .expect("shared-key zero output")
                .iter()
                .all(Zero::is_zero));
        }
    }
    // Failure after an earlier chunk may have completed must not publish a
    // partially initialized source. This adds no new proof fixture.
    let failed = DeviceK16Sources::from_row_major_producer(
        &metal,
        COLUMNS,
        8193,
        8224,
        |first, selected, masks| {
            if first != 0 {
                return Err(AkitaMetalError::Shape("injected producer failure".into()));
            }
            selected.fill(0);
            masks.fill(1);
            Ok(())
        },
    );
    assert!(failed.is_err());
}
