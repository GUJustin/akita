//! Root folds checked against polynomial multiplication by ring monomials.
#![cfg(target_os = "macos")]

use akita_algebra::CyclotomicRing;
use akita_metal::onehot::DeviceK16Sources;
use akita_metal::onehot_fold::{OneHotFoldChallenge, OneHotFoldShape, OneHotFoldWorkspace};
use akita_metal::AkitaMetal;
use jolt_field::{Prime128OffsetA7F7 as F, Ring, Zero};

#[test]
fn windowed_root_fold_matches_negacyclic_monomials() {
    const D: usize = 512;
    const P: usize = 4;
    const C: usize = 5;
    const B: usize = 4;
    let metal = AkitaMetal::new().expect("Metal shaders");
    let shape = OneHotFoldShape {
        num_positions: P,
        blocks_per_column: B,
        num_columns: C,
    };
    let ranges = [0..2, 2..3, 3..4, 4..4];
    let challenge_terms: Vec<(Vec<u32>, Vec<i8>)> = (0..C * B)
        .map(|index| {
            let positions = (0..41)
                .map(|term| ((term * 83 + index * 17) % D) as u32)
                .collect();
            let coeffs = (0..41)
                .map(|term| match term % 4 {
                    0 => -128,
                    1 => 127,
                    2 => -3,
                    _ => 2,
                })
                .collect();
            (positions, coeffs)
        })
        .collect();
    let challenges: Vec<_> = challenge_terms
        .iter()
        .map(|(positions, coefficients)| OneHotFoldChallenge {
            positions,
            coefficients,
        })
        .collect();
    let challenge_rings: Vec<_> = challenges
        .iter()
        .map(|challenge| {
            let mut coefficients = [F::zero(); D];
            for (&position, &value) in challenge.positions.iter().zip(challenge.coefficients) {
                let absolute = F::from_u64(u64::from(value.unsigned_abs()));
                coefficients[position as usize] = if value < 0 { -absolute } else { absolute };
            }
            CyclotomicRing::<F, D>::from_slice(&coefficients)
        })
        .collect();
    for prefix in [2 * P * 32 + 7, 0] {
        let mut selectors = vec![0; prefix * C];
        let mut masks = vec![0; prefix];
        for row in 0..prefix {
            selectors[row * C] = (row % 16) as u8;
            masks[row] = 1; // The first column commits zero, including row zero.
            selectors[row * C + 1] = if row % 5 == 0 { 0 } else { 15 };
            selectors[row * C + 3] = 7;
            // Columns two/four and unmasked zeros are absent.
        }
        let mut expected = vec![CyclotomicRing::<F, D>::zero(); P * ranges.len()];
        for row in 0..prefix {
            let ring = row / 32;
            let block = ring / P;
            let chunk = ranges
                .iter()
                .position(|range| range.contains(&block))
                .unwrap();
            for column in 0..C {
                let hot = selectors[row * C + column];
                if hot == 0 && masks[row] & (1 << column) == 0 {
                    continue;
                }
                expected[(ring % P) * ranges.len() + chunk] += challenge_rings[column * B + block]
                    .negacyclic_shift(16 * (row % 32) + usize::from(hot));
            }
        }
        let mut workspace =
            OneHotFoldWorkspace::new(&metal, shape, &challenges, &ranges).expect("workspace");
        for (first, blocks) in [(0, 1), (1, 2), (3, 1)] {
            let start = (first * P * 32).min(prefix);
            let end = ((first + blocks) * P * 32).min(prefix);
            let sources = DeviceK16Sources::from_row_major(
                &metal,
                selectors[start * C..end * C].to_vec(),
                &masks[start..end],
                C,
                blocks * P * 32,
            )
            .expect("source");
            workspace.accumulate(&sources, first).expect("fold window");
            assert!(
                workspace.accumulate(&sources, first).is_err(),
                "overlap violates accumulation bounds"
            );
        }
        let output: Vec<_> = workspace
            .read()
            .expect("read coefficients")
            .into_iter()
            .map(|value| {
                let absolute = F::from_u64(u64::from(value.unsigned_abs()));
                if value < 0 {
                    -absolute
                } else {
                    absolute
                }
            })
            .collect();
        let expected: Vec<_> = expected
            .iter()
            .flat_map(|ring| *ring.coefficients())
            .collect();
        assert_eq!(output, expected);
    }
}

/// Reuse the canonical CPU OneHotBatchView oracle. It accumulates the tensor
/// product of the position/packing weights per nonzero, independently of the
/// GPU's coefficient tiles and two-pass reduction.
#[test]
fn retained_coefficient_packing_matches_cpu_onehot() {
    use akita_cpu_backend::custom_source::{
        OneHotBatchView, RootOpeningSource, SubringCoefficientPackingBatchKernel,
        SubringCoefficientPackingPlan,
    };
    use akita_cpu_backend::{AkitaProverSetup, CpuBackend, OneHotPoly};
    use akita_metal::onehot_packing::{OneHotPackingShape, OneHotPackingWorkspace};
    use akita_params::{BasisMode, SetupMatrixCapacity, SubringCoefficientPackingGeometry};
    use akita_types::{PreparedSubringCoefficientPackingPoint, SubringCoefficientPackingTables};

    const D: usize = 512;
    const P: usize = 2048;
    const B: usize = 4;
    const C: usize = 3;
    const ROWS: usize = P * B * 32;
    let metal = AkitaMetal::new().expect("Metal shaders");
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        1,
        1,
        SetupMatrixCapacity {
            num_field_elements: 1,
        },
    )
    .expect("minimal CPU setup");
    let backend = CpuBackend::<F, F>::new(setup.expanded).expect("CPU backend");
    for prefix in [P * 32 + 1025 * 32 + 7, 0] {
        let mut selectors = vec![0; prefix * C];
        let mut masks = vec![0; prefix];
        let mut indices = vec![vec![None; ROWS]; C];
        for row in 0..prefix {
            selectors[row * C] = (row % 16) as u8;
            masks[row] = 1; // Distinguish committed zero from an absent zero.
            selectors[row * C + 1] = if row % 5 == 0 { 0 } else { 15 };
            for column in 0..C {
                let hot = selectors[row * C + column];
                if hot != 0 || (masks[row] >> column) & 1 != 0 {
                    indices[column][row] = Some(hot);
                }
            }
        }
        let polys = indices
            .into_iter()
            .map(|indices| OneHotPoly::<F, u8>::new(16, indices).unwrap())
            .collect::<Vec<_>>();
        let refs = polys.iter().collect::<Vec<_>>();
        let source_vars = (ROWS * 16).trailing_zeros() as usize;
        let point = (0..source_vars)
            .map(|index| -F::from_u64((index + 2) as u64))
            .collect::<Vec<_>>();
        // Both packing and no-packing limits, with nontrivial near-modulus
        // field values. Tile1024 and block boundaries occur within the prefix.
        for subring in [64, 512] {
            let geometry = SubringCoefficientPackingGeometry::try_new(1, D, subring).unwrap();
            let prepared = PreparedSubringCoefficientPackingPoint::new(
                geometry,
                BasisMode::Lagrange,
                P * B,
                P,
                source_vars,
                &point,
            )
            .unwrap();
            let tables = SubringCoefficientPackingTables::new(prepared).unwrap();
            let expected = <CpuBackend<F, F> as SubringCoefficientPackingBatchKernel<
                OneHotBatchView<'_, F, D, u8>,
                F,
                F,
                D,
            >>::coefficient_packing_partials_batch(
                &backend,
                None,
                <OneHotPoly<F, u8> as RootOpeningSource<F, D>>::opening_batch(&refs).unwrap(),
                SubringCoefficientPackingPlan { tables: &tables },
            )
            .unwrap();
            let shape = OneHotPackingShape {
                num_positions: P,
                blocks_per_column: B,
                num_columns: C,
                subring_dimension: subring,
            };
            let mut workspace = OneHotPackingWorkspace::new(
                &metal,
                shape,
                tables.position_weights(),
                tables.packing_weights(),
            )
            .unwrap();
            for (first, blocks) in [(0, 2), (2, 1)] {
                let start = (first * P * 32).min(prefix);
                let end = ((first + blocks) * P * 32).min(prefix);
                let sources = DeviceK16Sources::from_row_major(
                    &metal,
                    selectors[start * C..end * C].to_vec(),
                    &masks[start..end],
                    C,
                    blocks * P * 32,
                )
                .unwrap();
                workspace.accumulate(&sources, first).unwrap();
                assert!(workspace.accumulate(&sources, first).is_err());
                // The operation only borrows sources; the same owner remains
                // usable by the already independently checked root fold.
            }
            let got = workspace.read().unwrap();
            let expected = expected
                .iter()
                .flat_map(|column| column.coordinates().iter().copied())
                .collect::<Vec<_>>();
            assert_eq!(got, expected, "subring={subring}, prefix={prefix}");
        }
    }
}
