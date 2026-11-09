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
