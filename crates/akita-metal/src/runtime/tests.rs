use super::*;

fn eval_flat_negacyclic_shift_sequence_into(coefficients: &[F], alpha: F, evaluations: &mut [F]) {
    assert_eq!(evaluations.len(), coefficients.len());
    let mut evaluation = F::zero();
    let mut power = F::one();
    for &coefficient in coefficients {
        evaluation += power * coefficient;
        power *= alpha;
    }
    let wrap_correction = power + F::one();
    for (output, &coefficient) in evaluations.iter_mut().zip(coefficients.iter().rev()) {
        *output = evaluation;
        evaluation = alpha * evaluation - wrap_correction * coefficient;
    }
}

#[test]
fn packed_d128_decompose_fold_matches_model() {
    // At K = 256, D = 128 one trace row spans two ring positions.
    const POSITIONS: usize = 10;
    const BLOCKS_PER_COLUMN: usize = 129;
    const COLUMNS: usize = 3;
    const CHALLENGE_WEIGHT: usize = 4;
    let runtime = MetalRuntime::new().unwrap();
    let num_rows = POSITIONS * BLOCKS_PER_COLUMN / 2;
    let lanes = (0..num_rows * COLUMNS)
        .map(|index| match index % 7 {
            0 => 0,
            1 => 255,
            2 => 128,
            _ => 1 + ((17 * index + 29) % 254) as u8,
        })
        .collect::<Vec<_>>();
    for embedding_stride in [1, 2] {
        let mut challenge_positions = Vec::new();
        let mut challenge_coefficients = Vec::new();
        let mut dense_challenges = vec![0i8; COLUMNS * BLOCKS_PER_COLUMN * 64];
        for challenge in 0..COLUMNS * BLOCKS_PER_COLUMN {
            for term in 0..CHALLENGE_WEIGHT {
                let position =
                    ((97 * challenge + 131 * term) % (128 / embedding_stride)) * embedding_stride;
                let coefficient = [1, -2, 3, -4][term];
                challenge_positions.push(position as u16);
                challenge_coefficients.push(coefficient);
                if embedding_stride == 2 {
                    dense_challenges[challenge * 64 + position / 2] = coefficient;
                }
            }
        }
        let params = PackedDecomposeFoldParams {
            onehot_k: 256,
            zero_suffix_start: num_rows as u64,
            num_rows: num_rows as u64,
            num_columns: COLUMNS as u64,
            lane_stride: COLUMNS as u64,
            num_positions: POSITIONS as u64,
            position_start: 0,
            blocks_per_column: BLOCKS_PER_COLUMN as u64,
            challenge_weight: CHALLENGE_WEIGHT as u64,
            output_coefficients: (POSITIONS * 128) as u64,
            zero_column_mask: 1,
        };
        let mut active_zero_rows = vec![0u64; num_rows.div_ceil(u64::BITS as usize)];
        active_zero_rows[0] = 0b1011;
        let mut streamed = Vec::new();
        let actual = runtime
            .dispatch_packed_fp128_decompose_fold_streaming(
                128,
                &lanes,
                &active_zero_rows,
                &challenge_positions,
                &challenge_coefficients,
                (embedding_stride == 2).then_some(dense_challenges.as_slice()),
                params,
                4,
                |_, coefficients| streamed.extend_from_slice(coefficients),
            )
            .unwrap()
            .centered_coefficients;
        let mut expected = vec![0i32; POSITIONS * 128];
        for position in 0..POSITIONS {
            for trace_block in 0..BLOCKS_PER_COLUMN {
                for column in 0..COLUMNS {
                    let ring = trace_block * POSITIONS + position;
                    let row = ring / 2;
                    let half = ring % 2;
                    let hot = usize::from(lanes[row * COLUMNS + column]);
                    let committed_zero = hot == 0
                        && column == 0
                        && active_zero_rows[row / u64::BITS as usize]
                            & (1u64 << (row % u64::BITS as usize))
                            != 0;
                    if (hot == 0 && !committed_zero) || hot / 128 != half {
                        continue;
                    }
                    let source_coefficient = hot % 128;
                    let challenge = column * BLOCKS_PER_COLUMN + trace_block;
                    let challenge_start = challenge * CHALLENGE_WEIGHT;
                    for term in 0..CHALLENGE_WEIGHT {
                        let mut destination = source_coefficient
                            + usize::from(challenge_positions[challenge_start + term]);
                        let mut value = i32::from(challenge_coefficients[challenge_start + term]);
                        if destination >= 128 {
                            destination -= 128;
                            value = -value;
                        }
                        expected[position * 128 + destination] += value;
                    }
                }
            }
        }
        assert_eq!(actual, expected);
        assert_eq!(streamed, expected);
    }
}

#[test]
fn packed_d512_decompose_fold_matches_cpu() {
    const POSITIONS: usize = 9;
    const BLOCKS_PER_COLUMN: usize = 2;
    const COLUMNS: usize = 3;
    const CHALLENGE_WEIGHT: usize = 4;
    let runtime = MetalRuntime::new().unwrap();
    let num_rows = 2 * POSITIONS * BLOCKS_PER_COLUMN;
    let lanes = (0..num_rows * COLUMNS)
        .map(|index| match index % 11 {
            0 => 0,
            1 => 255,
            _ => 1 + ((17 * index + 29) % 254) as u8,
        })
        .collect::<Vec<_>>();
    let mut challenge_positions = Vec::new();
    let mut challenge_coefficients = Vec::new();
    for challenge in 0..COLUMNS * BLOCKS_PER_COLUMN {
        for term in 0..CHALLENGE_WEIGHT {
            challenge_positions.push(((97 * challenge + 131 * term) % 512) as u16);
            challenge_coefficients.push([1, -2, 3, -4][term]);
        }
    }
    let params = PackedDecomposeFoldParams {
        onehot_k: 256,
        zero_suffix_start: num_rows as u64,
        num_rows: num_rows as u64,
        num_columns: COLUMNS as u64,
        lane_stride: COLUMNS as u64,
        num_positions: POSITIONS as u64,
        position_start: 0,
        blocks_per_column: BLOCKS_PER_COLUMN as u64,
        challenge_weight: CHALLENGE_WEIGHT as u64,
        output_coefficients: (POSITIONS * 512) as u64,
        zero_column_mask: 1,
    };
    let active_zero_rows = [1u64];
    let mut streamed = Vec::new();
    let mut chunk_starts = Vec::new();
    let actual = runtime
        .dispatch_packed_fp128_d512_decompose_fold_streaming(
            &lanes,
            &active_zero_rows,
            &challenge_positions,
            &challenge_coefficients,
            None,
            params,
            3,
            |position_start, coefficients| {
                chunk_starts.push(position_start);
                streamed.extend_from_slice(coefficients);
            },
        )
        .unwrap()
        .centered_coefficients;
    let mut expected = vec![0i32; POSITIONS * 512];
    for position in 0..POSITIONS {
        for trace_block in 0..BLOCKS_PER_COLUMN {
            for column in 0..COLUMNS {
                for row_in_ring in 0..2 {
                    let ring = trace_block * POSITIONS + position;
                    let row = 2 * ring + row_in_ring;
                    let hot = usize::from(lanes[row * COLUMNS + column]);
                    let committed_zero = hot == 0
                        && column == 0
                        && active_zero_rows[row / u64::BITS as usize]
                            & (1u64 << (row % u64::BITS as usize))
                            != 0;
                    if hot == 0 && !committed_zero {
                        continue;
                    }
                    let source_coefficient = row_in_ring * 256 + hot;
                    let challenge = column * BLOCKS_PER_COLUMN + trace_block;
                    let challenge_start = challenge * CHALLENGE_WEIGHT;
                    for term in 0..CHALLENGE_WEIGHT {
                        let mut destination = source_coefficient
                            + usize::from(challenge_positions[challenge_start + term]);
                        let mut value = i32::from(challenge_coefficients[challenge_start + term]);
                        if destination >= 512 {
                            destination -= 512;
                            value = -value;
                        }
                        expected[position * 512 + destination] += value;
                    }
                }
            }
        }
    }
    assert_eq!(actual, expected);
    assert_eq!(streamed, expected);
    assert_eq!(chunk_starts, [0, 3, 6]);
}

#[test]
fn packed_d512_subring64_decompose_fold_routes_match_cpu() {
    const POSITIONS: usize = 9;
    const BLOCKS_PER_COLUMN: usize = 2;
    const COLUMNS: usize = 129;
    const CHALLENGE_WEIGHT: usize = 4;
    let runtime = MetalRuntime::new().unwrap();
    let num_rows = 2 * POSITIONS * BLOCKS_PER_COLUMN;
    let lanes = (0..num_rows * COLUMNS)
        .map(|index| 1 + (8 * ((17 * index + 29) % 32)) as u8)
        .collect::<Vec<_>>();
    let mut challenge_positions = Vec::new();
    let mut challenge_coefficients = Vec::new();
    let mut dense_challenges = vec![0i8; COLUMNS * BLOCKS_PER_COLUMN * 64];
    for challenge in 0..COLUMNS * BLOCKS_PER_COLUMN {
        for term in 0..CHALLENGE_WEIGHT {
            let subring_position = (13 * challenge + 17 * term) % 64;
            let coefficient = [1, -2, 2, -1][term];
            challenge_positions.push((subring_position * 8) as u16);
            challenge_coefficients.push(coefficient);
            dense_challenges[challenge * 64 + subring_position] = coefficient;
        }
    }
    let params = PackedDecomposeFoldParams {
        onehot_k: 256,
        zero_suffix_start: num_rows as u64,
        num_rows: num_rows as u64,
        num_columns: COLUMNS as u64,
        lane_stride: COLUMNS as u64,
        num_positions: POSITIONS as u64,
        position_start: 0,
        blocks_per_column: BLOCKS_PER_COLUMN as u64,
        challenge_weight: CHALLENGE_WEIGHT as u64,
        output_coefficients: (POSITIONS * 512) as u64,
        zero_column_mask: 0,
    };
    let actual = runtime
        .dispatch_packed_fp128_d512_decompose_fold_streaming(
            &lanes,
            &[],
            &challenge_positions,
            &challenge_coefficients,
            Some(&dense_challenges),
            params,
            POSITIONS,
            |_, _| {},
        )
        .unwrap()
        .centered_coefficients;
    let tasks_per_position = BLOCKS_PER_COLUMN * COLUMNS * 2;
    let tiles_per_position = tasks_per_position.div_ceil(256);
    let fold_params = PackedFoldIndexParams {
        num_rows: num_rows as u64,
        num_columns: COLUMNS as u64,
        lane_stride: COLUMNS as u64,
        num_positions: POSITIONS as u64,
        position_start: 0,
        blocks_per_column: BLOCKS_PER_COLUMN as u64,
        tasks_per_position: tasks_per_position as u64,
        tiles_per_position: tiles_per_position as u64,
        record_slots: (POSITIONS * tiles_per_position * 256) as u64,
        count_entries: (POSITIONS * tiles_per_position * 8) as u64,
        output_coefficients: (POSITIONS * 512) as u64,
        fold_digits: 0,
        fold_log_basis: 0,
    };
    let index = runtime
        .prepare_packed_fp128_d512_fold_index(&lanes, fold_params)
        .unwrap();
    let mut indexed_streamed = Vec::new();
    let mut indexed_digits = Vec::new();
    let indexed = runtime
        .dispatch_packed_fp128_d512_subring64_decompose_fold_streaming(
            &lanes,
            &dense_challenges,
            PackedFp128D512FoldSource::Retained(&index),
            params,
            4,
            4,
            4,
            |_, coefficients, digits| {
                indexed_streamed.extend_from_slice(coefficients);
                indexed_digits.extend_from_slice(digits);
            },
        )
        .unwrap()
        .centered_coefficients;
    let mut fused_streamed = Vec::new();
    let mut fused_digits = Vec::new();
    let mut fused_chunk_starts = Vec::new();
    let fused = runtime
        .dispatch_packed_fp128_d512_subring64_decompose_fold_streaming(
            &lanes,
            &dense_challenges,
            PackedFp128D512FoldSource::Fused(fold_params),
            params,
            4,
            4,
            4,
            |position_start, coefficients, digits| {
                fused_chunk_starts.push(position_start);
                fused_streamed.extend_from_slice(coefficients);
                fused_digits.extend_from_slice(digits);
            },
        )
        .unwrap()
        .centered_coefficients;

    let mut expected = vec![0i32; POSITIONS * 512];
    for position in 0..POSITIONS {
        for trace_block in 0..BLOCKS_PER_COLUMN {
            for column in 0..COLUMNS {
                for row_in_ring in 0..2 {
                    let ring = trace_block * POSITIONS + position;
                    let row = 2 * ring + row_in_ring;
                    let hot = usize::from(lanes[row * COLUMNS + column]);
                    if hot == 0 {
                        continue;
                    }
                    let source_coefficient = row_in_ring * 256 + hot;
                    let challenge = column * BLOCKS_PER_COLUMN + trace_block;
                    let challenge_start = challenge * CHALLENGE_WEIGHT;
                    for term in 0..CHALLENGE_WEIGHT {
                        let mut destination = source_coefficient
                            + usize::from(challenge_positions[challenge_start + term]);
                        let mut value = i32::from(challenge_coefficients[challenge_start + term]);
                        if destination >= 512 {
                            destination -= 512;
                            value = -value;
                        }
                        expected[position * 512 + destination] += value;
                    }
                }
            }
        }
    }
    assert_eq!(actual, expected);
    assert_eq!(indexed, expected);
    assert_eq!(indexed_streamed, expected);
    assert_eq!(fused, expected);
    assert_eq!(fused_streamed, expected);
    assert_eq!(fused_chunk_starts, [0, 4, 8]);
    for digits in [&indexed_digits, &fused_digits] {
        for position in 0..POSITIONS {
            for coefficient in 0..512 {
                let reconstructed = (0..4).rev().fold(0i32, |value, digit| {
                    let value_index = position * 4 * 512 + digit * 512 + coefficient;
                    let value_digit = digits[value_index];
                    assert!((-8..8).contains(&value_digit));
                    value * 16 + i32::from(value_digit)
                });
                assert_eq!(reconstructed, expected[position * 512 + coefficient]);
            }
        }
    }
}

#[test]
fn reduced_linear_sources_match_cpu_recurrence() {
    const D: usize = 64;
    let runtime = MetalRuntime::new().unwrap();
    let alpha = F::from_i64(7);
    let row_weights = [F::from_i64(3), F::from_i64(-5)];
    let matrix = (0..2 * D)
        .map(|index| F::from_i64((index as i64 % 19) - 9))
        .collect::<Vec<_>>();
    let matrix_limbs = matrix
        .iter()
        .copied()
        .map(Fp128Limbs::from_field)
        .collect::<Vec<_>>();
    let matrix_buffer = runtime.shared_buffer_from_slice(&matrix_limbs).unwrap();
    let mut power = F::one();
    let mut alpha_powers = Vec::with_capacity(D);
    for _ in 0..D {
        alpha_powers.push(Fp128Limbs::from_field(power));
        power *= alpha;
    }
    let alpha_limbs = Fp128Limbs::from_field(alpha);
    let wrap_correction = Fp128Limbs::from_field(power + F::one());
    let sources = vec![
        DirectRelationLinearSourceInput::ReducedSetup {
            matrix: matrix_buffer,
            ring_dimension: D,
            row_count: 2,
            column_count: 1,
            row_weights: row_weights
                .into_iter()
                .map(Fp128Limbs::from_field)
                .collect(),
            alpha_powers: alpha_powers.clone(),
            alpha: alpha_limbs,
            wrap_correction,
        },
        DirectRelationLinearSourceInput::ReducedSparse {
            ring_dimension: D,
            challenge_count: 1,
            term_offsets: vec![0, 3],
            positions: vec![0, 11, 63],
            coefficients: vec![1, -2, 3],
            alpha_powers,
            alpha: alpha_limbs,
            wrap_correction,
        },
    ];
    let segments = [
        DirectRelationLinearSegment {
            factor: Fp128Limbs::from_field(F::one()),
            source_index: 0,
            target_lane_start: 0,
            target_lane_stride: 1,
            source_lane_start: 0,
            source_lane_stride: 1,
            lane_count: 1,
        },
        DirectRelationLinearSegment {
            factor: Fp128Limbs::from_field(F::one()),
            source_index: 1,
            target_lane_start: 0,
            target_lane_stride: 1,
            source_lane_start: 0,
            source_lane_stride: 1,
            lane_count: 1,
        },
    ];
    let (session, _) = runtime
        .begin_fp128_direct_relation(
            &[0i8; D],
            D,
            3,
            D.trailing_zeros() as usize,
            &[Fp128Limbs::from_field(F::zero())],
            &segments,
            &[0, 2],
            &[0, 1],
            &sources,
            &[],
        )
        .unwrap();
    let output = runtime
        .shared_buffer(2 * D * size_of::<Fp128Limbs>())
        .unwrap();
    let command = runtime.queue.new_command_buffer();
    let encoder = command.new_blit_command_encoder();
    encoder.copy_from_buffer(
        &session.linear_tables[0],
        0,
        &output,
        0,
        (2 * D * size_of::<Fp128Limbs>()) as u64,
    );
    encoder.end_encoding();
    complete_command(command).unwrap();
    let actual =
        unsafe { std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), 2 * D) }
            .iter()
            .enumerate()
            .map(|(index, value)| value.into_field(index).unwrap())
            .collect::<Vec<_>>();

    let combined = (0..D)
        .map(|coefficient| {
            row_weights[0] * matrix[coefficient] + row_weights[1] * matrix[D + coefficient]
        })
        .collect::<Vec<_>>();
    let mut expected_setup = vec![F::zero(); D];
    eval_flat_negacyclic_shift_sequence_into(&combined, alpha, &mut expected_setup);
    let mut sparse = vec![F::zero(); D];
    sparse[0] = F::from_i64(1);
    sparse[11] = F::from_i64(-2);
    sparse[63] = F::from_i64(3);
    let mut expected_sparse = vec![F::zero(); D];
    eval_flat_negacyclic_shift_sequence_into(&sparse, alpha, &mut expected_sparse);
    assert_eq!(actual[..D], expected_setup);
    assert_eq!(actual[D..], expected_sparse);
}
