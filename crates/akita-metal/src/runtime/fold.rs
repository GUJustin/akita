use super::*;

impl MetalRuntime {
    pub(crate) fn dispatch_packed_fp128_d512_decompose_fold_streaming(
        &self,
        lanes: &[u8],
        active_zero_rows: &[u64],
        challenge_positions: &[u16],
        challenge_coefficients: &[i8],
        dense_subring64_challenges: Option<&[i8]>,
        params: PackedDecomposeFoldParams,
        position_chunk_len: usize,
        consume: impl FnMut(usize, &[i32]),
    ) -> Result<PackedDecomposeFoldDispatchOutcome, MetalCommitError> {
        self.dispatch_packed_fp128_decompose_fold_streaming(
            512,
            lanes,
            active_zero_rows,
            challenge_positions,
            challenge_coefficients,
            dense_subring64_challenges,
            params,
            position_chunk_len,
            consume,
        )
    }

    /// Packed decompose-fold at ring dimension 512 (rank-1 row) or 128
    /// (rank-3 row), with sparse or embedded subring-64 challenges.
    pub(crate) fn dispatch_packed_fp128_decompose_fold_streaming(
        &self,
        ring_d: usize,
        lanes: &[u8],
        active_zero_rows: &[u64],
        challenge_positions: &[u16],
        challenge_coefficients: &[i8],
        dense_subring64_challenges: Option<&[i8]>,
        params: PackedDecomposeFoldParams,
        position_chunk_len: usize,
        mut consume: impl FnMut(usize, &[i32]),
    ) -> Result<PackedDecomposeFoldDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let dense = dense_subring64_challenges.is_some();
            let (fold_pipeline, fold_threads) = match (ring_d, dense) {
                (512, false) => (&self.fp128_d512_decompose_fold_pipeline, 256u64),
                (128, false) => (&self.fp128_d128_decompose_fold_pipeline, 128u64),
                (512, true) => (&self.fp128_d512_subring64_decompose_fold_pipeline, 256u64),
                (128, true) => (&self.fp128_d128_subring64_decompose_fold_pipeline, 256u64),
                _ => {
                    return Err(MetalCommitError::UnsupportedShape(
                        "packed decompose-fold supports ring dimensions 512 and 128".into(),
                    ));
                }
            };
            let ring_d_u64 = ring_d as u64;
            let expected_challenge_terms = params
                .num_columns
                .checked_mul(params.blocks_per_column)
                .and_then(|count| count.checked_mul(params.challenge_weight))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "decompose-fold challenge terms",
                ))?;
            let expected_lanes = params
                .num_rows
                .checked_mul(params.lane_stride)
                .ok_or(MetalCommitError::ShapeOverflow("decompose-fold lanes"))?;
            let expected_output = params
                .num_positions
                .checked_mul(ring_d_u64)
                .ok_or(MetalCommitError::ShapeOverflow("decompose-fold output"))?;
            let expected_dense_challenges = params
                .num_columns
                .checked_mul(params.blocks_per_column)
                .and_then(|count| count.checked_mul(64))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "dense subring challenge coefficients",
                ))?;
            let expected_active_zero_words = params.num_rows.div_ceil(u64::BITS.into());
            let live_column_mask = if params.num_columns >= u64::BITS.into() {
                u64::MAX
            } else {
                (1u64 << params.num_columns) - 1
            };
            if !matches!((ring_d, params.onehot_k), (512, 16 | 256) | (128, 256))
                || params.zero_suffix_start > params.num_rows
                || params.num_rows == 0
                || params.num_columns == 0
                || params.num_columns > params.lane_stride
                || (ring_d == 128 && dense && params.num_columns > 32)
                || params.zero_column_mask & !live_column_mask != 0
                || (params.zero_column_mask == 0 && !active_zero_rows.is_empty())
                || (params.zero_column_mask != 0
                    && u64::try_from(active_zero_rows.len()).ok()
                        != Some(expected_active_zero_words))
                || params.challenge_weight == 0
                || params.position_start != 0
                || position_chunk_len == 0
                || params.output_coefficients != expected_output
                || u64::try_from(lanes.len()).ok() != Some(expected_lanes)
                || u64::try_from(challenge_positions.len()).ok() != Some(expected_challenge_terms)
                || challenge_positions.len() != challenge_coefficients.len()
                || dense_subring64_challenges.is_some_and(|dense| {
                    u64::try_from(dense.len()).ok() != Some(expected_dense_challenges)
                        || expected_dense_challenges > u64::from(u32::MAX) + 1
                })
                || params.num_positions > u64::from(u32::MAX)
                || fold_pipeline.max_total_threads_per_threadgroup() < fold_threads
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "fp128 packed decompose-fold geometry is unsupported".into(),
                ));
            }

            let buffer_start = Instant::now();
            let lane_buffer = self.packed_lane_buffer(lanes)?;
            let no_active_zero_rows = [0u64];
            let active_zero_rows = if active_zero_rows.is_empty() {
                no_active_zero_rows.as_slice()
            } else {
                active_zero_rows
            };
            let active_zero_buffer = self.shared_buffer_from_slice(active_zero_rows)?;
            let positions = self.shared_buffer_from_slice(challenge_positions)?;
            let coefficients = self.shared_buffer_from_slice(challenge_coefficients)?;
            let dense_challenges = dense_subring64_challenges
                .map(|dense| self.shared_buffer_from_slice(dense))
                .transpose()?;
            let output_count = usize::try_from(params.output_coefficients)
                .map_err(|_| MetalCommitError::ShapeOverflow("decompose-fold output count"))?;
            let output_bytes = output_count.checked_mul(size_of::<i32>()).ok_or(
                MetalCommitError::ShapeOverflow("decompose-fold output bytes"),
            )?;
            let mut centered_coefficients = vec![0i32; output_count];
            let output_zero_copy = centered_coefficients
                .as_ptr()
                .addr()
                .is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT)
                && output_bytes.is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT);
            let output = if output_zero_copy {
                self.device.new_buffer_with_bytes_no_copy(
                    centered_coefficients.as_mut_ptr().cast::<c_void>(),
                    output_bytes as u64,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
            } else {
                self.shared_buffer(output_bytes)?
            };
            let buffer_setup = buffer_start.elapsed();

            let total_positions = usize::try_from(params.num_positions)
                .map_err(|_| MetalCommitError::ShapeOverflow("decompose-fold position count"))?;
            let chunk_len = position_chunk_len.min(total_positions);
            let command_start = Instant::now();
            let mut commands = Vec::with_capacity(total_positions.div_ceil(chunk_len));
            for position_start in (0..total_positions).step_by(chunk_len) {
                let position_end = position_start
                    .saturating_add(chunk_len)
                    .min(total_positions);
                let position_count = position_end - position_start;
                let mut command_params = params;
                command_params.position_start = u64::try_from(position_start).map_err(|_| {
                    MetalCommitError::ShapeOverflow("decompose-fold position offset")
                })?;
                let output_offset = position_start
                    .checked_mul(ring_d)
                    .and_then(|count| count.checked_mul(size_of::<i32>()))
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "decompose-fold output offset",
                    ))?;
                let command = self.queue.new_command_buffer();
                command.set_label("Akita fp128 packed decompose-fold");
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita fp128 packed decompose-fold");
                encoder.set_compute_pipeline_state(fold_pipeline);
                if let Some(dense_challenges) = dense_challenges.as_ref() {
                    encoder.set_buffer(0, Some(&lane_buffer.buffer), 0);
                    encoder.set_buffer(1, Some(dense_challenges), 0);
                    encoder.set_buffer(2, Some(&output), output_offset as u64);
                    set_inline_bytes(encoder, 3, &command_params);
                    encoder.set_buffer(4, Some(&active_zero_buffer), 0);
                } else {
                    encoder.set_buffer(0, Some(&lane_buffer.buffer), 0);
                    encoder.set_buffer(1, Some(&positions), 0);
                    encoder.set_buffer(2, Some(&coefficients), 0);
                    encoder.set_buffer(3, Some(&output), output_offset as u64);
                    set_inline_bytes(encoder, 4, &command_params);
                    encoder.set_buffer(5, Some(&active_zero_buffer), 0);
                }
                encoder.dispatch_thread_groups(
                    MTLSize::new(position_count as u64, 1, 1),
                    MTLSize::new(fold_threads, 1, 1),
                );
                encoder.end_encoding();
                command.commit();
                commands.push((command, position_start, position_end));
            }

            let mut readback_copy = Duration::ZERO;
            let mut consumer_time = Duration::ZERO;
            for (command, position_start, position_end) in &commands {
                command.wait_until_completed();
                validate_completed_command(command)?;
                let coefficient_start = position_start * ring_d;
                let coefficient_end = position_end * ring_d;
                if !output_zero_copy {
                    let readback_start = Instant::now();
                    // SAFETY: this completed command initialized the disjoint
                    // coefficient range copied into equally sized host storage.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            output.contents().cast::<i32>().add(coefficient_start),
                            centered_coefficients.as_mut_ptr().add(coefficient_start),
                            coefficient_end - coefficient_start,
                        );
                    }
                    readback_copy += readback_start.elapsed();
                }
                let consumer_start = Instant::now();
                consume(
                    *position_start,
                    &centered_coefficients[coefficient_start..coefficient_end],
                );
                consumer_time += consumer_start.elapsed();
            }
            let command_wall = command_start.elapsed();
            let gpu = commands.first().and_then(|(first, _, _)| {
                commands
                    .last()
                    .and_then(|(last, _, _)| completed_commands_gpu_span(first, last))
            });
            let allocation_bytes = output_bytes
                .checked_add(size_of_val(challenge_positions))
                .and_then(|bytes| bytes.checked_add(size_of_val(challenge_coefficients)))
                .and_then(|bytes| bytes.checked_add(size_of_val(active_zero_rows)))
                .and_then(|bytes| {
                    bytes.checked_add(dense_subring64_challenges.map_or(0, size_of_val))
                })
                .ok_or(MetalCommitError::ShapeOverflow(
                    "decompose-fold allocation bytes",
                ))?;
            Ok(PackedDecomposeFoldDispatchOutcome {
                centered_coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                consumer_time,
                allocation_bytes,
            })
        })
    }

    pub(crate) fn prepare_packed_fp128_d512_fold_index(
        &self,
        lanes: &[u8],
        params: PackedFoldIndexParams,
    ) -> Result<PackedFp128D512FoldIndex, MetalCommitError> {
        autoreleasepool(|| {
            validate_packed_fold_index_geometry(params, lanes.len())?;
            if params.fold_digits != 0
                || params.fold_log_basis != 0
                || self
                    .fp128_d512_build_fold_index_pipeline
                    .max_total_threads_per_threadgroup()
                    < FP128_D512_FOLD_INDEX_TILE_TASKS as u64
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "fp128 D512 packed fold-index geometry is unsupported".into(),
                ));
            }

            let buffer_start = Instant::now();
            let lane_buffer = self.packed_lane_buffer(lanes)?;
            let record_count = usize::try_from(params.record_slots)
                .map_err(|_| MetalCommitError::ShapeOverflow("fold-index record count"))?;
            let record_bytes = record_count
                .checked_mul(size_of::<u32>())
                .ok_or(MetalCommitError::ShapeOverflow("fold-index record bytes"))?;
            let count_count = usize::try_from(params.count_entries)
                .map_err(|_| MetalCommitError::ShapeOverflow("fold-index count count"))?;
            let count_bytes = count_count
                .checked_mul(size_of::<u16>())
                .ok_or(MetalCommitError::ShapeOverflow("fold-index count bytes"))?;
            let records = self.private_buffer(record_bytes)?;
            let counts = self.private_buffer(count_bytes)?;
            let buffer_setup = buffer_start.elapsed();

            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 D512 packed fold index");
            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita fp128 D512 packed fold index");
            encoder.set_compute_pipeline_state(&self.fp128_d512_build_fold_index_pipeline);
            encoder.set_buffer(0, Some(&lane_buffer.buffer), 0);
            encoder.set_buffer(1, Some(&records), 0);
            encoder.set_buffer(2, Some(&counts), 0);
            set_inline_bytes(encoder, 3, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.num_positions, 1, 1),
                MTLSize::new(FP128_D512_FOLD_INDEX_TILE_TASKS as u64, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;
            let allocation_bytes =
                record_bytes
                    .checked_add(count_bytes)
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "fold-index allocation bytes",
                    ))?;
            Ok(PackedFp128D512FoldIndex {
                records,
                counts,
                params,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: Duration::ZERO,
                },
                allocation_bytes,
            })
        })
    }

    pub(crate) fn prepare_packed_fp128_d512_coefficient_packing_index(
        &self,
        lanes: &[u8],
        params: PackedCoefficientPackingIndexParams,
    ) -> Result<PackedFp128D512CoefficientPackingIndex, MetalCommitError> {
        autoreleasepool(|| {
            let expected_lanes = params
                .num_rows
                .checked_mul(params.lane_stride)
                .ok_or(MetalCommitError::ShapeOverflow("packing-index lanes"))?;
            let expected_tiles = params
                .num_positions
                .div_ceil(FP128_D512_PACKING_INDEX_TILE_POSITIONS as u64);
            let expected_streams = params
                .blocks_per_column
                .checked_mul(params.num_columns)
                .and_then(|count| count.checked_mul(2))
                .ok_or(MetalCommitError::ShapeOverflow("packing-index streams"))?;
            let expected_layouts = expected_streams
                .checked_mul(expected_tiles)
                .ok_or(MetalCommitError::ShapeOverflow("packing-index layouts"))?;
            let expected_records = expected_layouts
                .checked_mul(FP128_D512_PACKING_INDEX_TILE_POSITIONS as u64)
                .ok_or(MetalCommitError::ShapeOverflow("packing-index records"))?;
            let expected_offsets = expected_layouts
                .checked_mul(FP128_D512_PACKING_INDEX_BUCKET_OFFSETS as u64)
                .ok_or(MetalCommitError::ShapeOverflow("packing-index offsets"))?;
            let expected_groups = params
                .blocks_per_column
                .checked_mul(expected_tiles)
                .ok_or(MetalCommitError::ShapeOverflow("packing-index groups"))?;
            if params.num_rows == 0
                || params.num_columns == 0
                || params.num_columns > 32
                || params.num_columns > params.lane_stride
                || params.num_positions == 0
                || params.blocks_per_column == 0
                || params.position_tiles != expected_tiles
                || params.record_slots != expected_records
                || params.offset_entries != expected_offsets
                || u64::try_from(lanes.len()).ok() != Some(expected_lanes)
                || expected_groups > u64::from(u32::MAX)
                || self
                    .fp128_d512_build_coefficient_packing_index_pipeline
                    .max_total_threads_per_threadgroup()
                    < FP128_D512_PACKING_INDEX_TILE_POSITIONS as u64
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "fp128 D512 coefficient-packing index geometry is unsupported".into(),
                ));
            }

            let buffer_start = Instant::now();
            let lane_buffer = self.packed_lane_buffer(lanes)?;
            let record_count = usize::try_from(params.record_slots)
                .map_err(|_| MetalCommitError::ShapeOverflow("packing-index record count"))?;
            let record_bytes = record_count.checked_mul(size_of::<u16>()).ok_or(
                MetalCommitError::ShapeOverflow("packing-index record bytes"),
            )?;
            let offset_count = usize::try_from(params.offset_entries)
                .map_err(|_| MetalCommitError::ShapeOverflow("packing-index offset count"))?;
            let offset_bytes = offset_count.checked_mul(size_of::<u16>()).ok_or(
                MetalCommitError::ShapeOverflow("packing-index offset bytes"),
            )?;
            let records = self.private_buffer(record_bytes)?;
            let offsets = self.private_buffer(offset_bytes)?;
            let buffer_setup = buffer_start.elapsed();

            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 D512 coefficient-packing index");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(
                &self.fp128_d512_build_coefficient_packing_index_pipeline,
            );
            encoder.set_buffer(0, Some(&lane_buffer.buffer), 0);
            encoder.set_buffer(1, Some(&records), 0);
            encoder.set_buffer(2, Some(&offsets), 0);
            set_inline_bytes(encoder, 3, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(expected_groups, 1, 1),
                MTLSize::new(FP128_D512_PACKING_INDEX_TILE_POSITIONS as u64, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;
            Ok(PackedFp128D512CoefficientPackingIndex {
                records,
                offsets,
                params,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: Duration::ZERO,
                },
                allocation_bytes: record_bytes.checked_add(offset_bytes).ok_or(
                    MetalCommitError::ShapeOverflow("packing-index allocation bytes"),
                )?,
            })
        })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the packed root dispatch keeps source, challenge, digit, and streaming boundaries explicit"
    )]
    pub(crate) fn dispatch_packed_fp128_d512_subring64_decompose_fold_streaming(
        &self,
        lanes: &[u8],
        dense_subring64_challenges: &[i8],
        source: PackedFp128D512FoldSource<'_>,
        params: PackedDecomposeFoldParams,
        fold_digits: usize,
        fold_log_basis: u32,
        position_chunk_len: usize,
        mut consume: impl FnMut(usize, &[i32], &[i8]),
    ) -> Result<PackedDecomposeFoldDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let expected_dense = params
                .num_columns
                .checked_mul(params.blocks_per_column)
                .and_then(|count| count.checked_mul(64))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "indexed fold dense challenges",
                ))?;
            let expected_output = params
                .num_positions
                .checked_mul(512)
                .ok_or(MetalCommitError::ShapeOverflow("indexed fold output"))?;
            let index_params = match source {
                PackedFp128D512FoldSource::Retained(index) => index.params,
                PackedFp128D512FoldSource::Fused(params) => params,
            };
            validate_packed_fold_index_geometry(index_params, lanes.len())?;
            let max_threads = match source {
                PackedFp128D512FoldSource::Retained(_) => self
                    .fp128_d512_indexed_subring64_decompose_fold_pipeline
                    .max_total_threads_per_threadgroup(),
                PackedFp128D512FoldSource::Fused(_) => self
                    .fp128_d512_fused_subring64_decompose_fold_pipeline
                    .max_total_threads_per_threadgroup(),
            };
            if params.num_rows == 0
                || params.num_columns == 0
                || params.num_columns > params.lane_stride
                || params.position_start != 0
                || params.output_coefficients != expected_output
                || fold_digits == 0
                || !(1..=8).contains(&fold_log_basis)
                || position_chunk_len == 0
                || u64::try_from(dense_subring64_challenges.len()).ok() != Some(expected_dense)
                || index_params.num_rows != params.num_rows
                || index_params.num_columns != params.num_columns
                || index_params.lane_stride != params.lane_stride
                || index_params.num_positions != params.num_positions
                || index_params.blocks_per_column != params.blocks_per_column
                || max_threads < 256
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "fp128 D512 indexed fold geometry is unsupported".into(),
                ));
            }

            let buffer_start = Instant::now();
            let lane_buffer = match source {
                PackedFp128D512FoldSource::Retained(_) => None,
                PackedFp128D512FoldSource::Fused(_) => Some(self.packed_lane_buffer(lanes)?),
            };
            let packed_challenges = pack_biased_subring64_challenges(dense_subring64_challenges)?;
            let packed_challenges_buffer = self.shared_buffer_from_slice(&packed_challenges)?;
            let output_count = usize::try_from(params.output_coefficients)
                .map_err(|_| MetalCommitError::ShapeOverflow("indexed fold output count"))?;
            let output_bytes = output_count
                .checked_mul(size_of::<i32>())
                .ok_or(MetalCommitError::ShapeOverflow("indexed fold output bytes"))?;
            let digit_bytes = output_count
                .checked_mul(fold_digits)
                .ok_or(MetalCommitError::ShapeOverflow("indexed fold digit bytes"))?;
            let mut centered_coefficients = vec![0i32; output_count];
            let output_zero_copy = centered_coefficients
                .as_ptr()
                .addr()
                .is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT)
                && output_bytes.is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT);
            let output = if output_zero_copy {
                self.device.new_buffer_with_bytes_no_copy(
                    centered_coefficients.as_mut_ptr().cast::<c_void>(),
                    output_bytes as u64,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
            } else {
                self.shared_buffer(output_bytes)?
            };
            let digits = self.shared_buffer(digit_bytes)?;
            let buffer_setup = buffer_start.elapsed();

            let total_positions = usize::try_from(params.num_positions)
                .map_err(|_| MetalCommitError::ShapeOverflow("indexed fold positions"))?;
            let chunk_len = position_chunk_len.min(total_positions);
            let command_start = Instant::now();
            let mut commands = Vec::with_capacity(total_positions.div_ceil(chunk_len));
            for position_start in (0..total_positions).step_by(chunk_len) {
                let position_end = position_start
                    .saturating_add(chunk_len)
                    .min(total_positions);
                let position_count = position_end - position_start;
                let mut command_params = index_params;
                command_params.position_start = u64::try_from(position_start)
                    .map_err(|_| MetalCommitError::ShapeOverflow("indexed fold position offset"))?;
                command_params.output_coefficients = params.output_coefficients;
                command_params.fold_digits = u64::try_from(fold_digits)
                    .map_err(|_| MetalCommitError::ShapeOverflow("indexed fold digit count"))?;
                command_params.fold_log_basis = u64::from(fold_log_basis);
                let output_offset = position_start
                    .checked_mul(512)
                    .and_then(|count| count.checked_mul(size_of::<i32>()))
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "indexed fold output offset",
                    ))?;
                let digit_offset = position_start
                    .checked_mul(512)
                    .and_then(|count| count.checked_mul(fold_digits))
                    .ok_or(MetalCommitError::ShapeOverflow("indexed fold digit offset"))?;
                let command = self.queue.new_command_buffer();
                command.set_label("Akita fp128 D512 packed subring64 decompose-fold");
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita fp128 D512 packed subring64 decompose-fold");
                match source {
                    PackedFp128D512FoldSource::Retained(index) => {
                        encoder.set_compute_pipeline_state(
                            &self.fp128_d512_indexed_subring64_decompose_fold_pipeline,
                        );
                        encoder.set_buffer(0, Some(&index.records), 0);
                        encoder.set_buffer(1, Some(&index.counts), 0);
                        encoder.set_buffer(2, Some(&packed_challenges_buffer), 0);
                        encoder.set_buffer(3, Some(&output), output_offset as u64);
                        encoder.set_buffer(4, Some(&digits), digit_offset as u64);
                        set_inline_bytes(encoder, 5, &command_params);
                    }
                    PackedFp128D512FoldSource::Fused(_) => {
                        let lane_buffer = lane_buffer.as_ref().ok_or_else(|| {
                            MetalCommitError::UnsupportedShape(
                                "fused packed fold is missing its lane buffer".into(),
                            )
                        })?;
                        encoder.set_compute_pipeline_state(
                            &self.fp128_d512_fused_subring64_decompose_fold_pipeline,
                        );
                        encoder.set_buffer(0, Some(&lane_buffer.buffer), 0);
                        encoder.set_buffer(1, Some(&packed_challenges_buffer), 0);
                        encoder.set_buffer(2, Some(&output), output_offset as u64);
                        encoder.set_buffer(3, Some(&digits), digit_offset as u64);
                        set_inline_bytes(encoder, 4, &command_params);
                    }
                }
                encoder.dispatch_thread_groups(
                    MTLSize::new(position_count as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                encoder.end_encoding();
                command.commit();
                commands.push((command, position_start, position_end));
            }

            let mut readback_copy = Duration::ZERO;
            let mut consumer_time = Duration::ZERO;
            for (command, position_start, position_end) in &commands {
                command.wait_until_completed();
                validate_completed_command(command)?;
                let coefficient_start = position_start * 512;
                let coefficient_end = position_end * 512;
                if !output_zero_copy {
                    let readback_start = Instant::now();
                    // SAFETY: this completed command initialized the disjoint output range.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            output.contents().cast::<i32>().add(coefficient_start),
                            centered_coefficients.as_mut_ptr().add(coefficient_start),
                            coefficient_end - coefficient_start,
                        );
                    }
                    readback_copy += readback_start.elapsed();
                }
                let digit_start = coefficient_start * fold_digits;
                let digit_end = coefficient_end * fold_digits;
                // SAFETY: this completed command initialized the matching digit range.
                let digit_slice = unsafe {
                    std::slice::from_raw_parts(
                        digits.contents().cast::<i8>().add(digit_start),
                        digit_end - digit_start,
                    )
                };
                let consumer_start = Instant::now();
                consume(
                    *position_start,
                    &centered_coefficients[coefficient_start..coefficient_end],
                    digit_slice,
                );
                consumer_time += consumer_start.elapsed();
            }
            let command_wall = command_start.elapsed();
            let gpu = commands.first().and_then(|(first, _, _)| {
                commands
                    .last()
                    .and_then(|(last, _, _)| completed_commands_gpu_span(first, last))
            });
            let allocation_bytes = output_bytes
                .checked_add(digit_bytes)
                .and_then(|bytes| bytes.checked_add(size_of_val(packed_challenges.as_slice())))
                .and_then(|bytes| {
                    bytes.checked_add(
                        lane_buffer
                            .as_ref()
                            .map_or(0, |buffer| usize::from(!buffer.zero_copy) * lanes.len()),
                    )
                })
                .ok_or(MetalCommitError::ShapeOverflow(
                    "indexed fold allocation bytes",
                ))?;
            Ok(PackedDecomposeFoldDispatchOutcome {
                centered_coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                consumer_time,
                allocation_bytes,
            })
        })
    }
}
