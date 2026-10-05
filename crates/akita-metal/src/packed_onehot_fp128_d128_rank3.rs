//! Packed fp128 D128 rank-3 trace commitment.
//!
//! Companion to the D512 panels path for the K=256 catalog row that uses a
//! 128-dimensional inner ring at rank 3. The accumulator volume per hot entry
//! is 3 x 128 x 128 bits, three quarters of the D512 rank-1 row, and the
//! kernel streams one ring element's rows per tile so the matrix traffic
//! shrinks by the same factor (see `specs/akita-metal-d128-rank3-root-floor.md`
//! in Jolt).

use std::mem::size_of;
use std::time::{Duration, Instant};

use akita_error::AkitaError;
use akita_prover::compute::CommitInnerPlan;
use akita_prover::CommitInnerWitness;
use akita_types::RingVec;

use crate::backend::{MetalBackend, MetalCommitMetrics};
use crate::field::{Fp128Limbs, MetalField, F};
use crate::onehot::to_u64;
use crate::packed_onehot::PackedOneHotCommitView;
use crate::prepared::MetalPreparedSetup;
use crate::runtime::{
    MetalOneHotKernel, MetalRuntime, PackedOneHotCommitParams,
    FP128_D128_RANK3_POSITION_PARTIAL_ALIGNMENT, FP128_D512_POSITION_PARTIALS,
};
use crate::MetalCommitError;

pub(crate) const RING_D: usize = 128;
pub(crate) const INNER_RANK: usize = 3;
const ONEHOT_K: usize = 256;
const COLUMN_CAPACITY: usize = 32;
/// Rows of the packed source that map onto one ring position: K / D < 1, so
/// this is the inverse ratio and every trace row spans two positions.
const POSITIONS_PER_ROW: usize = ONEHOT_K / RING_D;
const MAX_BLOCKS_PER_COLUMN: usize = 1 << 12;
const LARGE_BLOCK_POSITIONS: usize = 1 << 17;

pub(crate) struct ValidatedShape {
    active_a_cols: usize,
    output_coefficients: usize,
    positions_per_partial: usize,
    blocks_per_column: usize,
    full_blocks_per_column: usize,
    live_columns: usize,
}

fn streams_per_command(num_positions: usize) -> usize {
    if num_positions >= LARGE_BLOCK_POSITIONS {
        8
    } else {
        32
    }
}

pub(crate) fn validate_shape<const D: usize>(
    source: PackedOneHotCommitView<'_>,
    plan: CommitInnerPlan,
) -> Result<ValidatedShape, AkitaError> {
    let partial_alignment =
        FP128_D512_POSITION_PARTIALS * FP128_D128_RANK3_POSITION_PARTIAL_ALIGNMENT;
    if D != RING_D
        || source.onehot_k() != ONEHOT_K
        || source.column_capacity() != COLUMN_CAPACITY
        || !(1..=COLUMN_CAPACITY).contains(&source.num_columns())
        || plan.n_a != INNER_RANK
        || plan.num_digits_inner != 1
        || plan.num_positions_per_block == 0
        || !plan
            .num_positions_per_block
            .is_multiple_of(partial_alignment)
    {
        return Err(MetalCommitError::UnsupportedShape(
            "fp128 Metal D128 rank-3 commit requires K256/capacity-32, rank 3, and sixteen-position alignment in every position partial"
                .into(),
        )
        .into_akita());
    }
    let positions_per_partial = plan.num_positions_per_block / FP128_D512_POSITION_PARTIALS;
    if !plan
        .num_positions_per_block
        .is_multiple_of(POSITIONS_PER_ROW)
    {
        return Err(MetalCommitError::UnsupportedShape(
            "fp128 D128 rank-3 blocks must hold whole trace rows".into(),
        )
        .into_akita());
    }
    let rows_per_block = plan.num_positions_per_block / POSITIONS_PER_ROW;
    if !source.num_rows().is_multiple_of(rows_per_block) {
        return Err(MetalCommitError::UnsupportedShape(
            "fp128 D128 rank-3 tiles require integral trace blocks".into(),
        )
        .into_akita());
    }
    let blocks_per_column = source.num_rows() / rows_per_block;
    if !blocks_per_column.is_power_of_two() || blocks_per_column > MAX_BLOCKS_PER_COLUMN {
        return Err(MetalCommitError::UnsupportedShape(
            "fp128 D128 rank-3 tiles require at most 4096 power-of-two blocks per column".into(),
        )
        .into_akita());
    }
    let full_blocks_per_column = source
        .zero_suffix_start()
        .div_ceil(rows_per_block)
        .max(1)
        .min(blocks_per_column);
    let output_coefficients = source
        .column_capacity()
        .checked_mul(blocks_per_column)
        .and_then(|count| count.checked_mul(INNER_RANK))
        .and_then(|count| count.checked_mul(D))
        .ok_or_else(|| MetalCommitError::ShapeOverflow("fp128 D128 rank-3 output").into_akita())?;
    Ok(ValidatedShape {
        active_a_cols: plan.num_positions_per_block,
        output_coefficients,
        positions_per_partial,
        blocks_per_column,
        full_blocks_per_column,
        live_columns: source.num_columns(),
    })
}

pub(crate) fn commit_validated<const D: usize>(
    backend: &MetalBackend,
    prepared: &MetalPreparedSetup,
    runtime: &MetalRuntime,
    source: PackedOneHotCommitView<'_>,
    plan: CommitInnerPlan,
    shape: ValidatedShape,
) -> Result<CommitInnerWitness<F>, AkitaError> {
    let total_start = Instant::now();
    let matrix = tracing::info_span!("packed_metal_matrix_prepare")
        .in_scope(|| prepared.matrix(runtime, D, plan.n_a, shape.active_a_cols))?;
    let work_units = shape
        .live_columns
        .checked_mul(shape.full_blocks_per_column)
        .ok_or_else(|| {
            MetalCommitError::ShapeOverflow("fp128 D128 rank-3 work units").into_akita()
        })?;
    let params = PackedOneHotCommitParams {
        num_rows: to_u64(source.num_rows(), "fp128 D128 row count")?,
        num_columns: to_u64(shape.live_columns, "fp128 D128 column count")?,
        lane_stride: to_u64(shape.live_columns, "fp128 D128 lane stride")?,
        column_capacity: to_u64(source.column_capacity(), "fp128 D128 column capacity")?,
        onehot_k: to_u64(source.onehot_k(), "fp128 D128 one-hot K")?,
        ring_d: D as u64,
        n_a: INNER_RANK as u64,
        positions_per_block: to_u64(plan.num_positions_per_block, "fp128 D128 block positions")?,
        num_digits_inner: 1,
        blocks_per_column: to_u64(shape.blocks_per_column, "fp128 D128 block count")?,
        full_blocks_per_column: to_u64(
            shape.full_blocks_per_column,
            "fp128 D128 full block count",
        )?,
        boundary_columns: 0,
        num_blocks: to_u64(work_units, "fp128 D128 work units")?,
        task_offset: 0,
        dispatch_tasks: to_u64(work_units, "fp128 D128 dispatch work units")?,
        lane_row_offset: 0,
        output_coefficients: to_u64(shape.output_coefficients, "fp128 D128 output")?,
        columns_per_threadgroup: 1,
        position_partials_per_block: FP128_D512_POSITION_PARTIALS as u64,
        positions_per_partial: to_u64(
            shape.positions_per_partial,
            "fp128 D128 positions per partial",
        )?,
        log_ring_d: 7,
        zero_column_mask: source.zero_column_mask(),
    };
    let outcome = tracing::info_span!("packed_metal_dispatch")
        .in_scope(|| {
            runtime.dispatch_packed_onehot_d128_rank3(
                matrix.buffer.as_ref(),
                source.lanes(),
                source.active_zero_rows(),
                params,
                streams_per_command(shape.active_a_cols),
            )
        })
        .map_err(MetalCommitError::into_akita)?;

    let _reconstruction_span = tracing::info_span!("packed_metal_reconstruction").entered();
    let reconstruction_start = Instant::now();
    let coefficients = outcome
        .coefficients
        .into_iter()
        .enumerate()
        .map(|(index, coefficient)| F::from_device(coefficient, index))
        .collect::<Result<Vec<_>, _>>()
        .map_err(MetalCommitError::into_akita)?;
    let witness = CommitInnerWitness {
        inner_rows: RingVec::from_coeffs_with_ring_dim(coefficients, D)?,
    };
    let output_reconstruction_time = reconstruction_start.elapsed();

    let field_additions = source
        .hot_entries()
        .checked_mul(INNER_RANK)
        .and_then(|count| count.checked_mul(D))
        .ok_or_else(|| MetalCommitError::ShapeOverflow("fp128 D128 additions").into_akita())?;
    let gathered_matrix_bytes = field_additions
        .checked_mul(size_of::<Fp128Limbs>())
        .ok_or_else(|| MetalCommitError::ShapeOverflow("fp128 D128 gathered bytes").into_akita())?;
    tracing::debug!(
        total_s = total_start.elapsed().as_secs_f64(),
        matrix_prepare_s = matrix.prepare_time.as_secs_f64(),
        matrix_cache_hit = matrix.cache_hit,
        buffer_setup_s = outcome.timings.buffer_setup.as_secs_f64(),
        command_wall_s = outcome.timings.command_wall.as_secs_f64(),
        gpu_s = outcome.timings.gpu.map(|duration| duration.as_secs_f64()),
        panel_gpu_active_s = outcome
            .panel_gpu_active
            .map(|duration| duration.as_secs_f64()),
        reduction_gpu_s = outcome.reduction_gpu.map(|duration| duration.as_secs_f64()),
        command_buffers = outcome.command_buffers,
        matrix_block_streams = outcome.matrix_block_streams,
        input_zero_copy = outcome.input_zero_copy,
        readback_copy_s = outcome.timings.readback_copy.as_secs_f64(),
        reconstruction_s = output_reconstruction_time.as_secs_f64(),
        hot_entries = source.hot_entries(),
        zero_suffix_start = source.zero_suffix_start(),
        blocks_per_column = shape.blocks_per_column,
        full_blocks_per_column = shape.full_blocks_per_column,
        output_coefficients = shape.output_coefficients,
        "completed packed Metal D128 rank-3 trace commitment"
    );
    backend
        .record_commit_metrics(MetalCommitMetrics {
            kernel: MetalOneHotKernel::PackedFp128D128Rank3,
            blocks_per_threadgroup: outcome.blocks_per_threadgroup,
            num_sources: 1,
            hot_entries: source.hot_entries(),
            field_additions: to_u64(field_additions, "fp128 D128 additions")?,
            gathered_matrix_bytes: to_u64(gathered_matrix_bytes, "fp128 D128 gathered bytes")?,
            output_bytes: shape
                .output_coefficients
                .checked_mul(size_of::<Fp128Limbs>())
                .ok_or_else(|| {
                    MetalCommitError::ShapeOverflow("fp128 D128 output bytes").into_akita()
                })?,
            scratch_bytes: outcome.scratch_bytes,
            matrix_bytes: matrix.bytes,
            matrix_cache_hit: matrix.cache_hit,
            matrix_prepare_time: matrix.prepare_time,
            index_pack_time: Duration::ZERO,
            buffer_setup_time: outcome.timings.buffer_setup,
            command_wall_time: outcome.timings.command_wall,
            gpu_time: outcome.timings.gpu,
            panel_gpu_active_time: outcome.panel_gpu_active,
            panel_gpu_span: outcome.panel_gpu_span,
            reduction_gpu_time: outcome.reduction_gpu,
            command_buffers: outcome.command_buffers,
            matrix_block_streams: outcome.matrix_block_streams,
            readback_copy_time: outcome.timings.readback_copy,
            output_reconstruction_time,
            total_time: total_start.elapsed(),
        })
        .map_err(MetalCommitError::into_akita)?;
    Ok(witness)
}

#[cfg(test)]
mod tests {
    use akita_prover::compute::RootCommitKernel;
    use akita_prover::{
        AkitaProverSetup, ComputeBackendSetup, CpuBackend, OneHotPoly, RootCommitSource,
    };
    use akita_types::SetupMatrixCapacity;

    use super::*;
    use crate::{MetalExecutionPolicy, PackedOneHotCommitView};

    /// The Metal D128 rank-3 commitment must equal the CPU `commit_inner_group`
    /// over the same one-hot source, including committed-zero lanes and a
    /// certified zero suffix.
    fn assert_rank3_parity(
        rows: usize,
        positions_per_block: usize,
        live_columns: usize,
        zero_suffix_start: Option<usize>,
    ) {
        const ZERO_COLUMN_MASK: u64 = 0b0_1010;

        let lanes = (0..rows * live_columns)
            .map(|index| {
                let row = index / live_columns;
                if row < zero_suffix_start.unwrap_or(rows) && index.is_multiple_of(7) {
                    ((index.wrapping_mul(73) % (ONEHOT_K - 1)) + 1) as u8
                } else {
                    0
                }
            })
            .collect::<Vec<_>>();
        let mut active_zero_rows = vec![0u64; rows / u64::BITS as usize];
        for row in (0..zero_suffix_start.unwrap_or(rows)).step_by(11) {
            active_zero_rows[row / u64::BITS as usize] |= 1u64 << (row % u64::BITS as usize);
        }
        let indices: Vec<Option<u8>> = (0..COLUMN_CAPACITY)
            .flat_map(|column| {
                let lanes = &lanes;
                let active_zero_rows = &active_zero_rows;
                (0..rows).map(move |row| {
                    if column >= live_columns {
                        None
                    } else {
                        let lane = lanes[row * live_columns + column];
                        let committed_zero = ZERO_COLUMN_MASK & (1u64 << column) != 0
                            && active_zero_rows[row / u64::BITS as usize]
                                & (1u64 << (row % u64::BITS as usize))
                                != 0;
                        (lane != 0 || committed_zero).then_some(lane)
                    }
                })
            })
            .collect();
        let hot_entries = indices.iter().filter(|entry| entry.is_some()).count();
        let generic = OneHotPoly::<F, u8>::new(ONEHOT_K, indices).unwrap();
        let plan = CommitInnerPlan {
            n_a: INNER_RANK,
            num_positions_per_block: positions_per_block,
            num_digits_inner: 1,
            log_basis_inner: 3,
        };
        let num_vars = (rows * COLUMN_CAPACITY * ONEHOT_K).trailing_zeros() as usize;
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            num_vars,
            1,
            SetupMatrixCapacity {
                num_field_elements: INNER_RANK * positions_per_block * RING_D,
            },
        )
        .unwrap();

        let cpu = CpuBackend::DEFAULT;
        let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
        let cpu_view =
            <OneHotPoly<F, u8> as RootCommitSource<F, RING_D>>::commit_view(&generic).unwrap();
        let cpu_witness = cpu
            .commit_inner_group(&cpu_prepared, vec![cpu_view], plan)
            .unwrap()
            .remove(0);

        let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let metal_prepared = metal.prepare_setup(&setup).unwrap();
        let source = match zero_suffix_start {
            Some(zero_suffix_start) => PackedOneHotCommitView::new_with_precomputed_metrics(
                ONEHOT_K,
                COLUMN_CAPACITY,
                live_columns,
                &lanes,
                &active_zero_rows,
                ZERO_COLUMN_MASK,
                hot_entries,
                zero_suffix_start,
            )
            .unwrap(),
            None => PackedOneHotCommitView::new_with_active_zero_rows(
                ONEHOT_K,
                COLUMN_CAPACITY,
                live_columns,
                &lanes,
                &active_zero_rows,
                ZERO_COLUMN_MASK,
            )
            .unwrap(),
        };
        let metal_witness = metal
            .commit_packed_onehot::<RING_D>(&metal_prepared, source, plan)
            .unwrap();
        assert_eq!(cpu_witness.inner_rows.ring_dim(), RING_D);
        assert_eq!(cpu_witness.inner_rows, metal_witness.inner_rows);

        let second = metal
            .commit_packed_onehot::<RING_D>(&metal_prepared, source, plan)
            .unwrap();
        assert_eq!(cpu_witness.inner_rows, second.inner_rows);
        let metrics = metal.last_commit_metrics().unwrap().unwrap();
        assert_eq!(metrics.kernel, MetalOneHotKernel::PackedFp128D128Rank3);
        assert_eq!(metrics.hot_entries, hot_entries);
        assert!(metrics.matrix_cache_hit);
    }

    #[test]
    fn parity_d128_rank3_k256() {
        assert_rank3_parity(1 << 12, 1 << 10, 5, None);
    }

    #[test]
    fn parity_d128_rank3_k256_skip_zero_suffix() {
        assert_rank3_parity(1 << 12, 1 << 10, 5, Some(2048));
    }

    #[test]
    fn parity_d128_rank3_k256_full_capacity_rows_2p16() {
        assert_rank3_parity(1 << 16, 1 << 12, 32, None);
    }

    #[test]
    fn parity_d128_rank3_k256_rows_2p20() {
        assert_rank3_parity(1 << 20, 1 << 16, 5, Some(1 << 19));
    }

    #[test]
    fn rejects_rank_one_and_d512_shapes() {
        let lanes = vec![0u8; 4096 * 2];
        let source = PackedOneHotCommitView::new(256, 32, 2, &lanes).unwrap();
        let rank_one = CommitInnerPlan {
            n_a: 1,
            num_positions_per_block: 1 << 10,
            num_digits_inner: 1,
            log_basis_inner: 3,
        };
        assert!(validate_shape::<RING_D>(source, rank_one).is_err());
        let rank_three = CommitInnerPlan {
            n_a: INNER_RANK,
            ..rank_one
        };
        assert!(validate_shape::<512>(source, rank_three).is_err());
        assert!(validate_shape::<RING_D>(source, rank_three).is_ok());
        let ragged = CommitInnerPlan {
            num_positions_per_block: 1 << 7,
            ..rank_three
        };
        assert!(validate_shape::<RING_D>(source, ragged).is_err());
    }
}
