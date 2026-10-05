use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ops::Range;
use std::time::{Duration, Instant};

use akita_algebra::{CrtCapacity, GarnerData, NttPrime};
#[cfg(test)]
use jolt_field::Zero;
use jolt_field::{One, Ring};
use metal::objc::rc::autoreleasepool;
use metal::objc::{runtime::Sel, Message};
use metal::{
    Buffer, CommandBufferRef, CommandQueue, CompileOptions, ComputeCommandEncoderRef,
    ComputePipelineState, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

use crate::error::metal_status::CommandStatus;
use crate::field::{Fp128Limbs, F};
use crate::MetalCommitError;

mod buffers;
mod coefficient_packing;
mod commands;
mod commit;
mod direct_range;
mod fold;
mod relation_prefix;
mod relation_resident;
mod relation_rounds;
mod relation_setup;
mod resources;
mod rows;
#[cfg(test)]
mod tests;

use commands::*;
use resources::{d512_linear_relation_resources, recursive_commit_resources};

const DIRECT_KERNEL_NAME: &str = "akita_onehot_commit_gather";
const BLOCK_BATCHED_KERNEL_NAME: &str = "akita_onehot_commit_block_batched";
const PACKED_FP128_D512_PANELS_KERNEL_NAME: &str = "akita_packed_onehot_commit_fp128_d512_panels";
const PACKED_PARTIAL_REDUCTION_KERNEL_NAME: &str = "akita_packed_onehot_reduce_partials";
const PACKED_FP128_D128_RANK3_KERNEL_NAME: &str = "akita_packed_onehot_commit_fp128_d128_rank3";
const FP128_D128_DECOMPOSE_FOLD_KERNEL_NAME: &str = "akita_fp128_d128_decompose_fold";
const FP128_D128_SUBRING64_DECOMPOSE_FOLD_KERNEL_NAME: &str =
    "akita_fp128_d128_subring64_decompose_fold";
const FP128_D64_DIGIT_ROWS_PARTIALS_KERNEL_NAME: &str = "akita_fp128_d64_digit_rows_partials";
const FP128_D64_DIGIT_ROWS_REDUCE_KERNEL_NAME: &str = "akita_fp128_d64_digit_rows_reduce";
const FP128_I8_COEFFICIENT_PACKING_KERNEL_NAME: &str = "akita_fp128_i8_coefficient_packing";
const FP128_PACKED_ONEHOT_COEFFICIENT_PACKING_PARTIALS_KERNEL_NAME: &str =
    "akita_fp128_packed_onehot_coefficient_packing_partials";
const FP128_PACKED_ONEHOT_COEFFICIENT_PACKING_REDUCE_KERNEL_NAME: &str =
    "akita_fp128_packed_onehot_coefficient_packing_reduce";
const FP128_D512_DECOMPOSE_FOLD_KERNEL_NAME: &str = "akita_fp128_d512_decompose_fold";
const FP128_D512_SUBRING64_DECOMPOSE_FOLD_KERNEL_NAME: &str =
    "akita_fp128_d512_subring64_decompose_fold";
const FP128_D512_BUILD_FOLD_INDEX_KERNEL_NAME: &str = "akita_fp128_d512_build_fold_index";
const FP128_D512_BUILD_COEFFICIENT_PACKING_INDEX_KERNEL_NAME: &str =
    "akita_fp128_d512_build_coefficient_packing_index";
const FP128_D512_INDEXED_COEFFICIENT_PACKING_KERNEL_NAME: &str =
    "akita_fp128_d512_indexed_coefficient_packing_partials";
const FP128_D512_INDEXED_COEFFICIENT_PACKING_REDUCE_KERNEL_NAME: &str =
    "akita_fp128_d512_indexed_coefficient_packing_reduce";
const FP128_D512_INDEXED_SUBRING64_DECOMPOSE_FOLD_KERNEL_NAME: &str =
    "akita_fp128_d512_indexed_subring64_decompose_fold";
const FP128_D512_FUSED_SUBRING64_DECOMPOSE_FOLD_KERNEL_NAME: &str =
    "akita_fp128_d512_fused_subring64_decompose_fold";
const FP128_D512_LINEAR_RELATION_PARTIALS_KERNEL_NAME: &str =
    "akita_fp128_d512_linear_relation_partials";
const FP128_D512_LINEAR_RELATION_REDUCE_KERNEL_NAME: &str =
    "akita_fp128_d512_linear_relation_reduce";
const FP128_D512_LINEAR_RELATION_RECONSTRUCT_KERNEL_NAME: &str =
    "akita_fp128_d512_linear_relation_reconstruct";
const FP128_RECURSIVE_COMMIT_MATRIX_NTT_KERNEL_NAME: &str =
    "akita_fp128_recursive_commit_matrix_ntt";
const FP128_RECURSIVE_COMMIT_MATVEC_KERNEL_NAME: &str = "akita_fp128_recursive_commit_matvec";
const FP128_RECURSIVE_COMMIT_RECONSTRUCT_KERNEL_NAME: &str =
    "akita_fp128_recursive_commit_reconstruct";
const FP128_DIRECT_RANGE_INITIAL_KERNEL_NAME: &str = "akita_fp128_direct_range_initial_partials";
const FP128_DIRECT_RANGE_COMPACT_FOLD_KERNEL_NAME: &str =
    "akita_fp128_direct_range_compact_fold_partials";
const FP128_DIRECT_RANGE_FIELD_FOLD_KERNEL_NAME: &str =
    "akita_fp128_direct_range_field_fold_partials";
const FP128_DIRECT_RANGE_REDUCE_KERNEL_NAME: &str = "akita_fp128_direct_range_reduce";
const FP128_DIRECT_RANGE_FINALIZE_KERNEL_NAME: &str = "akita_fp128_direct_range_finalize";
const FP128_BLAKE2B_SUMCHECK_CHALLENGE_KERNEL_NAME: &str = "akita_fp128_blake2b_sumcheck_challenge";
const FP128_BLAKE2B_RELATION_SUMCHECK_ROUND_KERNEL_NAME: &str =
    "akita_fp128_blake2b_relation_sumcheck_round";
const FP128_DIRECT_RELATION_INITIAL_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_initial_partials";
const FP128_DIRECT_RELATION_COMPACT_FOLD_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_compact_fold_partials";
const FP128_DIRECT_RELATION_FIELD_FOLD_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_field_fold_partials";
const FP128_DIRECT_RELATION_ADDITIONAL_COMPACT_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_additional_compact_partials";
const FP128_DIRECT_RELATION_ADDITIONAL_FIELD_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_additional_field_partials";
const FP128_DIRECT_RELATION_TWO_ROUND_PREFIX_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_two_round_prefix_partials";
const FP128_DIRECT_RELATION_TWO_ROUND_PREFIX_REDUCE_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_two_round_prefix_reduce";
const FP128_DIRECT_RELATION_LINEAR_FOLD_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_linear_fold";
const FP128_DIRECT_RELATION_ALPHA_FOLD_KERNEL_NAME: &str = "akita_fp128_direct_relation_alpha_fold";
const FP128_DIRECT_RELATION_SCALAR_ADVANCE_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_scalar_advance";
const FP128_DIRECT_RELATION_ADDITIONAL_FOLD_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_additional_fold";
const FP128_DIRECT_RELATION_SETUP_SOURCE_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_setup_source";
const FP128_DIRECT_RELATION_SPARSE_SOURCE_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_sparse_source";
const KERNEL_SOURCE: &str = include_str!("kernels/onehot.metal");
const FP128_D512_THREADS: usize = 1_024;
const PACKED_ONEHOT_BUFFER_ALIGNMENT: usize = 16 * 1024;
pub(crate) const FP128_D512_TASKS_PER_STREAM: usize = 32;
pub(crate) const FP128_D512_POSITION_PARTIALS: usize = 16;
pub(crate) const FP128_D128_RANK3_TASKS_PER_STREAM: usize = 64;
pub(crate) const FP128_D128_RANK3_POSITION_PARTIAL_ALIGNMENT: usize = 16;
const FP128_D128_RANK3_THREADGROUP_BYTES: usize = 5 * 1_024 * size_of::<u32>();
const FP128_D128_RANK3_RING_D: u64 = 128;
const FP128_D128_RANK3_INNER_RANK: u64 = 3;
const FP128_D512_TILE_FIELD_ELEMENTS: usize = 2_048;
const FP128_D512_THREADGROUP_BYTES: usize =
    FP128_D512_TILE_FIELD_ELEMENTS * size_of::<Fp128Limbs>();
const FP128_D512_COEFFICIENT_BANDS: usize = 2;
const FP128_D64_DIGIT_ROWS_THREADS: usize = 256;
const FP128_D64_DIGIT_ROWS_PARTIAL_THREADS: usize = 64;
pub(crate) const FP128_D64_DIGIT_ROWS_COLUMNS_PER_PARTIAL: usize = 128;
const FP128_COEFFICIENT_PACKING_THREADS: usize = 256;
pub(crate) const FP128_D512_PACKING_INDEX_TILE_POSITIONS: usize = 256;
pub(crate) const FP128_D512_PACKING_INDEX_BUCKET_OFFSETS: usize = 33;
const FP128_D512_PACKING_TILES_PER_CHUNK: usize = 32;
const FP128_PACKED_COEFFICIENT_PACKING_PARTIAL_THREADS: usize = 256;
const FP128_PACKED_COEFFICIENT_PACKING_REDUCE_THREADS: usize = 256;
pub(crate) const FP128_D512_FOLD_INDEX_TILE_TASKS: usize = 256;
pub(crate) const FP128_D512_FOLD_INDEX_COUNT_BUCKETS: usize = 8;
const FP128_D512_SUBRING_DIMENSION: usize = 64;
pub(crate) const FP128_PACKED_COEFFICIENT_PACKING_ROWS_PER_PARTIAL: usize =
    i32::MAX as usize / u16::MAX as usize;
const FP128_DIRECT_RANGE_THREADS: usize = 256;
const FP128_DIRECT_RANGE_MAX_WORKGROUPS: usize = 4_096;
const FP128_DIRECT_RANGE_STORED_COEFFICIENTS: usize = 4;
const FP128_DIRECT_RELATION_STORED_COEFFICIENTS: usize = 4;
const FP128_DIRECT_RELATION_TWO_ROUND_PREFIX_OUTPUTS: usize = 16;
const FP128_DIRECT_RELATION_PREFIX_LANES_PER_THREAD: usize = 2;
const FP128_D512_LINEAR_RELATION_THREADS: usize = 512;
const FP128_D512_LINEAR_RELATION_NTT_SIZE: usize = 1_024;
const FP128_D512_LINEAR_RELATION_COLUMNS_PER_TILE: usize = 64;
pub(crate) const FP128_D512_LINEAR_RELATION_NUM_PRIMES: usize = 6;
const FP128_D512_LINEAR_RELATION_RAW_PRIMES: [i32; FP128_D512_LINEAR_RELATION_NUM_PRIMES] = [
    1_073_692_673,
    1_073_668_097,
    1_073_707_009,
    1_073_738_753,
    1_073_732_609,
    1_073_698_817,
];
const FP128_RECURSIVE_COMMIT_THREADS: usize = 512;
const FP128_RECURSIVE_COMMIT_RECONSTRUCT_THREADS: usize = 256;
const FP128_RECURSIVE_COMMIT_BLOCKS_PER_GROUP: usize = 16;
const FP128_RECURSIVE_COMMIT_MAX_ROWS: usize = 8;

fn pack_biased_subring64_challenges(dense_challenges: &[i8]) -> Result<Vec<u32>, MetalCommitError> {
    if !dense_challenges
        .len()
        .is_multiple_of(FP128_D512_SUBRING_DIMENSION)
        || dense_challenges
            .iter()
            .any(|&coefficient| !(-2..=2).contains(&coefficient))
    {
        return Err(MetalCommitError::UnsupportedShape(
            "indexed D512 fold requires D64 challenge coefficients in [-2, 2]".into(),
        ));
    }

    let mut packed = Vec::with_capacity(dense_challenges.len());
    for challenge in dense_challenges.chunks_exact(FP128_D512_SUBRING_DIMENSION) {
        for source_phase in 0..8 {
            for destination_quad in 0..8 {
                let start = (8 * destination_quad + FP128_D512_SUBRING_DIMENSION - source_phase)
                    % FP128_D512_SUBRING_DIMENSION;
                let word = (0..8).fold(0u32, |word, offset| {
                    let position = (start + offset) % FP128_D512_SUBRING_DIMENSION;
                    let biased = u32::from((challenge[position] + 2) as u8);
                    word | (biased << (4 * offset))
                });
                packed.push(word);
            }
        }
    }
    Ok(packed)
}

fn validate_packed_fold_index_geometry(
    params: PackedFoldIndexParams,
    lane_count: usize,
) -> Result<(), MetalCommitError> {
    let expected_lanes = params
        .num_rows
        .checked_mul(params.lane_stride)
        .ok_or(MetalCommitError::ShapeOverflow("fold-index lanes"))?;
    let expected_tasks = params
        .blocks_per_column
        .checked_mul(params.num_columns)
        .and_then(|count| count.checked_mul(2))
        .ok_or(MetalCommitError::ShapeOverflow(
            "fold-index tasks per position",
        ))?;
    let expected_tiles = expected_tasks.div_ceil(FP128_D512_FOLD_INDEX_TILE_TASKS as u64);
    let expected_records = params
        .num_positions
        .checked_mul(expected_tiles)
        .and_then(|count| count.checked_mul(FP128_D512_FOLD_INDEX_TILE_TASKS as u64))
        .ok_or(MetalCommitError::ShapeOverflow("fold-index records"))?;
    let expected_counts = params
        .num_positions
        .checked_mul(expected_tiles)
        .and_then(|count| count.checked_mul(FP128_D512_FOLD_INDEX_COUNT_BUCKETS as u64))
        .ok_or(MetalCommitError::ShapeOverflow("fold-index counts"))?;
    if params.num_rows == 0
        || params.num_columns == 0
        || params.num_columns > params.lane_stride
        || params.num_positions == 0
        || params.position_start != 0
        || params.tasks_per_position != expected_tasks
        || params.tiles_per_position != expected_tiles
        || params.record_slots != expected_records
        || params.count_entries != expected_counts
        || u64::try_from(lane_count).ok() != Some(expected_lanes)
    {
        return Err(MetalCommitError::UnsupportedShape(
            "fp128 D512 packed fold-index geometry is unsupported".into(),
        ));
    }
    Ok(())
}

/// Metal implementation selected for a one-hot inner commitment.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MetalOneHotKernel {
    /// One output coefficient per thread with direct matrix gathers.
    #[default]
    DirectGather,
    /// Several source blocks per threadgroup with cache-local global reads.
    BlockBatched,
    /// Exact fp128 D512 panels with one block-column task per SIMDgroup.
    PackedFp128D512Panels,
    /// Exact fp128 D128 rank-3 per-element tiles with two tasks per SIMDgroup.
    PackedFp128D128Rank3,
}

/// Stable Metal device properties relevant to commitment scheduling.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalDeviceCapabilities {
    /// Registry-reported device name.
    pub name: String,
    /// Maximum allocation length.
    pub max_buffer_length: u64,
    /// Recommended resident working set.
    pub recommended_max_working_set_size: u64,
    /// Maximum threadgroup-local memory.
    pub max_threadgroup_memory_length: u64,
    /// SIMD execution width of the baseline one-hot pipeline.
    pub thread_execution_width: usize,
    /// Maximum threads in one baseline one-hot threadgroup.
    pub max_total_threads_per_threadgroup: usize,
    /// Static threadgroup memory reserved by the baseline pipeline.
    pub static_threadgroup_memory_length: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct OneHotCommitParams {
    pub(crate) num_sources: u64,
    pub(crate) chunks_per_source: u64,
    pub(crate) onehot_k: u64,
    pub(crate) ring_d: u64,
    pub(crate) n_a: u64,
    pub(crate) positions_per_block: u64,
    pub(crate) num_digits_inner: u64,
    pub(crate) num_blocks: u64,
    pub(crate) total_field_elements: u64,
    pub(crate) output_coefficients: u64,
    pub(crate) blocks_per_threadgroup: u64,
    pub(crate) log_onehot_k: u64,
    pub(crate) log_ring_d: u64,
}

const _: [(); 104] = [(); size_of::<OneHotCommitParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct PackedOneHotCommitParams {
    pub(crate) num_rows: u64,
    pub(crate) num_columns: u64,
    pub(crate) lane_stride: u64,
    pub(crate) column_capacity: u64,
    pub(crate) onehot_k: u64,
    pub(crate) ring_d: u64,
    pub(crate) n_a: u64,
    pub(crate) positions_per_block: u64,
    pub(crate) num_digits_inner: u64,
    pub(crate) blocks_per_column: u64,
    pub(crate) full_blocks_per_column: u64,
    pub(crate) boundary_columns: u64,
    pub(crate) num_blocks: u64,
    pub(crate) task_offset: u64,
    pub(crate) dispatch_tasks: u64,
    pub(crate) lane_row_offset: u64,
    pub(crate) output_coefficients: u64,
    pub(crate) columns_per_threadgroup: u64,
    pub(crate) position_partials_per_block: u64,
    pub(crate) positions_per_partial: u64,
    pub(crate) log_ring_d: u64,
    pub(crate) zero_column_mask: u64,
}

const _: [(); 176] = [(); size_of::<PackedOneHotCommitParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DigitRowsParams {
    pub(crate) num_vectors: u64,
    pub(crate) num_rows: u64,
    pub(crate) num_cols: u64,
    pub(crate) ring_d: u64,
    pub(crate) output_coefficients: u64,
    pub(crate) columns_per_partial: u64,
    pub(crate) column_partials: u64,
    pub(crate) retain_quotients: u64,
}

const _: [(); 64] = [(); size_of::<DigitRowsParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct I8CoefficientPackingParams {
    pub(crate) num_sources: u64,
    pub(crate) source_coefficients: u64,
    pub(crate) live_coefficients: u64,
    pub(crate) num_live_positions: u64,
    pub(crate) positions_per_block: u64,
    pub(crate) num_blocks: u64,
    pub(crate) ring_d: u64,
    pub(crate) stride: u64,
    pub(crate) subring_dimension: u64,
    pub(crate) output_coefficients: u64,
}

const _: [(); 80] = [(); size_of::<I8CoefficientPackingParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct PackedOneHotCoefficientPackingParams {
    pub(crate) num_rows: u64,
    pub(crate) num_columns: u64,
    pub(crate) column_capacity: u64,
    pub(crate) onehot_k: u64,
    pub(crate) ring_d: u64,
    pub(crate) positions_per_block: u64,
    pub(crate) blocks_per_column: u64,
    pub(crate) rows_per_block: u64,
    pub(crate) rows_per_partial: u64,
    pub(crate) row_partials_per_block: u64,
    pub(crate) num_blocks: u64,
    pub(crate) stride: u64,
    pub(crate) subring_dimension: u64,
    pub(crate) output_coefficients: u64,
    pub(crate) partial_coefficients: u64,
    pub(crate) zero_column_mask: u64,
}

const _: [(); 128] = [(); size_of::<PackedOneHotCoefficientPackingParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct PackedDecomposeFoldParams {
    pub(crate) num_rows: u64,
    pub(crate) num_columns: u64,
    pub(crate) lane_stride: u64,
    pub(crate) num_positions: u64,
    pub(crate) position_start: u64,
    pub(crate) blocks_per_column: u64,
    pub(crate) challenge_weight: u64,
    pub(crate) output_coefficients: u64,
    pub(crate) zero_column_mask: u64,
    pub(crate) onehot_k: u64,
    pub(crate) zero_suffix_start: u64,
}

const _: [(); 88] = [(); size_of::<PackedDecomposeFoldParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct PackedFoldIndexParams {
    pub(crate) num_rows: u64,
    pub(crate) num_columns: u64,
    pub(crate) lane_stride: u64,
    pub(crate) num_positions: u64,
    pub(crate) position_start: u64,
    pub(crate) blocks_per_column: u64,
    pub(crate) tasks_per_position: u64,
    pub(crate) tiles_per_position: u64,
    pub(crate) record_slots: u64,
    pub(crate) count_entries: u64,
    pub(crate) output_coefficients: u64,
    pub(crate) fold_digits: u64,
    pub(crate) fold_log_basis: u64,
}

const _: [(); 104] = [(); size_of::<PackedFoldIndexParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct PackedCoefficientPackingIndexParams {
    pub(crate) num_rows: u64,
    pub(crate) num_columns: u64,
    pub(crate) lane_stride: u64,
    pub(crate) num_positions: u64,
    pub(crate) blocks_per_column: u64,
    pub(crate) position_tiles: u64,
    pub(crate) record_slots: u64,
    pub(crate) offset_entries: u64,
}

const _: [(); 64] = [(); size_of::<PackedCoefficientPackingIndexParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct D512LinearRelationParams {
    pub(crate) num_columns: u64,
    pub(crate) columns_per_tile: u64,
    pub(crate) num_tiles: u64,
    pub(crate) num_primes: u64,
    pub(crate) ntt_size: u64,
    pub(crate) output_coefficients: u64,
    pub(crate) rhs_abs_bound: u64,
}

const _: [(); 56] = [(); size_of::<D512LinearRelationParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct RecursiveCommitParams {
    pub(crate) num_blocks: u64,
    pub(crate) blocks_per_group: u64,
    pub(crate) num_block_groups: u64,
    pub(crate) num_rows: u64,
    pub(crate) num_cols: u64,
    pub(crate) ring_d: u64,
    pub(crate) num_primes: u64,
    pub(crate) matrix_rings: u64,
    pub(crate) output_coefficients: u64,
    pub(crate) rhs_abs_bound: u64,
}

const _: [(); 80] = [(); size_of::<RecursiveCommitParams>()];

#[repr(C)]
#[derive(Clone, Copy)]
struct D512LinearNttPrime {
    p: i32,
    pinv: i32,
    mont: i32,
    montsq: i32,
}

const _: [(); 16] = [(); size_of::<D512LinearNttPrime>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DirectRangeParams {
    pub(crate) live_len: u64,
    pub(crate) current_len: u64,
    pub(crate) current_live_len: u64,
    pub(crate) input_live_len: u64,
    pub(crate) pair_count: u64,
    pub(crate) num_first: u64,
    pub(crate) num_second: u64,
    pub(crate) workgroups: u64,
    pub(crate) basis: u64,
    pub(crate) prefix_size: u64,
    pub(crate) materialize_prefix: u64,
    pub(crate) resident_challenges: u64,
}

const _: [(); 96] = [(); size_of::<DirectRangeParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Blake2bSumcheckChallengeParams {
    include_claim: u64,
    coefficient_count: u64,
    prior_squeezed_bytes: u64,
    reserved: u64,
}

const _: [(); 32] = [(); size_of::<Blake2bSumcheckChallengeParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DirectRelationParams {
    pub(crate) live_len: u64,
    pub(crate) current_len: u64,
    pub(crate) current_live_len: u64,
    pub(crate) input_live_len: u64,
    pub(crate) pair_count: u64,
    pub(crate) num_first: u64,
    pub(crate) num_second: u64,
    pub(crate) workgroups: u64,
    pub(crate) current_coeff_count: u64,
    pub(crate) live_lane_count: u64,
    pub(crate) prefix_size: u64,
    pub(crate) materialize_prefix: u64,
    pub(crate) linear_mode: u64,
    pub(crate) additional_pair_count: u64,
    pub(crate) additional_workgroups: u64,
    pub(crate) fold_lane_weights: u64,
    pub(crate) resident_challenges: u64,
}

const _: [(); 136] = [(); size_of::<DirectRelationParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DirectRelationTranscriptParams {
    prior_squeezed_bytes: u64,
    has_additional: u64,
}

const _: [(); 16] = [(); size_of::<DirectRelationTranscriptParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DirectRelationTwoRoundPrefixParams {
    live_lane_count: u64,
    coefficient_count: u64,
    y_quads: u64,
    equality_first_len: u64,
    workgroups: u64,
    lanes_per_thread: u64,
    norm_omitted_corner: u64,
    linear_mode: u64,
}

const _: [(); 64] = [(); size_of::<DirectRelationTwoRoundPrefixParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DirectRelationScalars {
    pub(crate) l_at_0: Fp128Limbs,
    pub(crate) l_at_1: Fp128Limbs,
    pub(crate) binary_batching: Fp128Limbs,
}

const _: [(); 48] = [(); size_of::<DirectRelationScalars>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DirectRelationLinearSegment {
    pub(crate) factor: Fp128Limbs,
    pub(crate) source_index: u32,
    pub(crate) target_lane_start: u32,
    pub(crate) target_lane_stride: u32,
    pub(crate) source_lane_start: u32,
    pub(crate) source_lane_stride: u32,
    pub(crate) lane_count: u32,
}

const _: [(); 48] = [(); size_of::<DirectRelationLinearSegment>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DirectRelationLinearFoldParams {
    current_coeff_count: u64,
    source_lane_count: u64,
    current_live_lane_count: u64,
    output_len: u64,
    mode: u64,
}

const _: [(); 40] = [(); size_of::<DirectRelationLinearFoldParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DirectRelationReducedSourceParams {
    ring_dimension: u64,
    row_count: u64,
    item_count: u64,
    reserved: u64,
    alpha: Fp128Limbs,
    wrap_correction: Fp128Limbs,
}

const _: [(); 64] = [(); size_of::<DirectRelationReducedSourceParams>()];

pub(crate) enum DirectRelationLinearSourceInput {
    Values(Vec<Fp128Limbs>),
    ReducedSetup {
        matrix: Buffer,
        ring_dimension: usize,
        row_count: usize,
        column_count: usize,
        row_weights: Vec<Fp128Limbs>,
        alpha_powers: Vec<Fp128Limbs>,
        alpha: Fp128Limbs,
        wrap_correction: Fp128Limbs,
    },
    ReducedSparse {
        ring_dimension: usize,
        challenge_count: usize,
        term_offsets: Vec<u32>,
        positions: Vec<u32>,
        coefficients: Vec<i8>,
        alpha_powers: Vec<Fp128Limbs>,
        alpha: Fp128Limbs,
        wrap_correction: Fp128Limbs,
    },
}

impl DirectRelationLinearSourceInput {
    fn element_len(&self) -> Option<usize> {
        match self {
            Self::Values(values) => Some(values.len()),
            Self::ReducedSetup {
                ring_dimension,
                column_count,
                ..
            } => ring_dimension.checked_mul(*column_count),
            Self::ReducedSparse {
                ring_dimension,
                challenge_count,
                ..
            } => ring_dimension.checked_mul(*challenge_count),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DirectRelationAdditionalPair {
    pub(crate) parent: u64,
    pub(crate) reserved: u64,
    pub(crate) linear: [Fp128Limbs; 2],
    pub(crate) binary: [Fp128Limbs; 2],
}

const _: [(); 80] = [(); size_of::<DirectRelationAdditionalPair>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DirectRelationAdditionalFoldMapping {
    parent: u64,
    left: u32,
    right: u32,
}

const _: [(); 16] = [(); size_of::<DirectRelationAdditionalFoldMapping>()];

trait PackedLaneSource {
    fn lane_count(&self) -> usize;

    fn wait_lanes(&self, rows: Range<usize>, lane_stride: usize)
        -> Result<&[u8], MetalCommitError>;
}

struct ResidentPackedLanes<'a> {
    lanes: &'a [u8],
}

impl PackedLaneSource for ResidentPackedLanes<'_> {
    fn lane_count(&self) -> usize {
        self.lanes.len()
    }

    fn wait_lanes(
        &self,
        rows: Range<usize>,
        lane_stride: usize,
    ) -> Result<&[u8], MetalCommitError> {
        let first = rows
            .start
            .checked_mul(lane_stride)
            .ok_or(MetalCommitError::ShapeOverflow("packed first lane"))?;
        let final_lane = rows
            .end
            .checked_mul(lane_stride)
            .ok_or(MetalCommitError::ShapeOverflow("packed final lane"))?;
        self.lanes
            .get(first..final_lane)
            .ok_or(MetalCommitError::ShapeOverflow("packed command lane range"))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DispatchTimings {
    pub(crate) buffer_setup: Duration,
    pub(crate) command_wall: Duration,
    pub(crate) gpu: Option<Duration>,
    pub(crate) readback_copy: Duration,
}

pub(crate) struct DispatchOutcome {
    pub(crate) coefficients: Vec<Fp128Limbs>,
    pub(crate) timings: DispatchTimings,
    pub(crate) panel_gpu_active: Option<Duration>,
    pub(crate) panel_gpu_span: Option<Duration>,
    pub(crate) reduction_gpu: Option<Duration>,
    pub(crate) command_buffers: usize,
    pub(crate) kernel: MetalOneHotKernel,
    pub(crate) blocks_per_threadgroup: usize,
    pub(crate) columns_per_threadgroup: usize,
    pub(crate) matrix_block_streams: usize,
    pub(crate) scratch_bytes: usize,
    pub(crate) input_zero_copy: bool,
}

pub(crate) struct DigitRowsDispatchOutcome {
    pub(crate) coefficients: Vec<Fp128Limbs>,
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct CoefficientPackingDispatchOutcome {
    pub(crate) coefficients: Vec<Fp128Limbs>,
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct PackedDecomposeFoldDispatchOutcome {
    pub(crate) centered_coefficients: Vec<i32>,
    pub(crate) timings: DispatchTimings,
    pub(crate) consumer_time: Duration,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct PackedFp128D512FoldIndex {
    records: Buffer,
    counts: Buffer,
    params: PackedFoldIndexParams,
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

#[derive(Clone, Copy)]
pub(crate) enum PackedFp128D512FoldSource<'a> {
    Retained(&'a PackedFp128D512FoldIndex),
    Fused(PackedFoldIndexParams),
}

pub(crate) struct PackedFp128D512CoefficientPackingIndex {
    records: Buffer,
    offsets: Buffer,
    params: PackedCoefficientPackingIndexParams,
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct D512LinearRelationDispatchOutcome {
    pub(crate) coefficients: Vec<Fp128Limbs>,
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct RecursiveCommitDispatchOutcome {
    pub(crate) coefficients: Vec<Fp128Limbs>,
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct RecursiveCommitMatrixNttOutcome {
    pub(crate) buffer: Buffer,
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct DirectRangeRoundOutcome {
    pub(crate) coefficients: [Fp128Limbs; FP128_DIRECT_RANGE_STORED_COEFFICIENTS],
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

#[cfg(test)]
pub(crate) struct Blake2bSumcheckChallengeOutcome {
    pub(crate) challenge: Fp128Limbs,
    pub(crate) chaining_value: [u8; 64],
}

pub(crate) struct DirectRangeAdvanceOutcome {
    pub(crate) next_coefficients: Option<[Fp128Limbs; FP128_DIRECT_RANGE_STORED_COEFFICIENTS]>,
    pub(crate) final_evaluation: Option<Fp128Limbs>,
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

#[cfg(test)]
pub(crate) struct DirectRangeResidentOutcome {
    pub(crate) round_coefficients: Vec<[Fp128Limbs; FP128_DIRECT_RANGE_STORED_COEFFICIENTS]>,
    pub(crate) challenges: Vec<Fp128Limbs>,
    pub(crate) final_evaluation: Fp128Limbs,
    pub(crate) chaining_value: [u8; 64],
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct DirectRangeSession {
    compact_digits: Buffer,
    tables: [Buffer; 2],
    partials: Buffer,
    round_output: Buffer,
    final_output: Buffer,
    live_len: usize,
    current_len: usize,
    current_live_len: usize,
    current_table: Option<usize>,
    compact_prefix_rounds: usize,
    rounds_folded: usize,
    allocation_bytes: usize,
}

pub(crate) struct DirectRelationRoundData<'a> {
    pub(crate) e_first: &'a [Fp128Limbs],
    pub(crate) e_second: &'a [Fp128Limbs],
    pub(crate) alpha: &'a [Fp128Limbs],
    pub(crate) additional_pairs: &'a [DirectRelationAdditionalPair],
    pub(crate) scalars: DirectRelationScalars,
    pub(crate) live_lane_count: usize,
}

pub(crate) struct DirectRelationRoundOutcome {
    pub(crate) coefficients: [Fp128Limbs; FP128_DIRECT_RELATION_STORED_COEFFICIENTS],
    pub(crate) additional_coefficients: [Fp128Limbs; FP128_DIRECT_RELATION_STORED_COEFFICIENTS],
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct DirectRelationTwoRoundPrefixOutcome {
    pub(crate) norm_evals_except_corner: [Fp128Limbs; 8],
    pub(crate) relation_evals_except_corner: [Fp128Limbs; 8],
    pub(crate) additional_coefficients: [Fp128Limbs; FP128_DIRECT_RELATION_STORED_COEFFICIENTS],
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct DirectRelationAdditionalOutcome {
    pub(crate) coefficients: [Fp128Limbs; FP128_DIRECT_RELATION_STORED_COEFFICIENTS],
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct DirectRelationAdvanceOutcome {
    pub(crate) next_coefficients: Option<[Fp128Limbs; FP128_DIRECT_RELATION_STORED_COEFFICIENTS]>,
    pub(crate) next_additional_coefficients:
        Option<[Fp128Limbs; FP128_DIRECT_RELATION_STORED_COEFFICIENTS]>,
    pub(crate) final_evaluation: Option<Fp128Limbs>,
    pub(crate) final_linear_evaluation: Option<Fp128Limbs>,
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct DirectRelationResidentEqRound {
    pub(crate) e_first: Vec<Fp128Limbs>,
    pub(crate) e_second: Vec<Fp128Limbs>,
    pub(crate) tau: Fp128Limbs,
}

pub(crate) struct DirectRelationResidentOutcome {
    pub(crate) round_coefficients: Vec<[Fp128Limbs; 3]>,
    pub(crate) coefficient_counts: Vec<usize>,
    pub(crate) challenges: Vec<Fp128Limbs>,
    pub(crate) final_evaluation: Fp128Limbs,
    pub(crate) final_linear_evaluation: Fp128Limbs,
    pub(crate) chaining_value: [u8; 64],
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct DirectRelationSession {
    compact_digits: Buffer,
    tables: [Buffer; 2],
    partials: Buffer,
    round_output: Buffer,
    additional_output: Buffer,
    final_output: Buffer,
    linear_final_output: Buffer,
    linear_segments: Buffer,
    lane_offsets: Buffer,
    lane_segments: Buffer,
    linear_source_lane_offsets: Buffer,
    linear_tables: [Buffer; 2],
    current_linear_table: usize,
    linear_mode: usize,
    linear_source_lane_count: usize,
    linear_current_coeff_count: usize,
    linear_current_live_lane_count: usize,
    lane_weight_tables: [Buffer; 2],
    two_round_prefix_partials: Buffer,
    two_round_prefix_output: Buffer,
    two_round_prefix_max_workgroups: usize,
    live_len: usize,
    current_len: usize,
    current_live_len: usize,
    current_table: Option<usize>,
    current_lane_weight_table: usize,
    current_lane_count: usize,
    coefficient_rounds: usize,
    compact_prefix_rounds: usize,
    rounds_folded: usize,
    allocation_bytes: usize,
}

struct DirectRelationRoundBuffers {
    e_first: Buffer,
    e_second: Buffer,
    alpha: Buffer,
    allocation_bytes: usize,
}

#[derive(Clone, Copy)]
enum Fp128KernelBinding<'a> {
    Inline(Fp128Limbs),
    Buffer(&'a Buffer, u64),
}

struct PackedLaneBuffer<'a> {
    buffer: Buffer,
    zero_copy: bool,
    marker: PhantomData<&'a [u8]>,
}

struct SharedByteBuffer<'a> {
    buffer: Buffer,
    zero_copy: bool,
    marker: PhantomData<&'a [u8]>,
}

pub(crate) struct SharedSliceBuffer<'a, T> {
    pub(crate) buffer: Buffer,
    pub(crate) zero_copy: bool,
    marker: PhantomData<&'a [T]>,
}

struct D512LinearRelationResources {
    primes: Buffer,
    fwd_twiddles: Buffer,
    inv_twiddles: Buffer,
    d_inv: Buffer,
    limb_weights: Buffer,
    field_moduli: Buffer,
    garner_gamma: Buffer,
    field_partial_products: Buffer,
}

struct RecursiveCommitResources {
    ring_d: usize,
    primes: Buffer,
    fwd_twiddles: Buffer,
    inv_twiddles: Buffer,
    psi_pows: Buffer,
    inverse_scale: Buffer,
    limb_weights: Buffer,
    field_moduli: Buffer,
    garner_gamma: Buffer,
    field_partial_products: Buffer,
}

pub(crate) struct MetalRuntime {
    device: Device,
    queue: CommandQueue,
    direct_pipeline: ComputePipelineState,
    block_batched_pipeline: ComputePipelineState,
    packed_fp128_d512_pipeline: ComputePipelineState,
    packed_fp128_d128_rank3_pipeline: ComputePipelineState,
    packed_partial_reduction_pipeline: ComputePipelineState,
    fp128_d64_digit_rows_partials_pipeline: ComputePipelineState,
    fp128_d64_digit_rows_reduce_pipeline: ComputePipelineState,
    fp128_i8_coefficient_packing_pipeline: ComputePipelineState,
    fp128_packed_onehot_coefficient_packing_partials_pipeline: ComputePipelineState,
    fp128_packed_onehot_coefficient_packing_reduce_pipeline: ComputePipelineState,
    fp128_d512_decompose_fold_pipeline: ComputePipelineState,
    fp128_d128_decompose_fold_pipeline: ComputePipelineState,
    fp128_d128_subring64_decompose_fold_pipeline: ComputePipelineState,
    fp128_d512_subring64_decompose_fold_pipeline: ComputePipelineState,
    fp128_d512_build_fold_index_pipeline: ComputePipelineState,
    fp128_d512_build_coefficient_packing_index_pipeline: ComputePipelineState,
    fp128_d512_indexed_coefficient_packing_pipeline: ComputePipelineState,
    fp128_d512_indexed_coefficient_packing_reduce_pipeline: ComputePipelineState,
    fp128_d512_indexed_subring64_decompose_fold_pipeline: ComputePipelineState,
    fp128_d512_fused_subring64_decompose_fold_pipeline: ComputePipelineState,
    fp128_d512_linear_relation_partials_pipeline: ComputePipelineState,
    fp128_d512_linear_relation_reduce_pipeline: ComputePipelineState,
    fp128_d512_linear_relation_reconstruct_pipeline: ComputePipelineState,
    fp128_d512_linear_relation_resources: D512LinearRelationResources,
    fp128_recursive_commit_matrix_ntt_pipeline: ComputePipelineState,
    fp128_recursive_commit_matvec_pipeline: ComputePipelineState,
    fp128_recursive_commit_reconstruct_pipeline: ComputePipelineState,
    fp128_recursive_commit_resources: [RecursiveCommitResources; 2],
    fp128_direct_range_initial_pipeline: ComputePipelineState,
    fp128_direct_range_compact_fold_pipeline: ComputePipelineState,
    fp128_direct_range_field_fold_pipeline: ComputePipelineState,
    fp128_direct_range_reduce_pipeline: ComputePipelineState,
    fp128_direct_range_finalize_pipeline: ComputePipelineState,
    fp128_blake2b_sumcheck_challenge_pipeline: ComputePipelineState,
    fp128_blake2b_relation_sumcheck_round_pipeline: ComputePipelineState,
    fp128_direct_relation_initial_pipeline: ComputePipelineState,
    fp128_direct_relation_compact_fold_pipeline: ComputePipelineState,
    fp128_direct_relation_field_fold_pipeline: ComputePipelineState,
    fp128_direct_relation_additional_compact_pipeline: ComputePipelineState,
    fp128_direct_relation_additional_field_pipeline: ComputePipelineState,
    fp128_direct_relation_two_round_prefix_pipeline: ComputePipelineState,
    fp128_direct_relation_two_round_prefix_reduce_pipeline: ComputePipelineState,
    fp128_direct_relation_linear_fold_pipeline: ComputePipelineState,
    fp128_direct_relation_alpha_fold_pipeline: ComputePipelineState,
    fp128_direct_relation_scalar_advance_pipeline: ComputePipelineState,
    fp128_direct_relation_additional_fold_pipeline: ComputePipelineState,
    fp128_direct_relation_setup_source_pipeline: ComputePipelineState,
    fp128_direct_relation_sparse_source_pipeline: ComputePipelineState,
}

impl MetalRuntime {
    pub(crate) fn new() -> Result<Self, MetalCommitError> {
        let device = Device::system_default().ok_or(MetalCommitError::DeviceUnavailable)?;
        Self::from_device(device)
    }

    pub(crate) fn from_device(device: Device) -> Result<Self, MetalCommitError> {
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(KERNEL_SOURCE, &options)
            .map_err(MetalCommitError::LibraryCompilation)?;
        let pipeline = |name| {
            let function = library
                .get_function(name, None)
                .map_err(|message| MetalCommitError::FunctionLookup { name, message })?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|message| MetalCommitError::PipelineCompilation { name, message })
        };
        let fp128_d512_linear_relation_resources = d512_linear_relation_resources(&device);
        let fp128_recursive_commit_resources = [
            recursive_commit_resources(&device, 64),
            recursive_commit_resources(&device, 128),
        ];
        Ok(Self {
            queue: device.new_command_queue(),
            direct_pipeline: pipeline(DIRECT_KERNEL_NAME)?,
            block_batched_pipeline: pipeline(BLOCK_BATCHED_KERNEL_NAME)?,
            packed_fp128_d512_pipeline: pipeline(PACKED_FP128_D512_PANELS_KERNEL_NAME)?,
            packed_fp128_d128_rank3_pipeline: pipeline(PACKED_FP128_D128_RANK3_KERNEL_NAME)?,
            packed_partial_reduction_pipeline: pipeline(PACKED_PARTIAL_REDUCTION_KERNEL_NAME)?,
            fp128_d64_digit_rows_partials_pipeline: pipeline(
                FP128_D64_DIGIT_ROWS_PARTIALS_KERNEL_NAME,
            )?,
            fp128_d64_digit_rows_reduce_pipeline: pipeline(
                FP128_D64_DIGIT_ROWS_REDUCE_KERNEL_NAME,
            )?,
            fp128_i8_coefficient_packing_pipeline: pipeline(
                FP128_I8_COEFFICIENT_PACKING_KERNEL_NAME,
            )?,
            fp128_packed_onehot_coefficient_packing_partials_pipeline: pipeline(
                FP128_PACKED_ONEHOT_COEFFICIENT_PACKING_PARTIALS_KERNEL_NAME,
            )?,
            fp128_packed_onehot_coefficient_packing_reduce_pipeline: pipeline(
                FP128_PACKED_ONEHOT_COEFFICIENT_PACKING_REDUCE_KERNEL_NAME,
            )?,
            fp128_d512_decompose_fold_pipeline: pipeline(FP128_D512_DECOMPOSE_FOLD_KERNEL_NAME)?,
            fp128_d128_decompose_fold_pipeline: pipeline(FP128_D128_DECOMPOSE_FOLD_KERNEL_NAME)?,
            fp128_d128_subring64_decompose_fold_pipeline: pipeline(
                FP128_D128_SUBRING64_DECOMPOSE_FOLD_KERNEL_NAME,
            )?,
            fp128_d512_subring64_decompose_fold_pipeline: pipeline(
                FP128_D512_SUBRING64_DECOMPOSE_FOLD_KERNEL_NAME,
            )?,
            fp128_d512_build_fold_index_pipeline: pipeline(
                FP128_D512_BUILD_FOLD_INDEX_KERNEL_NAME,
            )?,
            fp128_d512_build_coefficient_packing_index_pipeline: pipeline(
                FP128_D512_BUILD_COEFFICIENT_PACKING_INDEX_KERNEL_NAME,
            )?,
            fp128_d512_indexed_coefficient_packing_pipeline: pipeline(
                FP128_D512_INDEXED_COEFFICIENT_PACKING_KERNEL_NAME,
            )?,
            fp128_d512_indexed_coefficient_packing_reduce_pipeline: pipeline(
                FP128_D512_INDEXED_COEFFICIENT_PACKING_REDUCE_KERNEL_NAME,
            )?,
            fp128_d512_indexed_subring64_decompose_fold_pipeline: pipeline(
                FP128_D512_INDEXED_SUBRING64_DECOMPOSE_FOLD_KERNEL_NAME,
            )?,
            fp128_d512_fused_subring64_decompose_fold_pipeline: pipeline(
                FP128_D512_FUSED_SUBRING64_DECOMPOSE_FOLD_KERNEL_NAME,
            )?,
            fp128_d512_linear_relation_partials_pipeline: pipeline(
                FP128_D512_LINEAR_RELATION_PARTIALS_KERNEL_NAME,
            )?,
            fp128_d512_linear_relation_reduce_pipeline: pipeline(
                FP128_D512_LINEAR_RELATION_REDUCE_KERNEL_NAME,
            )?,
            fp128_d512_linear_relation_reconstruct_pipeline: pipeline(
                FP128_D512_LINEAR_RELATION_RECONSTRUCT_KERNEL_NAME,
            )?,
            fp128_d512_linear_relation_resources,
            fp128_recursive_commit_matrix_ntt_pipeline: pipeline(
                FP128_RECURSIVE_COMMIT_MATRIX_NTT_KERNEL_NAME,
            )?,
            fp128_recursive_commit_matvec_pipeline: pipeline(
                FP128_RECURSIVE_COMMIT_MATVEC_KERNEL_NAME,
            )?,
            fp128_recursive_commit_reconstruct_pipeline: pipeline(
                FP128_RECURSIVE_COMMIT_RECONSTRUCT_KERNEL_NAME,
            )?,
            fp128_recursive_commit_resources,
            fp128_direct_range_initial_pipeline: pipeline(FP128_DIRECT_RANGE_INITIAL_KERNEL_NAME)?,
            fp128_direct_range_compact_fold_pipeline: pipeline(
                FP128_DIRECT_RANGE_COMPACT_FOLD_KERNEL_NAME,
            )?,
            fp128_direct_range_field_fold_pipeline: pipeline(
                FP128_DIRECT_RANGE_FIELD_FOLD_KERNEL_NAME,
            )?,
            fp128_direct_range_reduce_pipeline: pipeline(FP128_DIRECT_RANGE_REDUCE_KERNEL_NAME)?,
            fp128_direct_range_finalize_pipeline: pipeline(
                FP128_DIRECT_RANGE_FINALIZE_KERNEL_NAME,
            )?,
            fp128_blake2b_sumcheck_challenge_pipeline: pipeline(
                FP128_BLAKE2B_SUMCHECK_CHALLENGE_KERNEL_NAME,
            )?,
            fp128_blake2b_relation_sumcheck_round_pipeline: pipeline(
                FP128_BLAKE2B_RELATION_SUMCHECK_ROUND_KERNEL_NAME,
            )?,
            fp128_direct_relation_initial_pipeline: pipeline(
                FP128_DIRECT_RELATION_INITIAL_KERNEL_NAME,
            )?,
            fp128_direct_relation_compact_fold_pipeline: pipeline(
                FP128_DIRECT_RELATION_COMPACT_FOLD_KERNEL_NAME,
            )?,
            fp128_direct_relation_field_fold_pipeline: pipeline(
                FP128_DIRECT_RELATION_FIELD_FOLD_KERNEL_NAME,
            )?,
            fp128_direct_relation_additional_compact_pipeline: pipeline(
                FP128_DIRECT_RELATION_ADDITIONAL_COMPACT_KERNEL_NAME,
            )?,
            fp128_direct_relation_additional_field_pipeline: pipeline(
                FP128_DIRECT_RELATION_ADDITIONAL_FIELD_KERNEL_NAME,
            )?,
            fp128_direct_relation_two_round_prefix_pipeline: pipeline(
                FP128_DIRECT_RELATION_TWO_ROUND_PREFIX_KERNEL_NAME,
            )?,
            fp128_direct_relation_two_round_prefix_reduce_pipeline: pipeline(
                FP128_DIRECT_RELATION_TWO_ROUND_PREFIX_REDUCE_KERNEL_NAME,
            )?,
            fp128_direct_relation_linear_fold_pipeline: pipeline(
                FP128_DIRECT_RELATION_LINEAR_FOLD_KERNEL_NAME,
            )?,
            fp128_direct_relation_alpha_fold_pipeline: pipeline(
                FP128_DIRECT_RELATION_ALPHA_FOLD_KERNEL_NAME,
            )?,
            fp128_direct_relation_scalar_advance_pipeline: pipeline(
                FP128_DIRECT_RELATION_SCALAR_ADVANCE_KERNEL_NAME,
            )?,
            fp128_direct_relation_additional_fold_pipeline: pipeline(
                FP128_DIRECT_RELATION_ADDITIONAL_FOLD_KERNEL_NAME,
            )?,
            fp128_direct_relation_setup_source_pipeline: pipeline(
                FP128_DIRECT_RELATION_SETUP_SOURCE_KERNEL_NAME,
            )?,
            fp128_direct_relation_sparse_source_pipeline: pipeline(
                FP128_DIRECT_RELATION_SPARSE_SOURCE_KERNEL_NAME,
            )?,
            device,
        })
    }

    pub(crate) fn capabilities(&self) -> MetalDeviceCapabilities {
        MetalDeviceCapabilities {
            name: self.device.name().to_owned(),
            max_buffer_length: self.device.max_buffer_length(),
            recommended_max_working_set_size: self.device.recommended_max_working_set_size(),
            max_threadgroup_memory_length: self.device.max_threadgroup_memory_length(),
            thread_execution_width: self.block_batched_pipeline.thread_execution_width() as usize,
            max_total_threads_per_threadgroup: self
                .block_batched_pipeline
                .max_total_threads_per_threadgroup()
                as usize,
            static_threadgroup_memory_length: self
                .block_batched_pipeline
                .static_threadgroup_memory_length(),
        }
    }

    pub(crate) fn supports_packed_fp128_d512_panels(&self) -> bool {
        self.packed_fp128_d512_pipeline
            .max_total_threads_per_threadgroup()
            >= FP128_D512_THREADS as u64
            && self
                .packed_fp128_d512_pipeline
                .static_threadgroup_memory_length()
                == FP128_D512_THREADGROUP_BYTES as u64
    }

    pub(crate) fn supports_packed_fp128_d128_rank3(&self) -> bool {
        self.packed_fp128_d128_rank3_pipeline
            .max_total_threads_per_threadgroup()
            >= FP128_D512_THREADS as u64
            && self
                .packed_fp128_d128_rank3_pipeline
                .static_threadgroup_memory_length()
                == FP128_D128_RANK3_THREADGROUP_BYTES as u64
    }
}
