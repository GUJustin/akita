#include <metal_stdlib>
using namespace metal;

#define AKITA_OFFSET 0xffffa7f7u
#define NONE_INDEX 0xffffu
#define MAX_DEFERRED_ACCUMULATIONS 32768u
#define PACKED_FP128_D512_PANEL_TILE_ELEMENTS 2048u
#define D512_LINEAR_NTT_SIZE 1024u
#define D512_LINEAR_NTT_HALF 512u
#define D512_LINEAR_NTT_PRIMES 6u
#define RECURSIVE_COMMIT_THREADS 1024u
#define SIMD_RECURSIVE_COMMIT_THREADS 512u
#define RECURSIVE_COMMIT_BLOCKS_PER_GROUP 16u
#define RECURSIVE_COMMIT_MAX_D 128u
#define D512_PACKING_TILES_PER_CHUNK 32u
#define RECURSIVE_COMMIT_MAX_ROWS 8u

struct AkitaFp128 {
    uint4 limb;
};

struct AkitaWide256 {
    uint limb[8];
};

struct AkitaHalfWidthWide192 {
    uint limb[6];
};

static_assert(
    sizeof(uint) * PACKED_FP128_D512_PANEL_TILE_ELEMENTS * 4 == 32768,
    "fp128 D512 panel tile must occupy 32 KiB");

struct AkitaCorrection {
    AkitaFp128 value;
    uint carry;
};

struct AkitaWideAccumulator {
    int4 low_digits;
    int4 high_digits;
};

struct AkitaTransposedFp128Accumulator {
    uint4 word_0;
    uint4 word_1;
    uint4 word_2;
    uint4 word_3;
    int4 wraps;
};

struct OneHotCommitParams {
    ulong num_sources;
    ulong chunks_per_source;
    ulong onehot_k;
    ulong ring_d;
    ulong n_a;
    ulong positions_per_block;
    ulong num_digits_inner;
    ulong num_blocks;
    ulong total_field_elements;
    ulong output_coefficients;
    ulong blocks_per_threadgroup;
    ulong log_onehot_k;
    ulong log_ring_d;
};

struct PackedOneHotCommitParams {
    ulong num_rows;
    ulong num_columns;
    ulong lane_stride;
    ulong column_capacity;
    ulong onehot_k;
    ulong ring_d;
    ulong n_a;
    ulong positions_per_block;
    ulong num_digits_inner;
    ulong blocks_per_column;
    ulong full_blocks_per_column;
    ulong boundary_columns;
    ulong num_blocks;
    ulong task_offset;
    ulong dispatch_tasks;
    ulong lane_row_offset;
    ulong output_coefficients;
    ulong columns_per_threadgroup;
    ulong position_partials_per_block;
    ulong positions_per_partial;
    ulong log_ring_d;
    ulong zero_column_mask;
};

struct DigitRowsParams {
    ulong num_vectors;
    ulong num_rows;
    ulong num_cols;
    ulong ring_d;
    ulong output_coefficients;
    ulong columns_per_partial;
    ulong column_partials;
    ulong retain_quotients;
};

struct I8CoefficientPackingParams {
    ulong num_sources;
    ulong source_coefficients;
    ulong live_coefficients;
    ulong num_live_positions;
    ulong positions_per_block;
    ulong num_blocks;
    ulong ring_d;
    ulong stride;
    ulong subring_dimension;
    ulong output_coefficients;
};

struct PackedOneHotCoefficientPackingParams {
    ulong num_rows;
    ulong num_columns;
    ulong column_capacity;
    ulong onehot_k;
    ulong ring_d;
    ulong positions_per_block;
    ulong blocks_per_column;
    ulong rows_per_block;
    ulong rows_per_partial;
    ulong row_partials_per_block;
    ulong num_blocks;
    ulong stride;
    ulong subring_dimension;
    ulong output_coefficients;
    ulong partial_coefficients;
    ulong zero_column_mask;
};

struct PackedDecomposeFoldParams {
    ulong num_rows;
    ulong num_columns;
    ulong lane_stride;
    ulong num_positions;
    ulong position_start;
    ulong blocks_per_column;
    ulong challenge_weight;
    ulong output_coefficients;
    ulong zero_column_mask;
    ulong onehot_k;
    ulong zero_suffix_start;
};

struct PackedFoldIndexParams {
    ulong num_rows;
    ulong num_columns;
    ulong lane_stride;
    ulong num_positions;
    ulong position_start;
    ulong blocks_per_column;
    ulong tasks_per_position;
    ulong tiles_per_position;
    ulong record_slots;
    ulong count_entries;
    ulong output_coefficients;
    ulong fold_digits;
    ulong fold_log_basis;
};

struct PackedCoefficientPackingIndexParams {
    ulong num_rows;
    ulong num_columns;
    ulong lane_stride;
    ulong num_positions;
    ulong blocks_per_column;
    ulong position_tiles;
    ulong record_slots;
    ulong offset_entries;
};

struct D512LinearRelationParams {
    ulong num_columns;
    ulong columns_per_tile;
    ulong num_tiles;
    ulong num_primes;
    ulong ntt_size;
    ulong output_coefficients;
    ulong rhs_abs_bound;
};

struct RecursiveCommitParams {
    ulong num_blocks;
    ulong blocks_per_group;
    ulong num_block_groups;
    ulong num_rows;
    ulong num_cols;
    ulong ring_d;
    ulong num_primes;
    ulong matrix_rings;
    ulong output_coefficients;
    ulong rhs_abs_bound;
};

struct D512LinearNttPrime {
    int p;
    int pinv;
    int mont;
    int montsq;
};

struct DirectRangeParams {
    ulong live_len;
    ulong current_len;
    ulong current_live_len;
    ulong input_live_len;
    ulong pair_count;
    ulong num_first;
    ulong num_second;
    ulong workgroups;
    ulong basis;
    ulong prefix_size;
    ulong materialize_prefix;
    ulong resident_challenges;
};

struct Blake2bSumcheckChallengeParams {
    ulong include_claim;
    ulong coefficient_count;
    ulong prior_squeezed_bytes;
    ulong reserved;
};

struct DirectRelationParams {
    ulong live_len;
    ulong current_len;
    ulong current_live_len;
    ulong input_live_len;
    ulong pair_count;
    ulong num_first;
    ulong num_second;
    ulong workgroups;
    ulong current_coeff_count;
    ulong live_lane_count;
    ulong prefix_size;
    ulong materialize_prefix;
    ulong linear_mode;
    ulong additional_pair_count;
    ulong additional_workgroups;
    ulong fold_lane_weights;
    ulong resident_challenges;
};

struct DirectRelationTranscriptParams {
    ulong prior_squeezed_bytes;
    ulong has_additional;
};

struct DirectRelationTwoRoundPrefixParams {
    ulong live_lane_count;
    ulong coefficient_count;
    ulong y_quads;
    ulong equality_first_len;
    ulong workgroups;
    ulong lanes_per_thread;
    ulong norm_omitted_corner;
    ulong linear_mode;
};

struct DirectRelationLinearFoldParams {
    ulong current_coeff_count;
    ulong source_lane_count;
    ulong current_live_lane_count;
    ulong output_len;
    ulong mode;
};

static_assert(sizeof(DirectRelationLinearFoldParams) == 40);

struct DirectRelationReducedSourceParams {
    ulong ring_dimension;
    ulong row_count;
    ulong item_count;
    ulong reserved;
    AkitaFp128 alpha;
    AkitaFp128 wrap_correction;
};

static_assert(sizeof(DirectRelationReducedSourceParams) == 64);

struct DirectRelationScalars {
    AkitaFp128 l_at_0;
    AkitaFp128 l_at_1;
    AkitaFp128 binary_batching;
};

struct DirectRelationLinearSegment {
    AkitaFp128 factor;
    uint source_index;
    uint target_lane_start;
    uint target_lane_stride;
    uint source_lane_start;
    uint source_lane_stride;
    uint lane_count;
};

static_assert(sizeof(DirectRelationLinearSegment) == 48);

struct DirectRelationAdditionalPair {
    ulong parent;
    ulong reserved;
    AkitaFp128 linear[2];
    AkitaFp128 binary[2];
};

struct DirectRelationAdditionalFoldMapping {
    ulong parent;
    uint left;
    uint right;
};

inline AkitaFp128 akita_zero() {
    AkitaFp128 result;
    result.limb = uint4(0u);
    return result;
}

inline AkitaFp128 akita_select(bool take_lhs, AkitaFp128 lhs, AkitaFp128 rhs) {
    AkitaFp128 result;
    uint mask = take_lhs ? 0xffffffffu : 0u;
    result.limb = (lhs.limb & mask) | (rhs.limb & ~mask);
    return result;
}

inline AkitaCorrection akita_add_offset(AkitaFp128 value) {
    AkitaCorrection result;
    ulong sum = (ulong)value.limb[0] + (ulong)AKITA_OFFSET;
    result.value.limb[0] = (uint)sum;
    ulong carry = sum >> 32;
    for (uint i = 1; i < 4; ++i) {
        sum = (ulong)value.limb[i] + carry;
        result.value.limb[i] = (uint)sum;
        carry = sum >> 32;
    }
    result.carry = (uint)carry;
    return result;
}

inline AkitaFp128 akita_sub_offset(AkitaFp128 value) {
    AkitaFp128 result;
    ulong subtrahend = (ulong)AKITA_OFFSET;
    for (uint i = 0; i < 4; ++i) {
        ulong word = (ulong)value.limb[i];
        result.limb[i] = (uint)(word - subtrahend);
        subtrahend = word < subtrahend ? 1ul : 0ul;
    }
    return result;
}

inline AkitaFp128 akita_add(AkitaFp128 lhs, AkitaFp128 rhs) {
    AkitaFp128 sum;
    ulong carry = 0ul;
    for (uint i = 0; i < 4; ++i) {
        ulong word = (ulong)lhs.limb[i] + (ulong)rhs.limb[i] + carry;
        sum.limb[i] = (uint)word;
        carry = word >> 32;
    }
    AkitaCorrection corrected = akita_add_offset(sum);
    return akita_select(carry != 0ul || corrected.carry != 0u, corrected.value, sum);
}

inline AkitaFp128 akita_sub(AkitaFp128 lhs, AkitaFp128 rhs) {
    AkitaFp128 difference;
    ulong borrow = 0ul;
    for (uint i = 0; i < 4; ++i) {
        ulong subtrahend = (ulong)rhs.limb[i] + borrow;
        ulong word = (ulong)lhs.limb[i];
        difference.limb[i] = (uint)(word - subtrahend);
        borrow = word < subtrahend ? 1ul : 0ul;
    }
    AkitaFp128 corrected = akita_sub_offset(difference);
    return akita_select(borrow != 0ul, corrected, difference);
}

inline AkitaWide256 akita_product_wide(AkitaFp128 lhs, AkitaFp128 rhs) {
    AkitaWide256 product;
    for (uint i = 0u; i < 8u; ++i) {
        product.limb[i] = 0u;
    }
    for (uint i = 0u; i < 4u; ++i) {
        ulong carry = 0ul;
        for (uint j = 0u; j < 4u; ++j) {
            uint k = i + j;
            ulong word = (ulong)lhs.limb[i] * (ulong)rhs.limb[j]
                + (ulong)product.limb[k]
                + carry;
            product.limb[k] = (uint)word;
            carry = word >> 32u;
        }
        product.limb[i + 4u] = (uint)carry;
    }
    return product;
}

inline AkitaFp128 akita_reduce_product(AkitaWide256 product) {
    AkitaFp128 folded;
    ulong carry = 0ul;
    for (uint i = 0u; i < 4u; ++i) {
        ulong word = (ulong)product.limb[i + 4u] * (ulong)AKITA_OFFSET
            + (ulong)product.limb[i]
            + carry;
        folded.limb[i] = (uint)word;
        carry = word >> 32u;
    }

    ulong word = (ulong)folded.limb[0] + carry * (ulong)AKITA_OFFSET;
    folded.limb[0] = (uint)word;
    carry = word >> 32u;
    for (uint i = 1u; i < 4u; ++i) {
        word = (ulong)folded.limb[i] + carry;
        folded.limb[i] = (uint)word;
        carry = word >> 32u;
    }

    AkitaCorrection corrected = akita_add_offset(folded);
    return akita_select(carry != 0ul || corrected.carry != 0u, corrected.value, folded);
}

inline AkitaFp128 akita_mul(AkitaFp128 lhs, AkitaFp128 rhs) {
    return akita_reduce_product(akita_product_wide(lhs, rhs));
}

inline AkitaHalfWidthWide192 akita_half_width_product_u64(
    AkitaFp128 lhs,
    ulong rhs)
{
    uint rhs_lo = (uint)rhs;
    uint rhs_hi = (uint)(rhs >> 32u);
    AkitaHalfWidthWide192 product;
    for (uint i = 0u; i < 6u; ++i) {
        product.limb[i] = 0u;
    }

    ulong carry = 0ul;
    ulong word = (ulong)lhs.limb[0] * (ulong)rhs_lo;
    product.limb[0] = (uint)word;
    carry = word >> 32u;
    word = (ulong)lhs.limb[0] * (ulong)rhs_hi + carry;
    product.limb[1] = (uint)word;
    product.limb[2] = (uint)(word >> 32u);

    word = (ulong)lhs.limb[1] * (ulong)rhs_lo + (ulong)product.limb[1];
    product.limb[1] = (uint)word;
    carry = word >> 32u;
    word = (ulong)lhs.limb[1] * (ulong)rhs_hi
        + (ulong)product.limb[2]
        + carry;
    product.limb[2] = (uint)word;
    product.limb[3] = (uint)(word >> 32u);

    word = (ulong)lhs.limb[2] * (ulong)rhs_lo + (ulong)product.limb[2];
    product.limb[2] = (uint)word;
    carry = word >> 32u;
    word = (ulong)lhs.limb[2] * (ulong)rhs_hi
        + (ulong)product.limb[3]
        + carry;
    product.limb[3] = (uint)word;
    product.limb[4] = (uint)(word >> 32u);

    word = (ulong)lhs.limb[3] * (ulong)rhs_lo + (ulong)product.limb[3];
    product.limb[3] = (uint)word;
    carry = word >> 32u;
    word = (ulong)lhs.limb[3] * (ulong)rhs_hi
        + (ulong)product.limb[4]
        + carry;
    product.limb[4] = (uint)word;
    product.limb[5] = (uint)(word >> 32u);
    return product;
}

inline AkitaFp128 akita_half_width_reduce_u192(AkitaHalfWidthWide192 product) {
    AkitaFp128 folded;
    ulong word = (ulong)product.limb[4] * (ulong)AKITA_OFFSET
        + (ulong)product.limb[0];
    folded.limb[0] = (uint)word;
    ulong carry = word >> 32u;
    word = (ulong)product.limb[5] * (ulong)AKITA_OFFSET
        + (ulong)product.limb[1]
        + carry;
    folded.limb[1] = (uint)word;
    carry = word >> 32u;
    word = (ulong)product.limb[2] + carry;
    folded.limb[2] = (uint)word;
    carry = word >> 32u;
    word = (ulong)product.limb[3] + carry;
    folded.limb[3] = (uint)word;
    ulong first_fold_carry = word >> 32u;

    word = (ulong)folded.limb[0] + first_fold_carry * (ulong)AKITA_OFFSET;
    folded.limb[0] = (uint)word;
    carry = word >> 32u;
    for (uint i = 1u; i < 4u; ++i) {
        word = (ulong)folded.limb[i] + carry;
        folded.limb[i] = (uint)word;
        carry = word >> 32u;
    }
    AkitaCorrection corrected = akita_add_offset(folded);
    return akita_select(corrected.carry != 0u, corrected.value, folded);
}

inline AkitaFp128 akita_mul_signed_i32(AkitaFp128 value, int scalar) {
    uint magnitude = (uint)(scalar < 0 ? -scalar : scalar);
    uint product[5];
    ulong carry = 0ul;
    for (uint i = 0u; i < 4u; ++i) {
        ulong word = (ulong)value.limb[i] * (ulong)magnitude + carry;
        product[i] = (uint)word;
        carry = word >> 32u;
    }
    product[4] = (uint)carry;

    AkitaFp128 folded;
    ulong word = (ulong)product[0] + (ulong)product[4] * (ulong)AKITA_OFFSET;
    folded.limb[0] = (uint)word;
    carry = word >> 32u;
    for (uint i = 1u; i < 4u; ++i) {
        word = (ulong)product[i] + carry;
        folded.limb[i] = (uint)word;
        carry = word >> 32u;
    }
    word = (ulong)folded.limb[0] + carry * (ulong)AKITA_OFFSET;
    folded.limb[0] = (uint)word;
    carry = word >> 32u;
    for (uint i = 1u; i < 4u; ++i) {
        word = (ulong)folded.limb[i] + carry;
        folded.limb[i] = (uint)word;
        carry = word >> 32u;
    }
    AkitaCorrection corrected = akita_add_offset(folded);
    AkitaFp128 result = akita_select(
        carry != 0ul || corrected.carry != 0u, corrected.value, folded);
    return scalar < 0 ? akita_sub(akita_zero(), result) : result;
}

inline AkitaFp128 akita_mul_signed_small(AkitaFp128 value, long scalar) {
    ulong magnitude = (ulong)(scalar < 0l ? -scalar : scalar);
    AkitaFp128 result = akita_half_width_reduce_u192(
        akita_half_width_product_u64(value, magnitude));
    return scalar < 0l ? akita_sub(akita_zero(), result) : result;
}

inline AkitaFp128 akita_from_u32(uint value) {
    AkitaFp128 result = akita_zero();
    result.limb[0] = value;
    return result;
}

inline int d512_ntt_reduce(long value, int modulus) {
    if (value >= (long)modulus) {
        value -= (long)modulus;
    }
    if (value < 0l) {
        value += (long)modulus;
    }
    return (int)value;
}

inline int d512_ntt_add(int lhs, int rhs, int modulus) {
    return d512_ntt_reduce((long)lhs + (long)rhs, modulus);
}

inline int d512_ntt_sub(int lhs, int rhs, int modulus) {
    return d512_ntt_reduce((long)lhs - (long)rhs, modulus);
}

inline int d512_ntt_mul_raw(int lhs, int rhs, D512LinearNttPrime prime) {
    long product = (long)lhs * (long)rhs;
    uint low = (uint)product;
    int correction = as_type<int>(low * as_type<uint>(prime.pinv));
    return (int)((product - (long)correction * (long)prime.p) >> 32u);
}

inline int d512_ntt_mul(int lhs, int rhs, D512LinearNttPrime prime) {
    return d512_ntt_reduce((long)d512_ntt_mul_raw(lhs, rhs, prime), prime.p);
}

inline uint d512_reduce_u32(uint value, uint modulus) {
    value = value >= modulus ? value - modulus : value;
    value = value >= modulus ? value - modulus : value;
    value = value >= modulus ? value - modulus : value;
    value = value >= modulus ? value - modulus : value;
    return value;
}

inline int d512_u32_to_mont(uint value, D512LinearNttPrime prime) {
    uint canonical = d512_reduce_u32(value, (uint)prime.p);
    return d512_ntt_mul((int)canonical, prime.montsq, prime);
}

inline bool d512_fp128_above_half(AkitaFp128 value) {
    const uint half_limbs[4] = {
        0x80002c04u,
        0xffffffffu,
        0xffffffffu,
        0x7fffffffu,
    };
    for (int limb = 3; limb >= 0; --limb) {
        if (value.limb[limb] != half_limbs[limb]) {
            return value.limb[limb] > half_limbs[limb];
        }
    }
    return false;
}

inline int d512_fp128_to_mont(
    AkitaFp128 value,
    uint prime_index,
    D512LinearNttPrime prime,
    device const int *limb_weights,
    device const int *field_moduli)
{
    int residue = 0;
    ulong weight_base = (ulong)prime_index * 4ul;
    for (uint limb = 0u; limb < 4u; ++limb) {
        int canonical = d512_u32_to_mont(value.limb[limb], prime);
        residue = d512_ntt_add(
            residue,
            d512_ntt_mul(canonical, limb_weights[weight_base + limb], prime),
            prime.p);
    }
    if (d512_fp128_above_half(value)) {
        residue = d512_ntt_sub(residue, field_moduli[prime_index], prime.p);
    }
    return residue;
}

inline int d512_i32_to_mont(int value, D512LinearNttPrime prime) {
    long canonical = value < 0
        ? (long)prime.p + (long)value
        : (long)value;
    return d512_ntt_mul((int)canonical, prime.montsq, prime);
}

inline long d512_positive_mod(long value, long modulus) {
    long result = value % modulus;
    return result < 0l ? result + modulus : result;
}

inline AkitaWideAccumulator akita_wide_zero() {
    AkitaWideAccumulator result;
    result.low_digits = int4(0);
    result.high_digits = int4(0);
    return result;
}

inline void akita_wide_accumulate(
    thread AkitaWideAccumulator &accumulator,
    AkitaFp128 value,
    bool positive)
{
    int sign = positive ? 1 : -1;
    accumulator.low_digits += sign * int4(value.limb & uint4(0xffffu));
    accumulator.high_digits += sign * int4(value.limb >> 16u);
}

inline void akita_wide_accumulate_scaled(
    thread AkitaWideAccumulator &accumulator,
    AkitaFp128 value,
    int scale)
{
    accumulator.low_digits += scale * int4(value.limb & uint4(0xffffu));
    accumulator.high_digits += scale * int4(value.limb >> 16u);
}

inline AkitaFp128 akita_reduce_wide(AkitaWideAccumulator accumulator) {
    AkitaFp128 base;
    long carry = 0l;
    for (uint word = 0u; word < 4u; ++word) {
        long low = (long)accumulator.low_digits[word] + carry;
        uint low_digit = (uint)low & 0xffffu;
        carry = low >> 16;
        long high = (long)accumulator.high_digits[word] + carry;
        uint high_digit = (uint)high & 0xffffu;
        carry = high >> 16;
        base.limb[word] = low_digit | (high_digit << 16u);
    }

    AkitaCorrection canonical = akita_add_offset(base);
    base = akita_select(canonical.carry != 0u, canonical.value, base);
    if (carry == 0l) {
        return base;
    }

    ulong magnitude = (ulong)(carry > 0l ? carry : -carry);
    ulong correction_word = magnitude * (ulong)AKITA_OFFSET;
    AkitaFp128 correction = akita_zero();
    correction.limb[0] = (uint)correction_word;
    correction.limb[1] = (uint)(correction_word >> 32u);
    return carry > 0l
        ? akita_add(base, correction)
        : akita_sub(base, correction);
}

inline AkitaTransposedFp128Accumulator akita_transposed_fp128_zero() {
    AkitaTransposedFp128Accumulator result;
    result.word_0 = uint4(0u);
    result.word_1 = uint4(0u);
    result.word_2 = uint4(0u);
    result.word_3 = uint4(0u);
    result.wraps = int4(0);
    return result;
}

inline AkitaFp128 akita_reduce_transposed_fp128(
    AkitaTransposedFp128Accumulator accumulator,
    uint component)
{
    AkitaFp128 base;
    base.limb = uint4(
        accumulator.word_0[component],
        accumulator.word_1[component],
        accumulator.word_2[component],
        accumulator.word_3[component]);
    AkitaCorrection canonical = akita_add_offset(base);
    base = akita_select(canonical.carry != 0u, canonical.value, base);

    long wraps = (long)accumulator.wraps[component];
    if (wraps == 0l) {
        return base;
    }
    ulong magnitude = (ulong)(wraps > 0l ? wraps : -wraps);
    ulong correction_word = magnitude * (ulong)AKITA_OFFSET;
    AkitaFp128 correction = akita_zero();
    correction.limb[0] = (uint)correction_word;
    correction.limb[1] = (uint)(correction_word >> 32u);
    return wraps > 0l
        ? akita_add(base, correction)
        : akita_sub(base, correction);
}

inline uint4 akita_add_transposed_word(
    uint4 lhs,
    uint4 rhs,
    thread uint4 &carry)
{
    uint4 base = lhs + rhs;
    uint4 sum = base + carry;
    carry = select(
        uint4(0u),
        uint4(1u),
        (base < lhs) | (sum < base));
    return sum;
}

kernel void akita_packed_onehot_reduce_partials(
    device const AkitaFp128 *partials [[buffer(0)]],
    device AkitaFp128 *output [[buffer(1)]],
    constant PackedOneHotCommitParams &params [[buffer(2)]],
    uint output_index [[thread_position_in_grid]])
{
    if ((ulong)output_index >= params.output_coefficients) {
        return;
    }
    ulong coefficients_per_block = params.n_a * params.ring_d;
    ulong block = (ulong)output_index / coefficients_per_block;
    ulong column = block / params.blocks_per_column;
    ulong column_block = block % params.blocks_per_column;
    if (column >= params.num_columns
        || column_block >= params.full_blocks_per_column) {
        output[output_index] = akita_zero();
        return;
    }
    AkitaFp128 accumulator = partials[output_index];
    for (ulong partial = 1ul;
         partial < params.position_partials_per_block;
         ++partial) {
        ulong partial_index =
            partial * params.output_coefficients + (ulong)output_index;
        accumulator = akita_add(accumulator, partials[partial_index]);
    }
    output[output_index] = accumulator;
}

kernel void akita_fp128_d64_digit_rows_partials(
    device const AkitaFp128 *matrix [[buffer(0)]],
    device const char *digits [[buffer(1)]],
    device AkitaFp128 *partials [[buffer(2)]],
    constant DigitRowsParams &params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup AkitaFp128 matrix_ring[64];
    threadgroup char digit_ring[64];

    uint partial_index = threadgroup_index.x;
    uint partials_per_vector =
        (uint)params.num_rows * (uint)params.column_partials;
    uint vector = partial_index / partials_per_vector;
    uint vector_local = partial_index % partials_per_vector;
    uint row = vector_local / (uint)params.column_partials;
    uint partial = vector_local % (uint)params.column_partials;
    uint column_start = partial * (uint)params.columns_per_partial;
    uint column_end = min(
        column_start + (uint)params.columns_per_partial,
        (uint)params.num_cols);
    uint coefficient = thread_index;
    AkitaWideAccumulator accumulator = akita_wide_zero();
    AkitaWideAccumulator quotient = akita_wide_zero();
    for (uint column = column_start; column < column_end; ++column) {
        ulong ring_start =
            ((ulong)row * params.num_cols + (ulong)column) * 64ul;
        matrix_ring[thread_index] = matrix[ring_start + (ulong)thread_index];
        digit_ring[thread_index] = digits[
            ((ulong)vector * params.num_cols + (ulong)column) * 64ul
                + (ulong)thread_index];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint digit_coefficient = 0u; digit_coefficient < 64u; ++digit_coefficient) {
            int digit = (int)digit_ring[digit_coefficient];
            if (digit == 0) {
                continue;
            }
            bool wraps = digit_coefficient > coefficient;
            uint source_coefficient =
                (coefficient + 64u - digit_coefficient) & 63u;
            AkitaFp128 value = matrix_ring[source_coefficient];
            akita_wide_accumulate_scaled(
                accumulator,
                value,
                wraps ? -digit : digit);
            if (params.retain_quotients != 0ul && wraps) {
                akita_wide_accumulate_scaled(quotient, value, digit);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    ulong output_index =
        (((ulong)vector * params.num_rows + (ulong)row)
            * params.column_partials + (ulong)partial) * 64ul
        + (ulong)coefficient;
    partials[output_index] = akita_reduce_wide(accumulator);
    if (params.retain_quotients != 0ul) {
        ulong partial_coefficients =
            params.num_vectors * params.num_rows * params.column_partials * 64ul;
        partials[partial_coefficients + output_index] = akita_reduce_wide(quotient);
    }
}

kernel void akita_fp128_d64_digit_rows_reduce(
    device const AkitaFp128 *partials [[buffer(0)]],
    device AkitaFp128 *output [[buffer(1)]],
    constant DigitRowsParams &params [[buffer(2)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup AkitaFp128 reduction[256];

    uint output_index = threadgroup_index.x;
    uint product = (uint)((ulong)output_index / params.output_coefficients);
    uint product_output_index =
        (uint)((ulong)output_index % params.output_coefficients);
    uint outputs_per_vector = (uint)params.num_rows * 64u;
    uint vector = product_output_index / outputs_per_vector;
    uint vector_local = product_output_index % outputs_per_vector;
    uint row = vector_local >> 6u;
    uint coefficient = vector_local & 63u;
    AkitaWideAccumulator accumulator = akita_wide_zero();
    for (uint partial = thread_index;
         partial < (uint)params.column_partials;
         partial += 256u) {
        ulong partial_index =
            (((ulong)vector * params.num_rows + (ulong)row)
                * params.column_partials + (ulong)partial) * 64ul
            + (ulong)coefficient;
        partial_index += (ulong)product
            * params.num_vectors * params.num_rows * params.column_partials * 64ul;
        akita_wide_accumulate(accumulator, partials[partial_index], true);
    }
    reduction[thread_index] = akita_reduce_wide(accumulator);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride != 0u; stride >>= 1u) {
        if (thread_index < stride) {
            reduction[thread_index] = akita_add(
                reduction[thread_index], reduction[thread_index + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index == 0u) {
        output[output_index] = reduction[0];
    }
}

kernel void akita_fp128_i8_coefficient_packing(
    device const char *sources [[buffer(0)]],
    device const AkitaFp128 *combined_weights [[buffer(1)]],
    device AkitaFp128 *output [[buffer(2)]],
    constant I8CoefficientPackingParams &params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup AkitaFp128 reduction[256];
    ulong output_index = (ulong)threadgroup_index.x;
    ulong outputs_per_source = params.num_blocks * params.subring_dimension;
    ulong source = output_index / outputs_per_source;
    ulong source_local = output_index - source * outputs_per_source;
    ulong block = source_local / params.subring_dimension;
    ulong subring = source_local - block * params.subring_dimension;
    ulong first_position = block * params.positions_per_block;
    ulong term_count = params.positions_per_block * params.stride;
    AkitaFp128 accumulator = akita_zero();

    for (ulong term = (ulong)thread_index; term < term_count; term += 256ul) {
        ulong local_position = term / params.stride;
        ulong low = term - local_position * params.stride;
        ulong position = first_position + local_position;
        if (position >= params.num_live_positions) {
            continue;
        }
        ulong coefficient = subring * params.stride + low;
        ulong flat = position * params.ring_d + coefficient;
        if (flat >= params.live_coefficients) {
            continue;
        }
        int digit = (int)sources[source * params.source_coefficients + flat];
        uint magnitude = (uint)(digit < 0 ? -digit : digit);
        AkitaFp128 weight = combined_weights[term];
        for (uint repeat = 0u; repeat < magnitude; ++repeat) {
            accumulator = digit < 0
                ? akita_sub(accumulator, weight)
                : akita_add(accumulator, weight);
        }
    }
    reduction[thread_index] = accumulator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset != 0u; offset >>= 1u) {
        if (thread_index < offset) {
            reduction[thread_index] =
                akita_add(reduction[thread_index], reduction[thread_index + offset]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index == 0u && output_index < params.output_coefficients) {
        output[output_index] = reduction[0];
    }
}

inline AkitaFp128 akita_simd_shuffle_fp128(AkitaFp128 value, uint source_lane) {
    AkitaFp128 result;
    result.limb = uint4(
        simd_shuffle(value.limb[0], source_lane),
        simd_shuffle(value.limb[1], source_lane),
        simd_shuffle(value.limb[2], source_lane),
        simd_shuffle(value.limb[3], source_lane));
    return result;
}

inline int4 akita_simd_shuffle_int4(int4 value, uint source_lane) {
    return int4(
        simd_shuffle(value.x, source_lane),
        simd_shuffle(value.y, source_lane),
        simd_shuffle(value.z, source_lane),
        simd_shuffle(value.w, source_lane));
}

inline AkitaFp128 akita_simd_sum_fp128(AkitaFp128 value) {
    for (uint offset = 16u; offset != 0u; offset >>= 1u) {
        AkitaFp128 partner;
        partner.limb = uint4(
            simd_shuffle_xor(value.limb[0], offset),
            simd_shuffle_xor(value.limb[1], offset),
            simd_shuffle_xor(value.limb[2], offset),
            simd_shuffle_xor(value.limb[3], offset));
        value = akita_add(value, partner);
    }
    return value;
}

kernel void akita_fp128_packed_onehot_coefficient_packing_partials(
    device const uchar *lanes [[buffer(0)]],
    device const AkitaFp128 *combined_weights [[buffer(1)]],
    device AkitaFp128 *partials [[buffer(2)]],
    device const ulong *active_zero_rows [[buffer(3)]],
    constant PackedOneHotCoefficientPackingParams &params [[buffer(4)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup atomic_int bucket_digits[256 * 8];
    ulong group = (ulong)threadgroup_index.x;
    ulong block = group / params.row_partials_per_block;
    ulong row_partial = group - block * params.row_partials_per_block;
    ulong column = block / params.blocks_per_column;
    ulong block_in_column = block - column * params.blocks_per_column;
    ulong row_block_start = block_in_column * params.rows_per_block;
    ulong row_block_end = min(row_block_start + params.rows_per_block, params.num_rows);
    ulong row_start = row_block_start + row_partial * params.rows_per_partial;
    ulong row_end = min(row_start + params.rows_per_partial, row_block_end);
    ulong digit_count = params.subring_dimension * 8ul;
    for (ulong digit = (ulong)thread_index;
         digit < digit_count;
         digit += 256ul) {
        atomic_store_explicit(
            &bucket_digits[digit], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (column < params.num_columns) {
        for (ulong row = row_start + (ulong)thread_index;
             row < row_end;
             row += 256ul) {
            uint hot = (uint)lanes[row * params.num_columns + column];
            bool committed = hot != 0u;
            if (!committed && ((params.zero_column_mask >> column) & 1ul) != 0ul) {
                ulong active_word = active_zero_rows[row >> 6ul];
                committed = ((active_word >> (row & 63ul)) & 1ul) != 0ul;
            }
            if (!committed || (ulong)hot >= params.onehot_k) {
                continue;
            }
            ulong field_in_block =
                (row - row_block_start) * params.onehot_k + (ulong)hot;
            ulong position = field_in_block / params.ring_d;
            ulong coefficient = field_in_block - position * params.ring_d;
            ulong bucket = coefficient / params.stride;
            if (position >= params.positions_per_block
                || bucket >= params.subring_dimension) {
                continue;
            }
            ulong low = coefficient - bucket * params.stride;
            AkitaFp128 weight = combined_weights[position * params.stride + low];
            ulong bucket_base = bucket * 8ul;
            for (uint limb = 0u; limb < 4u; ++limb) {
                atomic_fetch_add_explicit(
                    &bucket_digits[bucket_base + (ulong)(2u * limb)],
                    (int)(weight.limb[limb] & 0xffffu),
                    memory_order_relaxed);
                atomic_fetch_add_explicit(
                    &bucket_digits[bucket_base + (ulong)(2u * limb + 1u)],
                    (int)(weight.limb[limb] >> 16u),
                    memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong partial_base =
        (block * params.row_partials_per_block + row_partial)
            * params.subring_dimension;
    if ((ulong)thread_index < params.subring_dimension) {
        AkitaWideAccumulator accumulator = akita_wide_zero();
        ulong bucket_base = (ulong)thread_index * 8ul;
        for (uint limb = 0u; limb < 4u; ++limb) {
            accumulator.low_digits[limb] = atomic_load_explicit(
                &bucket_digits[bucket_base + (ulong)(2u * limb)],
                memory_order_relaxed);
            accumulator.high_digits[limb] = atomic_load_explicit(
                &bucket_digits[bucket_base + (ulong)(2u * limb + 1u)],
                memory_order_relaxed);
        }
        partials[partial_base + (ulong)thread_index] =
            akita_reduce_wide(accumulator);
    }
}

kernel void akita_fp128_packed_onehot_coefficient_packing_reduce(
    device const AkitaFp128 *partials [[buffer(0)]],
    device AkitaFp128 *output [[buffer(1)]],
    constant PackedOneHotCoefficientPackingParams &params [[buffer(2)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    ulong block = (ulong)threadgroup_index.x;
    ulong bucket = (ulong)thread_index;
    if (bucket >= params.subring_dimension) {
        return;
    }
    AkitaFp128 accumulator = akita_zero();
    ulong partial_base = block * params.row_partials_per_block
        * params.subring_dimension + bucket;
    for (ulong row_partial = 0ul;
         row_partial < params.row_partials_per_block;
         ++row_partial) {
        accumulator = akita_add(
            accumulator,
            partials[partial_base + row_partial * params.subring_dimension]);
    }
    output[block * params.subring_dimension + bucket] = accumulator;
}

kernel void akita_fp128_d512_decompose_fold(
    device const uchar *lanes [[buffer(0)]],
    device const ushort *challenge_positions [[buffer(1)]],
    device const char *challenge_coefficients [[buffer(2)]],
    device int *output [[buffer(3)]],
    constant PackedDecomposeFoldParams &params [[buffer(4)]],
    device const ulong *active_zero_rows [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup atomic_int accumulators[512];

    uint local_position = threadgroup_index.x;
    ulong position = params.position_start + (ulong)local_position;
    atomic_store_explicit(
        &accumulators[thread_index], 0, memory_order_relaxed);
    atomic_store_explicit(
        &accumulators[thread_index + 256u], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong row_shift = params.onehot_k == 16ul ? 5ul : 1ul;
    ulong rows_per_ring = 1ul << row_shift;
    ulong active_rings = (params.zero_suffix_start + rows_per_ring - 1ul) >> row_shift;
    ulong active_blocks = active_rings > position
        ? (active_rings - position + params.num_positions - 1ul) / params.num_positions
        : 0ul;
    ulong tasks_per_position =
        active_blocks * params.num_columns * rows_per_ring;
    for (ulong task = (ulong)thread_index;
         task < tasks_per_position;
         task += 256ul) {
        ulong trace_block = task / (params.num_columns * rows_per_ring);
        ulong block_local = task % (params.num_columns * rows_per_ring);
        ulong column = block_local >> row_shift;
        ulong row_in_ring = block_local & (rows_per_ring - 1ul);
        ulong ring = trace_block * params.num_positions + position;
        ulong row = ring * rows_per_ring + row_in_ring;
        uchar hot = lanes[row * params.lane_stride + column];
        bool committed = hot != 0u;
        if (!committed && column < 64ul
            && ((params.zero_column_mask >> column) & 1ul) != 0ul) {
            ulong active_word = active_zero_rows[row >> 6ul];
            committed = ((active_word >> (row & 63ul)) & 1ul) != 0ul;
        }
        if (!committed) {
            continue;
        }

        uint source_coefficient = (uint)(row_in_ring * params.onehot_k) + (uint)hot;
        ulong challenge = column * params.blocks_per_column + trace_block;
        ulong challenge_start = challenge * params.challenge_weight;
        for (ulong term = 0ul; term < params.challenge_weight; ++term) {
            uint destination = source_coefficient
                + (uint)challenge_positions[challenge_start + term];
            int value = (int)challenge_coefficients[challenge_start + term];
            if (destination >= 512u) {
                destination -= 512u;
                value = -value;
            }
            atomic_fetch_add_explicit(
                &accumulators[destination], value, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong output_base = (ulong)local_position * 512ul;
    output[output_base + (ulong)thread_index] = atomic_load_explicit(
        &accumulators[thread_index], memory_order_relaxed);
    output[output_base + (ulong)thread_index + 256ul] = atomic_load_explicit(
        &accumulators[thread_index + 256u], memory_order_relaxed);
}

kernel void akita_fp128_d512_subring64_decompose_fold(
    device const uchar *lanes [[buffer(0)]],
    device const char *dense_challenges [[buffer(1)]],
    device int *output [[buffer(2)]],
    constant PackedDecomposeFoldParams &params [[buffer(3)]],
    device const ulong *active_zero_rows [[buffer(4)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup uint partitioned_tasks[8 * 8 * 32];
    threadgroup uint partition_counts[8 * 8];

    uint simd_lane = thread_index & 31u;
    uint simdgroup = thread_index >> 5u;
    uint local_position = threadgroup_index.x;
    ulong position = params.position_start + (ulong)local_position;
    ulong row_shift = params.onehot_k == 16ul ? 5ul : 1ul;
    ulong rows_per_ring = 1ul << row_shift;
    ulong active_rings = (params.zero_suffix_start + rows_per_ring - 1ul) >> row_shift;
    ulong active_blocks = active_rings > position
        ? (active_rings - position + params.num_positions - 1ul) / params.num_positions
        : 0ul;
    ulong tasks_per_position =
        active_blocks * params.num_columns * rows_per_ring;
    int accumulator_low = 0;
    int accumulator_high = 0;

    for (ulong task_base = 0ul;
         task_base < tasks_per_position;
         task_base += 256ul) {
        ulong task = task_base + (ulong)thread_index;
        bool valid = task < tasks_per_position;
        uint hot = 0u;
        uint source_high = 0u;
        uint challenge = 0u;
        if (valid) {
            ulong trace_block = task / (params.num_columns * rows_per_ring);
            ulong block_local = task % (params.num_columns * rows_per_ring);
            ulong column = block_local >> row_shift;
            ulong row_in_ring = block_local & (rows_per_ring - 1ul);
            ulong ring = trace_block * params.num_positions + position;
            ulong row = ring * rows_per_ring + row_in_ring;
            hot = (uint)lanes[row * params.lane_stride + column];
            bool committed = hot != 0u;
            if (!committed && column < 64ul
                && ((params.zero_column_mask >> column) & 1ul) != 0ul) {
                ulong active_word = active_zero_rows[row >> 6ul];
                committed = ((active_word >> (row & 63ul)) & 1ul) != 0ul;
            }
            valid = committed;
            source_high = (uint)(row_in_ring * (params.onehot_k / 8ul)) + (hot >> 3u);
            challenge = (uint)(column * params.blocks_per_column + trace_block);
        }

        for (uint low = 0u; low < 8u; ++low) {
            bool selected = valid && ((hot & 7u) == low);
            uint selected_mask = uint(
                simd_ballot(selected).operator unsigned long());
            uint partition = simdgroup * 8u + low;
            if (simd_lane == 0u) {
                partition_counts[partition] = popcount(selected_mask);
            }
            if (selected) {
                uint preceding_mask = simd_lane == 0u
                    ? 0u
                    : ((1u << simd_lane) - 1u);
                uint rank = popcount(selected_mask & preceding_mask);
                partitioned_tasks[partition * 32u + rank] =
                    (challenge << 6u) | source_high;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint destination_low = simdgroup;
        uint destination_high_0 = simd_lane;
        uint destination_high_1 = simd_lane + 32u;
        for (uint producer = 0u; producer < 8u; ++producer) {
            uint partition = producer * 8u + destination_low;
            uint count = partition_counts[partition];
            for (uint index = 0u; index < count; ++index) {
                uint packed = partitioned_tasks[partition * 32u + index];
                uint source = packed & 63u;
                uint challenge_index = packed >> 6u;
                uint challenge_position_0 =
                    (destination_high_0 + 64u - source) & 63u;
                uint challenge_position_1 =
                    (destination_high_1 + 64u - source) & 63u;
                int value_0 = (int)dense_challenges[
                    (ulong)challenge_index * 64ul + (ulong)challenge_position_0];
                int value_1 = (int)dense_challenges[
                    (ulong)challenge_index * 64ul + (ulong)challenge_position_1];
                accumulator_low += destination_high_0 < source ? -value_0 : value_0;
                accumulator_high += destination_high_1 < source ? -value_1 : value_1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    ulong output_base = (ulong)local_position * 512ul;
    output[output_base + (ulong)simdgroup
        + 8ul * (ulong)simd_lane] = accumulator_low;
    output[output_base + (ulong)simdgroup
        + 8ul * (ulong)(simd_lane + 32u)] = accumulator_high;
}

// Each SIMD group owns its tasks and both coefficient-parity classes. Groups
// meet only after the task loop, with no cross-group synchronization per tile.
kernel void akita_fp128_d128_subring64_decompose_fold(
    device const uchar *lanes [[buffer(0)]],
    device const char *dense_challenges [[buffer(1)]],
    device int *output [[buffer(2)]],
    constant PackedDecomposeFoldParams &params [[buffer(3)]],
    device const ulong *active_zero_rows [[buffer(4)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup int partial_sums[8 * 128];

    uint simd_lane = thread_index & 31u;
    uint simdgroup = thread_index >> 5u;
    uint local_position = threadgroup_index.x;
    ulong position = params.position_start + (ulong)local_position;
    int2 accumulators[2] = { int2(0), int2(0) };

    for (ulong trace_block = (ulong)simdgroup;
         trace_block < params.blocks_per_column;
         trace_block += 8ul) {
        ulong column = (ulong)simd_lane;
        bool valid = column < params.num_columns;
        uint hot = 0u;
        uint source_high = 0u;
        uint challenge = 0u;
        if (valid) {
            ulong ring = trace_block * params.num_positions + position;
            ulong row = ring >> 1ul;
            hot = (uint)lanes[row * params.lane_stride + column];
            bool committed = hot != 0u;
            if (!committed && column < 64ul
                && ((params.zero_column_mask >> column) & 1ul) != 0ul) {
                ulong active_word = active_zero_rows[row >> 6ul];
                committed = ((active_word >> (row & 63ul)) & 1ul) != 0ul;
            }
            valid = committed && (hot >> 7u) == (uint)(ring & 1ul);
            source_high = (hot & 127u) >> 1u;
            challenge = (uint)(column * params.blocks_per_column + trace_block);
        }

        uint task_data = (challenge << 6u) | source_high;
        for (uint low = 0u; low < 2u; ++low) {
            bool selected = valid && ((hot & 1u) == low);
            uint selected_mask = uint(simd_ballot(selected).operator unsigned long());
            while (selected_mask != 0u) {
                uint selected_lane = ctz(selected_mask);
                uint packed = simd_shuffle(task_data, selected_lane);
                uint source = packed & 63u;
                uint challenge_index = packed >> 6u;
                uint destination_0 = simd_lane;
                uint destination_1 = simd_lane + 32u;
                int value_0 = (int)dense_challenges[
                    (ulong)challenge_index * 64ul + ((destination_0 + 64u - source) & 63u)];
                int value_1 = (int)dense_challenges[
                    (ulong)challenge_index * 64ul + ((destination_1 + 64u - source) & 63u)];
                accumulators[low] += int2(
                    destination_0 < source ? -value_0 : value_0,
                    destination_1 < source ? -value_1 : value_1);
                selected_mask &= selected_mask - 1u;
            }
        }
    }

    uint partial_base = simdgroup * 128u;
    for (uint low = 0u; low < 2u; ++low) {
        partial_sums[partial_base + low + 2u * simd_lane] = accumulators[low].x;
        partial_sums[partial_base + low + 2u * (simd_lane + 32u)] = accumulators[low].y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < 128u) {
        int sum = 0;
        for (uint partition = 0u; partition < 8u; ++partition) {
            sum += partial_sums[partition * 128u + thread_index];
        }
        output[(ulong)local_position * 128ul + thread_index] = sum;
    }
}

kernel void akita_fp128_d512_build_fold_index(
    device const uchar *lanes [[buffer(0)]],
    device uint *records [[buffer(1)]],
    device ushort *counts [[buffer(2)]],
    constant PackedFoldIndexParams &params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup ushort partition_counts[8 * 8];

    uint simd_lane = thread_index & 31u;
    uint simdgroup = thread_index >> 5u;
    ulong position = (ulong)threadgroup_index.x;
    for (ulong tile = 0ul; tile < params.tiles_per_position; ++tile) {
        ulong task = tile * 256ul + (ulong)thread_index;
        bool valid = task < params.tasks_per_position;
        uint hot = 0u;
        uint source_high = 0u;
        uint challenge = 0u;
        if (valid) {
            ulong trace_block = task / (params.num_columns * 2ul);
            ulong block_local = task % (params.num_columns * 2ul);
            ulong column = block_local >> 1ul;
            ulong row_in_ring = block_local & 1ul;
            ulong ring = trace_block * params.num_positions + position;
            ulong row = ring * 2ul + row_in_ring;
            hot = (uint)lanes[row * params.lane_stride + column];
            valid = hot != 0u;
            source_high = (uint)(row_in_ring * 32ul) + (hot >> 3u);
            challenge = (uint)(column * params.blocks_per_column + trace_block);
        }

        uint selected_low = hot & 7u;
        uint selected_rank = 0u;
        for (uint low = 0u; low < 8u; ++low) {
            bool selected = valid && selected_low == low;
            uint selected_mask = uint(
                simd_ballot(selected).operator unsigned long());
            uint partition = simdgroup * 8u + low;
            if (simd_lane == 0u) {
                partition_counts[partition] = (ushort)popcount(selected_mask);
            }
            if (selected) {
                uint preceding_mask = simd_lane == 0u
                    ? 0u
                    : ((1u << simd_lane) - 1u);
                selected_rank = popcount(selected_mask & preceding_mask);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        ulong tile_index = position * params.tiles_per_position + tile;
        if (valid) {
            uint output_index = 0u;
            for (uint low = 0u; low < selected_low; ++low) {
                for (uint producer = 0u; producer < 8u; ++producer) {
                    output_index += (uint)partition_counts[producer * 8u + low];
                }
            }
            for (uint producer = 0u; producer < simdgroup; ++producer) {
                output_index +=
                    (uint)partition_counts[producer * 8u + selected_low];
            }
            output_index += selected_rank;
            records[tile_index * 256ul + (ulong)output_index] =
                (challenge << 6u) | source_high;
        }
        if (thread_index < 8u) {
            uint count = 0u;
            for (uint producer = 0u; producer < 8u; ++producer) {
                count += (uint)partition_counts[producer * 8u + thread_index];
            }
            counts[tile_index * 8ul + (ulong)thread_index] = (ushort)count;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void akita_fp128_d512_build_coefficient_packing_index(
    device const uchar *lanes [[buffer(0)]],
    device ushort *records [[buffer(1)]],
    device ushort *offsets [[buffer(2)]],
    constant PackedCoefficientPackingIndexParams &params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup atomic_uint bucket_cursors[64 * 32];

    ulong group = (ulong)threadgroup_index.x;
    ulong trace_block = group / params.position_tiles;
    ulong tile = group - trace_block * params.position_tiles;
    uint stream_count = (uint)(params.num_columns * 2ul);
    uint cursor_count = stream_count * 32u;
    for (uint cursor = thread_index; cursor < cursor_count; cursor += 256u) {
        atomic_store_explicit(&bucket_cursors[cursor], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong position = tile * 256ul + (ulong)thread_index;
    bool valid_position = trace_block < params.blocks_per_column
        && position < params.num_positions;
    ulong ring = trace_block * params.num_positions + position;
    ulong row_0 = ring * 2ul;
    ulong row_1 = row_0 + 1ul;
    for (uint column = 0u; column < (uint)params.num_columns; ++column) {
        uint stream = column * 2u;
        if (valid_position && row_0 < params.num_rows) {
            uint hot = (uint)lanes[row_0 * params.lane_stride + (ulong)column];
            if (hot != 0u) {
                atomic_fetch_add_explicit(
                    &bucket_cursors[stream * 32u + (hot >> 3u)],
                    1u,
                    memory_order_relaxed);
            }
        }
        if (valid_position && row_1 < params.num_rows) {
            uint hot = (uint)lanes[row_1 * params.lane_stride + (ulong)column];
            if (hot != 0u) {
                atomic_fetch_add_explicit(
                    &bucket_cursors[(stream + 1u) * 32u + (hot >> 3u)],
                    1u,
                    memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (thread_index < stream_count) {
        uint stream = thread_index;
        ulong global_stream =
            (trace_block * params.num_columns * 2ul) + (ulong)stream;
        ulong layout = global_stream * params.position_tiles + tile;
        uint running = 0u;
        for (uint bucket = 0u; bucket < 32u; ++bucket) {
            uint cursor = stream * 32u + bucket;
            uint count = atomic_load_explicit(
                &bucket_cursors[cursor], memory_order_relaxed);
            offsets[layout * 33ul + (ulong)bucket] = (ushort)running;
            atomic_store_explicit(
                &bucket_cursors[cursor], running, memory_order_relaxed);
            running += count;
        }
        offsets[layout * 33ul + 32ul] = (ushort)running;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint column = 0u; column < (uint)params.num_columns; ++column) {
        uint stream = column * 2u;
        if (valid_position && row_0 < params.num_rows) {
            uint hot = (uint)lanes[row_0 * params.lane_stride + (ulong)column];
            if (hot != 0u) {
                uint bucket = hot >> 3u;
                uint destination = atomic_fetch_add_explicit(
                    &bucket_cursors[stream * 32u + bucket],
                    1u,
                    memory_order_relaxed);
                ulong global_stream =
                    trace_block * params.num_columns * 2ul + (ulong)stream;
                ulong layout = global_stream * params.position_tiles + tile;
                records[layout * 256ul + (ulong)destination] =
                    (ushort)(((hot & 7u) << 8u) | thread_index);
            }
        }
        if (valid_position && row_1 < params.num_rows) {
            uint hot = (uint)lanes[row_1 * params.lane_stride + (ulong)column];
            if (hot != 0u) {
                uint bucket = hot >> 3u;
                uint odd_stream = stream + 1u;
                uint destination = atomic_fetch_add_explicit(
                    &bucket_cursors[odd_stream * 32u + bucket],
                    1u,
                    memory_order_relaxed);
                ulong global_stream =
                    trace_block * params.num_columns * 2ul + (ulong)odd_stream;
                ulong layout = global_stream * params.position_tiles + tile;
                records[layout * 256ul + (ulong)destination] =
                    (ushort)(((hot & 7u) << 8u) | thread_index);
            }
        }
    }
}

inline AkitaFp128 akita_reduce_unsigned_limb_sums(ulong4 sums) {
    AkitaFp128 base;
    ulong word = sums[0];
    base.limb[0] = (uint)word;
    ulong carry = word >> 32u;
    word = sums[1] + carry;
    base.limb[1] = (uint)word;
    carry = word >> 32u;
    word = sums[2] + carry;
    base.limb[2] = (uint)word;
    carry = word >> 32u;
    word = sums[3] + carry;
    base.limb[3] = (uint)word;
    ulong high = word >> 32u;

    AkitaCorrection canonical = akita_add_offset(base);
    base = akita_select(canonical.carry != 0u, canonical.value, base);
    if (high == 0ul) {
        return base;
    }
    ulong correction_word = high * (ulong)AKITA_OFFSET;
    AkitaFp128 correction = akita_zero();
    correction.limb[0] = (uint)correction_word;
    correction.limb[1] = (uint)(correction_word >> 32u);
    return akita_add(base, correction);
}

kernel void akita_fp128_d512_indexed_coefficient_packing_partials(
    device const ushort *records [[buffer(0)]],
    device const ushort *offsets [[buffer(1)]],
    device const AkitaFp128 *combined_weights [[buffer(2)]],
    device AkitaFp128 *partials [[buffer(3)]],
    constant PackedCoefficientPackingIndexParams &index_params [[buffer(4)]],
    constant PackedOneHotCoefficientPackingParams &params [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup ulong4 reduction[256];
    ulong live_streams = params.blocks_per_column * params.num_columns * 2ul;
    ulong group = (ulong)threadgroup_index.x;
    ulong tile_chunk = group / live_streams;
    ulong stream = group - tile_chunk * live_streams;
    ulong streams_per_block = params.num_columns * 2ul;
    ulong trace_block = stream / streams_per_block;
    ulong stream_local = stream - trace_block * streams_per_block;
    ulong column = stream_local >> 1ul;
    ulong parity = stream_local & 1ul;
    uint simd_lane = thread_index & 31u;
    uint bucket_lane = simd_lane & 7u;
    uint bucket_owner = simd_lane - bucket_lane;
    uint bucket = thread_index >> 3u;
    ulong tile_chunks =
        (index_params.position_tiles + (ulong)D512_PACKING_TILES_PER_CHUNK - 1ul)
            / (ulong)D512_PACKING_TILES_PER_CHUNK;
    ulong4 sums = ulong4(0ul);
    ulong tile_start = tile_chunk * (ulong)D512_PACKING_TILES_PER_CHUNK;
    ulong tile_end = min(
        tile_start + (ulong)D512_PACKING_TILES_PER_CHUNK,
        index_params.position_tiles);
    for (ulong tile = tile_start; tile < tile_end; ++tile) {
        ulong layout = stream * index_params.position_tiles + tile;
        uint start = 0u;
        uint end = 0u;
        if (bucket_lane == 0u) {
            start = (uint)offsets[layout * 33ul + (ulong)bucket];
            end = (uint)offsets[layout * 33ul + (ulong)bucket + 1ul];
        }
        start = simd_shuffle(start, bucket_owner);
        end = simd_shuffle(end, bucket_owner);
        ulong record_base = layout * 256ul;
        for (uint record_index = start + bucket_lane;
             record_index < end;
             record_index += 8u) {
            uint record = (uint)records[record_base + (ulong)record_index];
            ulong position = tile * 256ul + (ulong)(record & 255u);
            ulong low = (ulong)(record >> 8u);
            AkitaFp128 weight = combined_weights[position * 8ul + low];
            sums += ulong4(
                (ulong)weight.limb[0],
                (ulong)weight.limb[1],
                (ulong)weight.limb[2],
                (ulong)weight.limb[3]);
        }
    }
    reduction[thread_index] = sums;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (bucket_lane < 4u) {
        reduction[thread_index] += reduction[thread_index + 4u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (bucket_lane < 2u) {
        reduction[thread_index] += reduction[thread_index + 2u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (bucket_lane == 0u) {
        ulong4 total = reduction[thread_index] + reduction[thread_index + 1u];
        ulong partial_index = (stream * 32ul + (ulong)bucket) * tile_chunks + tile_chunk;
        partials[partial_index] = akita_reduce_unsigned_limb_sums(total);
    }
}

kernel void akita_fp128_d512_indexed_coefficient_packing_reduce(
    device const AkitaFp128 *partials [[buffer(0)]],
    device AkitaFp128 *output [[buffer(1)]],
    constant PackedCoefficientPackingIndexParams &index_params [[buffer(2)]],
    constant PackedOneHotCoefficientPackingParams &params [[buffer(3)]],
    uint3 thread_position [[thread_position_in_grid]])
{
    ulong output_index = (ulong)thread_position.x;
    if (output_index >= params.output_coefficients) {
        return;
    }
    ulong tile_chunks =
        (index_params.position_tiles + (ulong)D512_PACKING_TILES_PER_CHUNK - 1ul)
            / (ulong)D512_PACKING_TILES_PER_CHUNK;
    ulong coefficient = output_index % params.subring_dimension;
    ulong output_block = output_index / params.subring_dimension;
    ulong trace_block = output_block % params.blocks_per_column;
    ulong column = output_block / params.blocks_per_column;
    if (column >= params.num_columns) {
        output[output_index] = akita_zero();
        return;
    }
    ulong parity = coefficient >> 5ul;
    ulong bucket = coefficient & 31ul;
    ulong stream = (trace_block * params.num_columns + column) * 2ul + parity;
    ulong partial_base = (stream * 32ul + bucket) * tile_chunks;
    AkitaFp128 sum = akita_zero();
    for (ulong chunk = 0ul; chunk < tile_chunks; ++chunk) {
        sum = akita_add(sum, partials[partial_base + chunk]);
    }
    output[output_index] = sum;
}

inline uint akita_lower_byte_mask(uint count) {
    if (count == 0u) {
        return 0u;
    }
    if (count >= 4u) {
        return 0xffffffffu;
    }
    return (1u << (8u * count)) - 1u;
}

inline uint akita_apply_negacyclic_signs(uint packed, uint negative_count) {
    uint mask = akita_lower_byte_mask(negative_count);
    uint negated = 0x04040404u - packed;
    return (packed & ~mask) | (negated & mask);
}

inline void akita_store_indexed_fold_value(
    device int *output,
    device char *digits,
    ulong output_base,
    ulong digit_position_base,
    uint coefficient,
    int value,
    ulong fold_digits,
    uint fold_log_basis)
{
    output[output_base + (ulong)coefficient] = value;
    int basis = 1 << fold_log_basis;
    int half_basis = basis >> 1;
    int mask = basis - 1;
    for (ulong digit = 0ul; digit < fold_digits; ++digit) {
        int raw = value & mask;
        int balanced = raw >= half_basis ? raw - basis : raw;
        value = (value - balanced) >> fold_log_basis;
        digits[digit_position_base + digit * 512ul + (ulong)coefficient] =
            (char)balanced;
    }
}

kernel void akita_fp128_d512_fused_subring64_decompose_fold(
    device const uchar *lanes [[buffer(0)]],
    device const uint *packed_challenges [[buffer(1)]],
    device int *output [[buffer(2)]],
    device char *digits [[buffer(3)]],
    constant PackedFoldIndexParams &params [[buffer(4)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup uint partitioned_tasks[8 * 8 * 32];
    threadgroup ushort partition_counts[8 * 8];

    uint simd_lane = thread_index & 31u;
    uint simdgroup = thread_index >> 5u;
    uint residue = simdgroup;
    uint source_group = simd_lane >> 3u;
    uint destination_owner = simd_lane & 7u;
    int4 accumulator_even = int4(0);
    int4 accumulator_odd = int4(0);
    ulong local_position = (ulong)threadgroup_index.x;
    ulong position = params.position_start + local_position;

    for (ulong tile = 0ul; tile < params.tiles_per_position; ++tile) {
        ulong task = tile * 256ul + (ulong)thread_index;
        bool valid = task < params.tasks_per_position;
        uint hot = 0u;
        uint source_high = 0u;
        uint challenge = 0u;
        if (valid) {
            ulong trace_block = task / (params.num_columns * 2ul);
            ulong block_local = task % (params.num_columns * 2ul);
            ulong column = block_local >> 1ul;
            ulong row_in_ring = block_local & 1ul;
            ulong ring = trace_block * params.num_positions + position;
            ulong row = ring * 2ul + row_in_ring;
            hot = (uint)lanes[row * params.lane_stride + column];
            valid = hot != 0u;
            source_high = (uint)(row_in_ring * 32ul) + (hot >> 3u);
            challenge = (uint)(column * params.blocks_per_column + trace_block);
        }

        for (uint low = 0u; low < 8u; ++low) {
            bool selected = valid && ((hot & 7u) == low);
            uint selected_mask = uint(
                simd_ballot(selected).operator unsigned long());
            uint partition = simdgroup * 8u + low;
            if (simd_lane == 0u) {
                partition_counts[partition] = (ushort)popcount(selected_mask);
            }
            if (selected) {
                uint preceding_mask = simd_lane == 0u
                    ? 0u
                    : ((1u << simd_lane) - 1u);
                uint rank = popcount(selected_mask & preceding_mask);
                partitioned_tasks[partition * 32u + rank] =
                    (challenge << 6u) | source_high;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint producer = 0u; producer < 8u; ++producer) {
            uint partition = producer * 8u + residue;
            uint count = (uint)partition_counts[partition];
            uint group_count = count > source_group
                ? (count + 3u - source_group) / 4u
                : 0u;
            uint packed_even = 0u;
            uint packed_odd = 0u;
            uint iterations = (count + 3u) / 4u;
            for (uint iteration = 0u; iteration < iterations; ++iteration) {
                uint relative_index = 4u * iteration + source_group;
                bool record_valid = relative_index < count;
                uint record = 0u;
                uint source_lane = 8u * source_group;
                if (simd_lane == source_lane && record_valid) {
                    record = partitioned_tasks[
                        partition * 32u + relative_index];
                }
                record = simd_shuffle(record, source_lane);
                if (record_valid) {
                    uint source = record & 63u;
                    uint challenge_index = record >> 6u;
                    uint source_phase = source & 7u;
                    uint source_rotation = source >> 3u;
                    uint challenge_quad =
                        (destination_owner + 8u - source_rotation) & 7u;
                    uint word = packed_challenges[
                        (ulong)challenge_index * 64ul
                            + (ulong)source_phase * 8ul
                            + (ulong)challenge_quad];
                    uint even = word & 0x0f0f0f0fu;
                    uint odd = (word >> 4u) & 0x0f0f0f0fu;
                    uint destination_base = 8u * destination_owner;
                    uint negative_count = source > destination_base
                        ? min(source - destination_base, 8u)
                        : 0u;
                    packed_even += akita_apply_negacyclic_signs(
                        even, (negative_count + 1u) >> 1u);
                    packed_odd += akita_apply_negacyclic_signs(
                        odd, negative_count >> 1u);
                }
            }
            int bias = (int)(2u * group_count);
            accumulator_even += int4(
                (int)(packed_even & 255u) - bias,
                (int)((packed_even >> 8u) & 255u) - bias,
                (int)((packed_even >> 16u) & 255u) - bias,
                (int)((packed_even >> 24u) & 255u) - bias);
            accumulator_odd += int4(
                (int)(packed_odd & 255u) - bias,
                (int)((packed_odd >> 8u) & 255u) - bias,
                (int)((packed_odd >> 16u) & 255u) - bias,
                (int)((packed_odd >> 24u) & 255u) - bias);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    int4 total_even =
        akita_simd_shuffle_int4(accumulator_even, destination_owner)
        + akita_simd_shuffle_int4(accumulator_even, destination_owner + 8u)
        + akita_simd_shuffle_int4(accumulator_even, destination_owner + 16u)
        + akita_simd_shuffle_int4(accumulator_even, destination_owner + 24u);
    int4 total_odd =
        akita_simd_shuffle_int4(accumulator_odd, destination_owner)
        + akita_simd_shuffle_int4(accumulator_odd, destination_owner + 8u)
        + akita_simd_shuffle_int4(accumulator_odd, destination_owner + 16u)
        + akita_simd_shuffle_int4(accumulator_odd, destination_owner + 24u);
    if (source_group != 0u) {
        return;
    }

    ulong output_base = local_position * 512ul;
    ulong digit_position_base = local_position * params.fold_digits * 512ul;
    for (uint component = 0u; component < 4u; ++component) {
        uint destination_high = 8u * destination_owner + 2u * component;
        uint coefficient = residue + 8u * destination_high;
        akita_store_indexed_fold_value(
            output, digits, output_base, digit_position_base, coefficient,
            total_even[component], params.fold_digits,
            (uint)params.fold_log_basis);
        destination_high += 1u;
        coefficient = residue + 8u * destination_high;
        akita_store_indexed_fold_value(
            output, digits, output_base, digit_position_base, coefficient,
            total_odd[component], params.fold_digits,
            (uint)params.fold_log_basis);
    }
}

kernel void akita_fp128_d512_indexed_subring64_decompose_fold(
    device const uint *records [[buffer(0)]],
    device const ushort *counts [[buffer(1)]],
    device const uint *packed_challenges [[buffer(2)]],
    device int *output [[buffer(3)]],
    device char *digits [[buffer(4)]],
    constant PackedFoldIndexParams &params [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    uint simd_lane = thread_index & 31u;
    uint residue = thread_index >> 5u;
    uint source_group = simd_lane >> 3u;
    uint destination_owner = simd_lane & 7u;
    int4 accumulator_even = int4(0);
    int4 accumulator_odd = int4(0);
    ulong local_position = (ulong)threadgroup_index.x;
    ulong position = params.position_start + local_position;

    for (ulong tile = 0ul; tile < params.tiles_per_position; ++tile) {
        ulong tile_index = position * params.tiles_per_position + tile;
        ulong count_base = tile_index * 8ul;
        uint record_start = 0u;
        uint count = 0u;
        if (simd_lane == 0u) {
            for (uint low = 0u; low < residue; ++low) {
                record_start += (uint)counts[count_base + (ulong)low];
            }
            count = (uint)counts[count_base + (ulong)residue];
        }
        record_start = simd_shuffle(record_start, 0u);
        count = simd_shuffle(count, 0u);
        ulong record_base = tile_index * 256ul + (ulong)record_start;
        for (uint batch_start = 0u; batch_start < count; batch_start += 252u) {
            uint batch_count = min(252u, count - batch_start);
            uint group_count = batch_count > source_group
                ? (batch_count + 3u - source_group) / 4u
                : 0u;
            uint packed_even = 0u;
            uint packed_odd = 0u;
            uint iterations = (batch_count + 3u) / 4u;
            for (uint iteration = 0u; iteration < iterations; ++iteration) {
                uint relative_index = 4u * iteration + source_group;
                bool valid = relative_index < batch_count;
                uint record = 0u;
                uint source_lane = 8u * source_group;
                if (simd_lane == source_lane && valid) {
                    record = records[
                        record_base + (ulong)(batch_start + relative_index)];
                }
                record = simd_shuffle(record, source_lane);
                if (valid) {
                    uint source = record & 63u;
                    uint challenge = record >> 6u;
                    uint source_phase = source & 7u;
                    uint source_rotation = source >> 3u;
                    uint challenge_quad =
                        (destination_owner + 8u - source_rotation) & 7u;
                    uint word = packed_challenges[
                        (ulong)challenge * 64ul
                            + (ulong)source_phase * 8ul
                            + (ulong)challenge_quad];
                    uint even = word & 0x0f0f0f0fu;
                    uint odd = (word >> 4u) & 0x0f0f0f0fu;
                    uint destination_base = 8u * destination_owner;
                    uint negative_count = source > destination_base
                        ? min(source - destination_base, 8u)
                        : 0u;
                    packed_even += akita_apply_negacyclic_signs(
                        even, (negative_count + 1u) >> 1u);
                    packed_odd += akita_apply_negacyclic_signs(
                        odd, negative_count >> 1u);
                }
            }
            int bias = (int)(2u * group_count);
            accumulator_even += int4(
                (int)(packed_even & 255u) - bias,
                (int)((packed_even >> 8u) & 255u) - bias,
                (int)((packed_even >> 16u) & 255u) - bias,
                (int)((packed_even >> 24u) & 255u) - bias);
            accumulator_odd += int4(
                (int)(packed_odd & 255u) - bias,
                (int)((packed_odd >> 8u) & 255u) - bias,
                (int)((packed_odd >> 16u) & 255u) - bias,
                (int)((packed_odd >> 24u) & 255u) - bias);
        }
    }

    int4 total_even =
        akita_simd_shuffle_int4(accumulator_even, destination_owner)
        + akita_simd_shuffle_int4(accumulator_even, destination_owner + 8u)
        + akita_simd_shuffle_int4(accumulator_even, destination_owner + 16u)
        + akita_simd_shuffle_int4(accumulator_even, destination_owner + 24u);
    int4 total_odd =
        akita_simd_shuffle_int4(accumulator_odd, destination_owner)
        + akita_simd_shuffle_int4(accumulator_odd, destination_owner + 8u)
        + akita_simd_shuffle_int4(accumulator_odd, destination_owner + 16u)
        + akita_simd_shuffle_int4(accumulator_odd, destination_owner + 24u);
    if (source_group != 0u) {
        return;
    }

    ulong output_base = local_position * 512ul;
    ulong digit_position_base = local_position * params.fold_digits * 512ul;
    for (uint component = 0u; component < 4u; ++component) {
        uint destination_high = 8u * destination_owner + 2u * component;
        uint coefficient = residue + 8u * destination_high;
        akita_store_indexed_fold_value(
            output, digits, output_base, digit_position_base, coefficient,
            total_even[component], params.fold_digits,
            (uint)params.fold_log_basis);
        destination_high += 1u;
        coefficient = residue + 8u * destination_high;
        akita_store_indexed_fold_value(
            output, digits, output_base, digit_position_base, coefficient,
            total_odd[component], params.fold_digits,
            (uint)params.fold_log_basis);
    }
}

kernel void akita_fp128_d512_linear_relation_partials(
    device const AkitaFp128 *matrix [[buffer(0)]],
    device const int *rhs [[buffer(1)]],
    device int *partials [[buffer(2)]],
    device const D512LinearNttPrime *primes [[buffer(3)]],
    device const int *limb_weights [[buffer(4)]],
    device const int *field_moduli [[buffer(5)]],
    device const int *fwd_twiddles [[buffer(6)]],
    constant D512LinearRelationParams &params [[buffer(7)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup int matrix_values[D512_LINEAR_NTT_SIZE];
    threadgroup int rhs_values[D512_LINEAR_NTT_SIZE];

    ulong group = (ulong)threadgroup_index.x;
    uint prime_index = (uint)(group % params.num_primes);
    ulong tile = group / params.num_primes;
    ulong column_start = tile * params.columns_per_tile;
    ulong column_end = min(column_start + params.columns_per_tile, params.num_columns);
    D512LinearNttPrime prime = primes[prime_index];
    ulong twiddle_base = (ulong)prime_index * params.ntt_size;
    int accumulator_low = 0;
    int accumulator_high = 0;

    for (ulong column = column_start; column < column_end; ++column) {
        ulong coefficient = (ulong)thread_index;
        ulong source_index = column * D512_LINEAR_NTT_HALF + coefficient;
        matrix_values[thread_index] = d512_fp128_to_mont(
            matrix[source_index],
            prime_index,
            prime,
            limb_weights,
            field_moduli);
        matrix_values[thread_index + D512_LINEAR_NTT_HALF] = 0;
        rhs_values[thread_index] = d512_i32_to_mont(rhs[source_index], prime);
        rhs_values[thread_index + D512_LINEAR_NTT_HALF] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint len = D512_LINEAR_NTT_HALF; len != 0u; len >>= 1u) {
            uint butterfly = thread_index;
            uint block = butterfly / len;
            uint offset = butterfly - block * len;
            uint left = block * (len << 1u) + offset;
            uint right = left + len;
            int twiddle = fwd_twiddles[twiddle_base + (ulong)(len - 1u + offset)];

            int matrix_left = matrix_values[left];
            int matrix_right = matrix_values[right];
            matrix_values[left] = d512_ntt_add(matrix_left, matrix_right, prime.p);
            matrix_values[right] = d512_ntt_mul(
                d512_ntt_sub(matrix_left, matrix_right, prime.p),
                twiddle,
                prime);
            int rhs_left = rhs_values[left];
            int rhs_right = rhs_values[right];
            rhs_values[left] = d512_ntt_add(rhs_left, rhs_right, prime.p);
            rhs_values[right] = d512_ntt_mul(
                d512_ntt_sub(rhs_left, rhs_right, prime.p),
                twiddle,
                prime);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        accumulator_low = d512_ntt_add(
            accumulator_low,
            d512_ntt_mul(
                matrix_values[thread_index],
                rhs_values[thread_index],
                prime),
            prime.p);
        accumulator_high = d512_ntt_add(
            accumulator_high,
            d512_ntt_mul(
                matrix_values[thread_index + D512_LINEAR_NTT_HALF],
                rhs_values[thread_index + D512_LINEAR_NTT_HALF],
                prime),
            prime.p);
    }

    ulong partial_base = group * params.ntt_size;
    partials[partial_base + (ulong)thread_index] = accumulator_low;
    partials[partial_base + (ulong)thread_index + D512_LINEAR_NTT_HALF] = accumulator_high;
}


kernel void akita_fp128_d512_linear_relation_reduce(
    device const int *partials [[buffer(0)]],
    device uint *residues [[buffer(1)]],
    device const D512LinearNttPrime *primes [[buffer(2)]],
    device const int *inv_twiddles [[buffer(3)]],
    device const int *d_inv [[buffer(4)]],
    constant D512LinearRelationParams &params [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup int values[D512_LINEAR_NTT_SIZE];

    uint prime_index = threadgroup_index.x;
    D512LinearNttPrime prime = primes[prime_index];
    int low = 0;
    int high = 0;
    for (ulong tile = 0ul; tile < params.num_tiles; ++tile) {
        ulong base = (tile * params.num_primes + (ulong)prime_index) * params.ntt_size;
        low = d512_ntt_add(low, partials[base + (ulong)thread_index], prime.p);
        high = d512_ntt_add(
            high,
            partials[base + (ulong)thread_index + D512_LINEAR_NTT_HALF],
            prime.p);
    }
    values[thread_index] = low;
    values[thread_index + D512_LINEAR_NTT_HALF] = high;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong twiddle_base = (ulong)prime_index * params.ntt_size;
    for (uint len = 1u; len < D512_LINEAR_NTT_SIZE; len <<= 1u) {
        uint butterfly = thread_index;
        uint block = butterfly / len;
        uint offset = butterfly - block * len;
        uint left = block * (len << 1u) + offset;
        uint right = left + len;
        int twiddle = inv_twiddles[twiddle_base + (ulong)(len - 1u + offset)];
        int lhs = values[left];
        int rhs_value = d512_ntt_mul(values[right], twiddle, prime);
        values[left] = d512_ntt_add(lhs, rhs_value, prime.p);
        values[right] = d512_ntt_sub(lhs, rhs_value, prime.p);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    int scale = d_inv[prime_index];
    int scaled_low = d512_ntt_mul(values[thread_index], scale, prime);
    int scaled_high = d512_ntt_mul(
        values[thread_index + D512_LINEAR_NTT_HALF], scale, prime);
    int canonical_low = d512_ntt_mul_raw(scaled_low, 1, prime);
    int canonical_high = d512_ntt_mul_raw(scaled_high, 1, prime);
    canonical_low = d512_ntt_reduce((long)canonical_low, prime.p);
    canonical_high = d512_ntt_reduce((long)canonical_high, prime.p);
    ulong residue_base = (ulong)prime_index * params.ntt_size;
    residues[residue_base + (ulong)thread_index] = (uint)canonical_low;
    residues[residue_base + (ulong)thread_index + D512_LINEAR_NTT_HALF] =
        (uint)canonical_high;
}
kernel void akita_fp128_d512_linear_relation_reconstruct(
    device const uint *residues [[buffer(0)]],
    device AkitaFp128 *output [[buffer(1)]],
    device const D512LinearNttPrime *primes [[buffer(2)]],
    device const uint *garner_gamma [[buffer(3)]],
    device const AkitaFp128 *field_partial_products [[buffer(4)]],
    constant D512LinearRelationParams &params [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    long digits[D512_LINEAR_NTT_PRIMES];
    ulong coefficient = D512_LINEAR_NTT_HALF + (ulong)thread_index;
    for (uint prime_index = 0u; prime_index < D512_LINEAR_NTT_PRIMES; ++prime_index) {
        long modulus = (long)primes[prime_index].p;
        long digit = (long)residues[(ulong)prime_index * params.ntt_size + coefficient];
        for (uint prior = 0u; prior < prime_index; ++prior) {
            digit = d512_positive_mod(digit - digits[prior], modulus);
            digit = (digit * (long)garner_gamma[prime_index * D512_LINEAR_NTT_PRIMES + prior])
                % modulus;
        }
        digits[prime_index] = digit > modulus / 2l ? digit - modulus : digit;
    }

    AkitaFp128 reconstructed = akita_zero();
    for (uint prime_index = 0u; prime_index < D512_LINEAR_NTT_PRIMES; ++prime_index) {
        reconstructed = akita_add(
            reconstructed,
            akita_mul_signed_small(field_partial_products[prime_index], digits[prime_index]));
    }
    output[thread_index] = reconstructed;
}

kernel void akita_fp128_recursive_commit_matrix_ntt(
    device const AkitaFp128 *matrix [[buffer(0)]],
    device int *matrix_ntt [[buffer(1)]],
    device const D512LinearNttPrime *primes [[buffer(2)]],
    device const int *limb_weights [[buffer(3)]],
    device const int *field_moduli [[buffer(4)]],
    device const int *fwd_twiddles [[buffer(5)]],
    device const int *psi_pows [[buffer(6)]],
    constant RecursiveCommitParams &params [[buffer(7)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup int values[RECURSIVE_COMMIT_MAX_D];

    ulong group = (ulong)threadgroup_index.x;
    uint prime_index = (uint)(group % params.num_primes);
    ulong ring = group / params.num_primes;
    uint ring_d = (uint)params.ring_d;
    D512LinearNttPrime prime = primes[prime_index];
    ulong table_base = (ulong)prime_index * params.ring_d;
    int value = d512_fp128_to_mont(
        matrix[ring * params.ring_d + (ulong)thread_index],
        prime_index,
        prime,
        limb_weights,
        field_moduli);
    values[thread_index] = d512_ntt_mul(
        value,
        psi_pows[table_base + (ulong)thread_index],
        prime);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint len = ring_d >> 1u; len != 0u; len >>= 1u) {
        if (thread_index < ring_d >> 1u) {
            uint block = thread_index / len;
            uint offset = thread_index - block * len;
            uint left = block * (len << 1u) + offset;
            uint right = left + len;
            int lhs = values[left];
            int rhs = values[right];
            int twiddle = fwd_twiddles[table_base + (ulong)(len - 1u + offset)];
            values[left] = d512_ntt_add(lhs, rhs, prime.p);
            values[right] = d512_ntt_mul(
                d512_ntt_sub(lhs, rhs, prime.p),
                twiddle,
                prime);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    ulong output = ((ulong)prime_index * params.matrix_rings + ring)
        * params.ring_d + (ulong)thread_index;
    matrix_ntt[output] = values[thread_index];
}

kernel void akita_fp128_recursive_commit_matvec_barrier_reference(
    device const char *digits [[buffer(0)]],
    device const int *matrix_ntt [[buffer(1)]],
    device uint *residues [[buffer(2)]],
    device const D512LinearNttPrime *primes [[buffer(3)]],
    device const int *fwd_twiddles [[buffer(4)]],
    device const int *inv_twiddles [[buffer(5)]],
    device const int *psi_pows [[buffer(6)]],
    device const int *inverse_scale [[buffer(7)]],
    constant RecursiveCommitParams &params [[buffer(8)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup int values[RECURSIVE_COMMIT_THREADS];
    threadgroup int matrix_values[RECURSIVE_COMMIT_MAX_ROWS * RECURSIVE_COMMIT_MAX_D];
    int accumulators[RECURSIVE_COMMIT_MAX_ROWS * 2u];
    for (uint index = 0u; index < RECURSIVE_COMMIT_MAX_ROWS * 2u; ++index) {
        accumulators[index] = 0;
    }

    ulong group = (ulong)threadgroup_index.x;
    uint prime_index = (uint)(group % params.num_primes);
    ulong block_group = group / params.num_primes;
    ulong first_block = block_group * params.blocks_per_group;
    uint ring_d = (uint)params.ring_d;
    uint coefficient = thread_index % ring_d;
    uint slot = thread_index / ring_d;
    uint slots_per_wave = RECURSIVE_COMMIT_THREADS / ring_d;
    uint waves = (RECURSIVE_COMMIT_BLOCKS_PER_GROUP + slots_per_wave - 1u)
        / slots_per_wave;
    D512LinearNttPrime prime = primes[prime_index];
    ulong table_base = (ulong)prime_index * params.ring_d;

    for (ulong column = 0ul; column < params.num_cols; ++column) {
        if ((ulong)thread_index < params.num_rows * params.ring_d) {
            uint row = thread_index / ring_d;
            uint matrix_coefficient = thread_index - row * ring_d;
            ulong matrix_ring = (ulong)row * params.num_cols + column;
            ulong matrix_index = ((ulong)prime_index * params.matrix_rings + matrix_ring)
                * params.ring_d + (ulong)matrix_coefficient;
            matrix_values[thread_index] = matrix_ntt[matrix_index];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint wave = 0u; wave < waves; ++wave) {
            uint local_block = wave * slots_per_wave + slot;
            ulong block = first_block + (ulong)local_block;
            int digit = 0;
            if (local_block < RECURSIVE_COMMIT_BLOCKS_PER_GROUP
                && block < params.num_blocks) {
                ulong source_index = (block * params.num_cols + column)
                    * params.ring_d + (ulong)coefficient;
                digit = (int)digits[source_index];
            }
            values[thread_index] = d512_ntt_mul(
                d512_i32_to_mont(digit, prime),
                psi_pows[table_base + (ulong)coefficient],
                prime);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint len = ring_d >> 1u; len != 0u; len >>= 1u) {
                if (coefficient < ring_d >> 1u) {
                    uint butterfly_block = coefficient / len;
                    uint offset = coefficient - butterfly_block * len;
                    uint left = slot * ring_d
                        + butterfly_block * (len << 1u) + offset;
                    uint right = left + len;
                    int lhs = values[left];
                    int rhs = values[right];
                    int twiddle = fwd_twiddles[
                        table_base + (ulong)(len - 1u + offset)];
                    values[left] = d512_ntt_add(lhs, rhs, prime.p);
                    values[right] = d512_ntt_mul(
                        d512_ntt_sub(lhs, rhs, prime.p),
                        twiddle,
                        prime);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (local_block < RECURSIVE_COMMIT_BLOCKS_PER_GROUP
                && block < params.num_blocks) {
                int rhs = values[thread_index];
                for (uint row = 0u; row < (uint)params.num_rows; ++row) {
                    uint accumulator = wave * RECURSIVE_COMMIT_MAX_ROWS + row;
                    accumulators[accumulator] = d512_ntt_add(
                        accumulators[accumulator],
                        d512_ntt_mul(
                            matrix_values[row * ring_d + coefficient], rhs, prime),
                        prime.p);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint wave = 0u; wave < waves; ++wave) {
        uint local_block = wave * slots_per_wave + slot;
        ulong block = first_block + (ulong)local_block;
        bool live = local_block < RECURSIVE_COMMIT_BLOCKS_PER_GROUP
            && block < params.num_blocks;
        for (uint row = 0u; row < (uint)params.num_rows; ++row) {
            values[thread_index] = live
                ? accumulators[wave * RECURSIVE_COMMIT_MAX_ROWS + row]
                : 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint len = 1u; len < ring_d; len <<= 1u) {
                if (coefficient < ring_d >> 1u) {
                    uint butterfly_block = coefficient / len;
                    uint offset = coefficient - butterfly_block * len;
                    uint left = slot * ring_d
                        + butterfly_block * (len << 1u) + offset;
                    uint right = left + len;
                    int lhs = values[left];
                    int rhs = d512_ntt_mul(
                        values[right],
                        inv_twiddles[table_base + (ulong)(len - 1u + offset)],
                        prime);
                    values[left] = d512_ntt_add(lhs, rhs, prime.p);
                    values[right] = d512_ntt_sub(lhs, rhs, prime.p);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (live) {
                int scaled = d512_ntt_mul(
                    values[thread_index],
                    inverse_scale[table_base + (ulong)coefficient],
                    prime);
                int canonical = d512_ntt_mul_raw(scaled, 1, prime);
                canonical = d512_ntt_reduce((long)canonical, prime.p);
                ulong output_coefficient = (block * params.num_rows + (ulong)row)
                    * params.ring_d + (ulong)coefficient;
                residues[output_coefficient * params.num_primes + (ulong)prime_index] =
                    (uint)canonical;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

kernel void akita_fp128_recursive_commit_matvec(
    device const char *digits [[buffer(0)]],
    device const int *matrix_ntt [[buffer(1)]],
    device uint *residues [[buffer(2)]],
    device const D512LinearNttPrime *primes [[buffer(3)]],
    device const int *fwd_twiddles [[buffer(4)]],
    device const int *inv_twiddles [[buffer(5)]],
    device const int *psi_pows [[buffer(6)]],
    device const int *inverse_scale [[buffer(7)]],
    constant RecursiveCommitParams &params [[buffer(8)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    int values[4] = {0, 0, 0, 0};
    int accumulators[RECURSIVE_COMMIT_MAX_ROWS * 4u];
    for (uint index = 0u; index < RECURSIVE_COMMIT_MAX_ROWS * 4u; ++index) {
        accumulators[index] = 0;
    }

    ulong group = (ulong)threadgroup_index.x;
    uint prime_index = (uint)(group % params.num_primes);
    ulong block_group = group / params.num_primes;
    uint block_in_group = thread_index >> 5u;
    uint lane = thread_index & 31u;
    ulong block = block_group * params.blocks_per_group + (ulong)block_in_group;
    bool live = block_in_group < RECURSIVE_COMMIT_BLOCKS_PER_GROUP
        && block < params.num_blocks;
    uint ring_d = (uint)params.ring_d;
    uint register_count = ring_d >> 5u;
    D512LinearNttPrime prime = primes[prime_index];
    ulong table_base = (ulong)prime_index * params.ring_d;

    for (ulong column = 0ul; column < params.num_cols; ++column) {
        for (uint register_index = 0u; register_index < register_count; ++register_index) {
            uint coefficient = lane + (register_index << 5u);
            int digit = 0;
            if (live) {
                ulong source_index = (block * params.num_cols + column)
                    * params.ring_d + (ulong)coefficient;
                digit = (int)digits[source_index];
            }
            values[register_index] = d512_ntt_mul(
                d512_i32_to_mont(digit, prime),
                psi_pows[table_base + (ulong)coefficient],
                prime);
        }

        if (ring_d == 128u) {
            int value_0 = values[0];
            int value_1 = values[1];
            int value_2 = values[2];
            int value_3 = values[3];
            values[0] = d512_ntt_add(value_0, value_2, prime.p);
            values[2] = d512_ntt_mul(
                d512_ntt_sub(value_0, value_2, prime.p),
                fwd_twiddles[table_base + 63ul + (ulong)lane],
                prime);
            values[1] = d512_ntt_add(value_1, value_3, prime.p);
            values[3] = d512_ntt_mul(
                d512_ntt_sub(value_1, value_3, prime.p),
                fwd_twiddles[table_base + 95ul + (ulong)lane],
                prime);
        }

        int stage_32_twiddle = fwd_twiddles[table_base + 31ul + (ulong)lane];
        for (uint pair = 0u; pair < register_count; pair += 2u) {
            int lhs = values[pair];
            int rhs = values[pair + 1u];
            values[pair] = d512_ntt_add(lhs, rhs, prime.p);
            values[pair + 1u] = d512_ntt_mul(
                d512_ntt_sub(lhs, rhs, prime.p),
                stage_32_twiddle,
                prime);
        }

        for (uint len = 16u; len != 0u; len >>= 1u) {
            bool right_lane = (lane & len) != 0u;
            uint offset = lane & (len - 1u);
            int twiddle = fwd_twiddles[table_base + (ulong)(len - 1u + offset)];
            for (uint register_index = 0u;
                 register_index < register_count;
                 ++register_index) {
                int value = values[register_index];
                int partner = simd_shuffle_xor(value, len);
                int lhs = right_lane ? partner : value;
                int rhs = right_lane ? value : partner;
                values[register_index] = right_lane
                    ? d512_ntt_mul(d512_ntt_sub(lhs, rhs, prime.p), twiddle, prime)
                    : d512_ntt_add(lhs, rhs, prime.p);
            }
        }

        if (live) {
            for (uint row = 0u; row < (uint)params.num_rows; ++row) {
                ulong matrix_ring = (ulong)row * params.num_cols + column;
                ulong matrix_base = ((ulong)prime_index * params.matrix_rings + matrix_ring)
                    * params.ring_d;
                for (uint register_index = 0u;
                     register_index < register_count;
                     ++register_index) {
                    uint coefficient = lane + (register_index << 5u);
                    uint accumulator = row * 4u + register_index;
                    accumulators[accumulator] = d512_ntt_add(
                        accumulators[accumulator],
                        d512_ntt_mul(
                            matrix_ntt[matrix_base + (ulong)coefficient],
                            values[register_index],
                            prime),
                        prime.p);
                }
            }
        }
    }

    if (!live) {
        return;
    }
    for (uint row = 0u; row < (uint)params.num_rows; ++row) {
        for (uint register_index = 0u; register_index < register_count; ++register_index) {
            values[register_index] = accumulators[row * 4u + register_index];
        }
        for (uint len = 1u; len <= 16u; len <<= 1u) {
            bool right_lane = (lane & len) != 0u;
            uint offset = lane & (len - 1u);
            int twiddle = inv_twiddles[table_base + (ulong)(len - 1u + offset)];
            for (uint register_index = 0u;
                 register_index < register_count;
                 ++register_index) {
                int value = values[register_index];
                int partner = simd_shuffle_xor(value, len);
                int lhs = right_lane ? partner : value;
                int rhs_raw = right_lane ? value : partner;
                int rhs = d512_ntt_mul(rhs_raw, twiddle, prime);
                values[register_index] = right_lane
                    ? d512_ntt_sub(lhs, rhs, prime.p)
                    : d512_ntt_add(lhs, rhs, prime.p);
            }
        }

        int stage_32_twiddle = inv_twiddles[table_base + 31ul + (ulong)lane];
        for (uint pair = 0u; pair < register_count; pair += 2u) {
            int lhs = values[pair];
            int rhs = d512_ntt_mul(values[pair + 1u], stage_32_twiddle, prime);
            values[pair] = d512_ntt_add(lhs, rhs, prime.p);
            values[pair + 1u] = d512_ntt_sub(lhs, rhs, prime.p);
        }
        if (ring_d == 128u) {
            int value_0 = values[0];
            int value_1 = values[1];
            int rhs_2 = d512_ntt_mul(
                values[2],
                inv_twiddles[table_base + 63ul + (ulong)lane],
                prime);
            int rhs_3 = d512_ntt_mul(
                values[3],
                inv_twiddles[table_base + 95ul + (ulong)lane],
                prime);
            values[0] = d512_ntt_add(value_0, rhs_2, prime.p);
            values[2] = d512_ntt_sub(value_0, rhs_2, prime.p);
            values[1] = d512_ntt_add(value_1, rhs_3, prime.p);
            values[3] = d512_ntt_sub(value_1, rhs_3, prime.p);
        }

        for (uint register_index = 0u; register_index < register_count; ++register_index) {
            uint coefficient = lane + (register_index << 5u);
            int scaled = d512_ntt_mul(
                values[register_index],
                inverse_scale[table_base + (ulong)coefficient],
                prime);
            int canonical = d512_ntt_mul_raw(scaled, 1, prime);
            canonical = d512_ntt_reduce((long)canonical, prime.p);
            ulong output_coefficient = (block * params.num_rows + (ulong)row)
                * params.ring_d + (ulong)coefficient;
            residues[output_coefficient * params.num_primes + (ulong)prime_index] =
                (uint)canonical;
        }
    }
}

kernel void akita_fp128_recursive_commit_reconstruct(
    device const uint *residues [[buffer(0)]],
    device AkitaFp128 *output [[buffer(1)]],
    device const D512LinearNttPrime *primes [[buffer(2)]],
    device const uint *garner_gamma [[buffer(3)]],
    device const AkitaFp128 *field_partial_products [[buffer(4)]],
    constant RecursiveCommitParams &params [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    ulong coefficient = (ulong)threadgroup_index.x * 256ul + (ulong)thread_index;
    if (coefficient >= params.output_coefficients) {
        return;
    }
    long digits[D512_LINEAR_NTT_PRIMES];
    ulong residue_base = coefficient * params.num_primes;
    for (uint prime_index = 0u; prime_index < D512_LINEAR_NTT_PRIMES; ++prime_index) {
        long modulus = (long)primes[prime_index].p;
        long digit = (long)residues[residue_base + (ulong)prime_index];
        for (uint prior = 0u; prior < prime_index; ++prior) {
            digit = d512_positive_mod(digit - digits[prior], modulus);
            digit = (digit * (long)garner_gamma[
                prime_index * D512_LINEAR_NTT_PRIMES + prior]) % modulus;
        }
        digits[prime_index] = digit > modulus / 2l ? digit - modulus : digit;
    }

    AkitaFp128 reconstructed = akita_zero();
    for (uint prime_index = 0u; prime_index < D512_LINEAR_NTT_PRIMES; ++prime_index) {
        reconstructed = akita_add(
            reconstructed,
            akita_mul_signed_small(field_partial_products[prime_index], digits[prime_index]));
    }
    output[coefficient] = reconstructed;
}

constant ulong AKITA_BLAKE2B_IV[8] = {
    0x6a09e667f3bcc908ul, 0xbb67ae8584caa73bul,
    0x3c6ef372fe94f82bul, 0xa54ff53a5f1d36f1ul,
    0x510e527fade682d1ul, 0x9b05688c2b3e6c1ful,
    0x1f83d9abfb41bd6bul, 0x5be0cd19137e2179ul,
};

constant uchar AKITA_BLAKE2B_SIGMA[12][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
    { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
    { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
    { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
    { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
    { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
    { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
    { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
    { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
};

inline ulong akita_blake2b_load64(thread const uchar *bytes)
{
    ulong value = 0ul;
    for (uint i = 0u; i < 8u; ++i) {
        value |= (ulong)bytes[i] << (8u * i);
    }
    return value;
}

inline void akita_blake2b_store64(thread uchar *bytes, ulong value)
{
    for (uint i = 0u; i < 8u; ++i) {
        bytes[i] = (uchar)(value >> (8u * i));
    }
}

inline ulong akita_blake2b_rotr(ulong value, uint shift)
{
    return (value >> shift) | (value << (64u - shift));
}

inline void akita_blake2b_g(
    thread ulong *v,
    uint a,
    uint b,
    uint c,
    uint d,
    ulong x,
    ulong y)
{
    v[a] = v[a] + v[b] + x;
    v[d] = akita_blake2b_rotr(v[d] ^ v[a], 32u);
    v[c] += v[d];
    v[b] = akita_blake2b_rotr(v[b] ^ v[c], 24u);
    v[a] = v[a] + v[b] + y;
    v[d] = akita_blake2b_rotr(v[d] ^ v[a], 16u);
    v[c] += v[d];
    v[b] = akita_blake2b_rotr(v[b] ^ v[c], 63u);
}

inline void akita_blake2b_compress(
    thread ulong *h,
    thread const uchar *block,
    ulong byte_count,
    bool final_block)
{
    ulong m[16];
    ulong v[16];
    for (uint i = 0u; i < 16u; ++i) {
        m[i] = akita_blake2b_load64(block + 8u * i);
        v[i] = i < 8u ? h[i] : AKITA_BLAKE2B_IV[i - 8u];
    }
    v[12] ^= byte_count;
    if (final_block) {
        v[14] = ~v[14];
    }
    for (uint round = 0u; round < 12u; ++round) {
        constant const uchar *s = AKITA_BLAKE2B_SIGMA[round];
        akita_blake2b_g(v, 0u, 4u, 8u, 12u, m[s[0]], m[s[1]]);
        akita_blake2b_g(v, 1u, 5u, 9u, 13u, m[s[2]], m[s[3]]);
        akita_blake2b_g(v, 2u, 6u, 10u, 14u, m[s[4]], m[s[5]]);
        akita_blake2b_g(v, 3u, 7u, 11u, 15u, m[s[6]], m[s[7]]);
        akita_blake2b_g(v, 0u, 5u, 10u, 15u, m[s[8]], m[s[9]]);
        akita_blake2b_g(v, 1u, 6u, 11u, 12u, m[s[10]], m[s[11]]);
        akita_blake2b_g(v, 2u, 7u, 8u, 13u, m[s[12]], m[s[13]]);
        akita_blake2b_g(v, 3u, 4u, 9u, 14u, m[s[14]], m[s[15]]);
    }
    for (uint i = 0u; i < 8u; ++i) {
        h[i] ^= v[i] ^ v[i + 8u];
    }
}

inline void akita_blake2b512(
    thread const uchar *input,
    ulong input_len,
    thread ulong *output)
{
    ulong h[8];
    for (uint i = 0u; i < 8u; ++i) {
        h[i] = AKITA_BLAKE2B_IV[i];
    }
    h[0] ^= 0x01010040ul;

    uchar block[128];
    ulong offset = 0ul;
    ulong byte_count = 0ul;
    while (input_len - offset > 128ul) {
        for (uint i = 0u; i < 128u; ++i) {
            block[i] = input[offset + (ulong)i];
        }
        offset += 128ul;
        byte_count += 128ul;
        akita_blake2b_compress(h, block, byte_count, false);
    }
    ulong remaining = input_len - offset;
    for (uint i = 0u; i < 128u; ++i) {
        block[i] = (ulong)i < remaining ? input[offset + (ulong)i] : (uchar)0;
    }
    byte_count += remaining;
    akita_blake2b_compress(h, block, byte_count, true);
    for (uint i = 0u; i < 8u; ++i) {
        output[i] = h[i];
    }
}

inline void akita_blake2b_copy_words_to_bytes(
    thread const ulong *words,
    thread uchar *bytes)
{
    for (uint i = 0u; i < 8u; ++i) {
        akita_blake2b_store64(bytes + 8u * i, words[i]);
    }
}

inline AkitaFp128 akita_fp128_from_blake2b_words(ulong low, ulong high)
{
    AkitaFp128 value;
    value.limb = uint4((uint)low, (uint)(low >> 32u), (uint)high, (uint)(high >> 32u));
    if (value.limb.y == 0xffffffffu
        && value.limb.z == 0xffffffffu
        && value.limb.w == 0xffffffffu
        && value.limb.x >= 0x00005809u) {
        value.limb = uint4(value.limb.x - 0x00005809u, 0u, 0u, 0u);
    }
    return value;
}

kernel void akita_fp128_blake2b_sumcheck_challenge(
    device uchar *chaining_value [[buffer(0)]],
    device const uchar *claim [[buffer(1)]],
    device const uchar *coefficients [[buffer(2)]],
    device AkitaFp128 *challenge [[buffer(3)]],
    constant Blake2bSumcheckChallengeParams &params [[buffer(4)]])
{
    uchar input[320];
    ulong digest[8];

    for (uint i = 0u; i < 320u; ++i) {
        input[i] = (uchar)0;
    }
    input[127] = (uchar)2;
    for (uint i = 0u; i < 64u; ++i) {
        input[128u + i] = chaining_value[i];
    }
    for (uint i = 0u; i < 8u; ++i) {
        input[192u + i] = (uchar)(params.prior_squeezed_bytes >> (56u - 8u * i));
    }
    akita_blake2b512(input, 200ul, digest);

    for (uint i = 0u; i < 320u; ++i) {
        input[i] = (uchar)0;
    }
    input[127] = (uchar)0;
    akita_blake2b_copy_words_to_bytes(digest, input + 128u);
    ulong cursor = 192ul;
    if (params.include_claim != 0ul) {
        input[cursor] = (uchar)16;
        cursor += 8ul;
        for (uint i = 0u; i < 16u; ++i) {
            input[cursor + (ulong)i] = claim[i];
        }
        cursor += 16ul;
    }
    ulong coefficient_bytes = params.coefficient_count * 16ul;
    input[cursor] = (uchar)coefficient_bytes;
    cursor += 8ul;
    for (ulong i = 0ul; i < coefficient_bytes; ++i) {
        input[cursor + i] = coefficients[i];
    }
    cursor += coefficient_bytes;
    akita_blake2b512(input, cursor, digest);

    akita_blake2b_copy_words_to_bytes(digest, input);
    akita_blake2b512(input, 64ul, digest);
    for (uint i = 0u; i < 8u; ++i) {
        akita_blake2b_store64(input + 8u * i, digest[i]);
    }
    for (uint i = 0u; i < 64u; ++i) {
        chaining_value[i] = input[i];
    }

    for (uint i = 0u; i < 320u; ++i) {
        input[i] = (uchar)0;
    }
    input[127] = (uchar)1;
    akita_blake2b_copy_words_to_bytes(digest, input + 128u);
    akita_blake2b512(input, 200ul, digest);
    AkitaFp128 low = akita_fp128_from_blake2b_words(digest[0], digest[1]);
    AkitaFp128 high = akita_fp128_from_blake2b_words(digest[2], digest[3]);
    challenge[0] = akita_add(
        low,
        akita_mul_signed_small(high, (long)AKITA_OFFSET));
}

inline bool akita_fp128_is_zero(AkitaFp128 value)
{
    return all(value.limb == uint4(0u));
}

inline uchar akita_fp128_serialized_byte(AkitaFp128 value, uint index)
{
    uint word = value.limb[index >> 2u];
    return (uchar)(word >> (8u * (index & 3u)));
}

kernel void akita_fp128_blake2b_relation_sumcheck_round(
    device uchar *chaining_value [[buffer(0)]],
    device const AkitaFp128 *main_coefficients [[buffer(1)]],
    device const AkitaFp128 *additional_coefficients [[buffer(2)]],
    device AkitaFp128 *proof_coefficients [[buffer(3)]],
    device uint *coefficient_count_output [[buffer(4)]],
    device AkitaFp128 *challenge [[buffer(5)]],
    constant DirectRelationTranscriptParams &params [[buffer(6)]])
{
    AkitaFp128 coefficients[3];
    for (uint i = 0u; i < 3u; ++i) {
        coefficients[i] = params.has_additional != 0ul
            ? akita_add(main_coefficients[i], additional_coefficients[i])
            : main_coefficients[i];
        proof_coefficients[i] = coefficients[i];
    }
    uint coefficient_count = 3u;
    while (coefficient_count > 1u
        && akita_fp128_is_zero(coefficients[coefficient_count - 1u])) {
        coefficient_count -= 1u;
    }
    coefficient_count_output[0] = coefficient_count;

    uchar input[320];
    ulong digest[8];
    for (uint i = 0u; i < 320u; ++i) {
        input[i] = (uchar)0;
    }
    input[127] = (uchar)2;
    for (uint i = 0u; i < 64u; ++i) {
        input[128u + i] = chaining_value[i];
    }
    for (uint i = 0u; i < 8u; ++i) {
        input[192u + i] = (uchar)(params.prior_squeezed_bytes >> (56u - 8u * i));
    }
    akita_blake2b512(input, 200ul, digest);

    for (uint i = 0u; i < 320u; ++i) {
        input[i] = (uchar)0;
    }
    akita_blake2b_copy_words_to_bytes(digest, input + 128u);
    ulong cursor = 192ul;
    ulong coefficient_bytes = (ulong)coefficient_count * 16ul;
    input[cursor] = (uchar)coefficient_bytes;
    cursor += 8ul;
    for (uint coefficient = 0u; coefficient < coefficient_count; ++coefficient) {
        for (uint byte = 0u; byte < 16u; ++byte) {
            input[cursor++] = akita_fp128_serialized_byte(coefficients[coefficient], byte);
        }
    }
    akita_blake2b512(input, cursor, digest);

    akita_blake2b_copy_words_to_bytes(digest, input);
    akita_blake2b512(input, 64ul, digest);
    for (uint i = 0u; i < 8u; ++i) {
        akita_blake2b_store64(input + 8u * i, digest[i]);
    }
    for (uint i = 0u; i < 64u; ++i) {
        chaining_value[i] = input[i];
    }

    for (uint i = 0u; i < 320u; ++i) {
        input[i] = (uchar)0;
    }
    input[127] = (uchar)1;
    akita_blake2b_copy_words_to_bytes(digest, input + 128u);
    akita_blake2b512(input, 200ul, digest);
    AkitaFp128 low = akita_fp128_from_blake2b_words(digest[0], digest[1]);
    AkitaFp128 high = akita_fp128_from_blake2b_words(digest[2], digest[3]);
    challenge[0] = akita_add(
        low,
        akita_mul_signed_small(high, (long)AKITA_OFFSET));
}

inline AkitaFp128 akita_direct_range_eq_weight(
    device const AkitaFp128 *e_first,
    device const AkitaFp128 *e_second,
    ulong pair_index,
    constant DirectRangeParams &params)
{
    ulong low = pair_index & (params.num_first - 1ul);
    ulong high = pair_index / params.num_first;
    return akita_mul(e_first[low], e_second[high]);
}

// Coefficients q1..q4 of the per-pair range polynomial q(X) = A(X) B(X) with
// A(X) = (l + dX)(l + dX - 2) and, for basis eight, B(X) = (l + dX)^2 - 18(l + dX) + 72.
// The constant coefficient q0 is omitted: the normalized eq-factored round message
// stores `[q_1, ..., q_d]` and the verifier recovers q0 from the running claim.
inline void akita_direct_range_q_coefficients(
    AkitaFp128 left,
    AkitaFp128 right,
    uint basis,
    thread AkitaFp128 &q1,
    thread AkitaFp128 &q2,
    thread AkitaFp128 &q3,
    thread AkitaFp128 &q4)
{
    AkitaFp128 delta = akita_sub(right, left);
    AkitaFp128 delta_squared = akita_mul(delta, delta);
    if (basis == 4u) {
        q1 = akita_mul(
            delta,
            akita_sub(akita_mul_signed_small(left, 2l), akita_from_u32(2u)));
        q2 = delta_squared;
        q3 = akita_zero();
        q4 = akita_zero();
        return;
    }

    AkitaFp128 left_squared = akita_mul(left, left);
    AkitaFp128 first_quadratic = akita_sub(
        left_squared,
        akita_mul_signed_small(left, 2l));
    AkitaFp128 second_quadratic = akita_add(
        akita_sub(left_squared, akita_mul_signed_small(left, 18l)),
        akita_from_u32(72u));
    AkitaFp128 first_linear = akita_mul(
        delta,
        akita_sub(akita_mul_signed_small(left, 2l), akita_from_u32(2u)));
    AkitaFp128 second_linear = akita_mul(
        delta,
        akita_sub(akita_mul_signed_small(left, 2l), akita_from_u32(18u)));

    q1 = akita_add(
        akita_mul(first_quadratic, second_linear),
        akita_mul(first_linear, second_quadratic));
    q2 = akita_add(
        akita_add(
            akita_mul(first_quadratic, delta_squared),
            akita_mul(first_linear, second_linear)),
        akita_mul(delta_squared, second_quadratic));
    q3 = akita_mul(delta_squared, akita_add(first_linear, second_linear));
    q4 = akita_mul(delta_squared, delta_squared);
}

inline AkitaFp128 akita_direct_range_prefix_weight(
    device const AkitaFp128 *weights_or_challenges,
    ulong prefix,
    constant DirectRangeParams &params)
{
    if (params.resident_challenges == 0ul) {
        return weights_or_challenges[prefix];
    }
    AkitaFp128 weight = akita_from_u32(1u);
    ulong width = params.prefix_size;
    uint challenge_index = 0u;
    while (width > 1ul) {
        AkitaFp128 r = weights_or_challenges[challenge_index];
        AkitaFp128 factor = ((prefix >> challenge_index) & 1ul) != 0ul
            ? r
            : akita_sub(akita_from_u32(1u), r);
        weight = akita_mul(weight, factor);
        width >>= 1ul;
        challenge_index += 1u;
    }
    return weight;
}

kernel void akita_fp128_direct_range_initial_partials(
    device const char *digits [[buffer(0)]],
    device const AkitaFp128 *e_first [[buffer(1)]],
    device const AkitaFp128 *e_second [[buffer(2)]],
    device AkitaFp128 *partials [[buffer(3)]],
    constant DirectRangeParams &params [[buffer(4)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup AkitaFp128 reduction_1[256];
    threadgroup AkitaFp128 reduction_2[256];
    threadgroup AkitaFp128 reduction_3[256];
    threadgroup AkitaFp128 reduction_4[256];

    AkitaFp128 sum_1 = akita_zero();
    AkitaFp128 sum_2 = akita_zero();
    AkitaFp128 sum_3 = akita_zero();
    AkitaFp128 sum_4 = akita_zero();
    ulong global_thread = (ulong)threadgroup_index.x * 256ul + (ulong)thread_index;
    ulong stride = params.workgroups * 256ul;
    for (ulong pair = global_thread; pair < params.pair_count; pair += stride) {
        ulong left_index = pair * 2ul;
        int left_digit = left_index < params.live_len ? (int)digits[left_index] : 0;
        int right_digit = left_index + 1ul < params.live_len
            ? (int)digits[left_index + 1ul]
            : 0;
        long left = (long)left_digit * (long)(left_digit + 1);
        long right = (long)right_digit * (long)(right_digit + 1);
        long delta = right - left;
        long delta_squared = delta * delta;
        // Same `[q_1, ..., q_d]` layout as the fold kernels: q_0 is implied by the
        // running claim and must not occupy the first slot.
        long c1 = delta * (2l * left - 2l);
        long c2 = delta_squared;
        long c3 = 0l;
        long c4 = 0l;
        if (params.basis == 8ul) {
            c1 = left * (left - 2l) * delta * (2l * left - 18l)
                + delta * (2l * left - 2l) * (left * left - 18l * left + 72l);
            c2 = delta_squared * (108l - 60l * left + 6l * left * left);
            c3 = delta_squared * delta * (-20l + 4l * left);
            c4 = delta_squared * delta_squared;
        }
        AkitaFp128 weight = akita_direct_range_eq_weight(
            e_first, e_second, pair, params);
        sum_1 = akita_add(sum_1, akita_mul_signed_small(weight, c1));
        sum_2 = akita_add(sum_2, akita_mul_signed_small(weight, c2));
        sum_3 = akita_add(sum_3, akita_mul_signed_small(weight, c3));
        sum_4 = akita_add(sum_4, akita_mul_signed_small(weight, c4));
    }
    reduction_1[thread_index] = sum_1;
    reduction_2[thread_index] = sum_2;
    reduction_3[thread_index] = sum_3;
    reduction_4[thread_index] = sum_4;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 128u; width != 0u; width >>= 1u) {
        if (thread_index < width) {
            reduction_1[thread_index] = akita_add(
                reduction_1[thread_index], reduction_1[thread_index + width]);
            reduction_2[thread_index] = akita_add(
                reduction_2[thread_index], reduction_2[thread_index + width]);
            reduction_3[thread_index] = akita_add(
                reduction_3[thread_index], reduction_3[thread_index + width]);
            reduction_4[thread_index] = akita_add(
                reduction_4[thread_index], reduction_4[thread_index + width]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index == 0u) {
        ulong output = (ulong)threadgroup_index.x * 4ul;
        partials[output] = reduction_1[0];
        partials[output + 1ul] = reduction_2[0];
        partials[output + 2ul] = reduction_3[0];
        partials[output + 3ul] = reduction_4[0];
    }
}

kernel void akita_fp128_direct_range_reduce(
    device const AkitaFp128 *partials [[buffer(0)]],
    device AkitaFp128 *output [[buffer(1)]],
    constant DirectRangeParams &params [[buffer(2)]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    threadgroup AkitaFp128 reduction_0[256];
    threadgroup AkitaFp128 reduction_2[256];
    threadgroup AkitaFp128 reduction_3[256];
    threadgroup AkitaFp128 reduction_4[256];
    AkitaFp128 sum_0 = akita_zero();
    AkitaFp128 sum_2 = akita_zero();
    AkitaFp128 sum_3 = akita_zero();
    AkitaFp128 sum_4 = akita_zero();
    for (ulong group = (ulong)thread_index;
         group < params.workgroups;
         group += 256ul) {
        ulong input = group * 4ul;
        sum_0 = akita_add(sum_0, partials[input]);
        sum_2 = akita_add(sum_2, partials[input + 1ul]);
        sum_3 = akita_add(sum_3, partials[input + 2ul]);
        sum_4 = akita_add(sum_4, partials[input + 3ul]);
    }
    reduction_0[thread_index] = sum_0;
    reduction_2[thread_index] = sum_2;
    reduction_3[thread_index] = sum_3;
    reduction_4[thread_index] = sum_4;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 128u; width != 0u; width >>= 1u) {
        if (thread_index < width) {
            reduction_0[thread_index] = akita_add(
                reduction_0[thread_index], reduction_0[thread_index + width]);
            reduction_2[thread_index] = akita_add(
                reduction_2[thread_index], reduction_2[thread_index + width]);
            reduction_3[thread_index] = akita_add(
                reduction_3[thread_index], reduction_3[thread_index + width]);
            reduction_4[thread_index] = akita_add(
                reduction_4[thread_index], reduction_4[thread_index + width]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index == 0u) {
        output[0] = reduction_0[0];
        output[1] = reduction_2[0];
        output[2] = reduction_3[0];
        output[3] = reduction_4[0];
    }
}

kernel void akita_fp128_direct_range_compact_fold_partials(
    device const char *digits [[buffer(0)]],
    device AkitaFp128 *folded [[buffer(1)]],
    device const AkitaFp128 *e_first [[buffer(2)]],
    device const AkitaFp128 *e_second [[buffer(3)]],
    device AkitaFp128 *partials [[buffer(4)]],
    device const AkitaFp128 *prefix_weights [[buffer(5)]],
    constant DirectRangeParams &params [[buffer(6)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup AkitaFp128 reduction_0[256];
    threadgroup AkitaFp128 reduction_2[256];
    threadgroup AkitaFp128 reduction_3[256];
    threadgroup AkitaFp128 reduction_4[256];
    AkitaFp128 sum_0 = akita_zero();
    AkitaFp128 sum_2 = akita_zero();
    AkitaFp128 sum_3 = akita_zero();
    AkitaFp128 sum_4 = akita_zero();
    ulong global_thread = (ulong)threadgroup_index.x * 256ul + (ulong)thread_index;
    ulong stride = params.workgroups * 256ul;
    for (ulong pair = global_thread; pair < params.pair_count; pair += stride) {
        AkitaFp128 values[2];
        for (uint side = 0u; side < 2u; ++side) {
            ulong output_index = pair * 2ul + (ulong)side;
            AkitaFp128 value = akita_zero();
            if (output_index < params.current_live_len) {
                ulong input_start = output_index * params.prefix_size;
                for (ulong prefix = 0ul; prefix < params.prefix_size; ++prefix) {
                    ulong input_index = input_start + prefix;
                    int digit = input_index < params.live_len ? (int)digits[input_index] : 0;
                    long range_image = (long)digit * (long)(digit + 1);
                    value = akita_add(
                        value,
                        akita_mul_signed_small(
                            akita_direct_range_prefix_weight(
                                prefix_weights, prefix, params),
                            range_image));
                }
            }
            values[side] = value;
            if (params.materialize_prefix != 0ul
                && output_index < params.current_live_len) {
                folded[output_index] = value;
            }
        }
        AkitaFp128 q1;
        AkitaFp128 q2;
        AkitaFp128 q3;
        AkitaFp128 q4;
        akita_direct_range_q_coefficients(
            values[0], values[1], (uint)params.basis, q1, q2, q3, q4);
        AkitaFp128 weight = akita_direct_range_eq_weight(
            e_first, e_second, pair, params);
        sum_0 = akita_add(sum_0, akita_mul(weight, q1));
        sum_2 = akita_add(sum_2, akita_mul(weight, q2));
        sum_3 = akita_add(sum_3, akita_mul(weight, q3));
        sum_4 = akita_add(sum_4, akita_mul(weight, q4));
    }
    reduction_0[thread_index] = sum_0;
    reduction_2[thread_index] = sum_2;
    reduction_3[thread_index] = sum_3;
    reduction_4[thread_index] = sum_4;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 128u; width != 0u; width >>= 1u) {
        if (thread_index < width) {
            reduction_0[thread_index] = akita_add(
                reduction_0[thread_index], reduction_0[thread_index + width]);
            reduction_2[thread_index] = akita_add(
                reduction_2[thread_index], reduction_2[thread_index + width]);
            reduction_3[thread_index] = akita_add(
                reduction_3[thread_index], reduction_3[thread_index + width]);
            reduction_4[thread_index] = akita_add(
                reduction_4[thread_index], reduction_4[thread_index + width]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index == 0u) {
        ulong output = (ulong)threadgroup_index.x * 4ul;
        partials[output] = reduction_0[0];
        partials[output + 1ul] = reduction_2[0];
        partials[output + 2ul] = reduction_3[0];
        partials[output + 3ul] = reduction_4[0];
    }
}

kernel void akita_fp128_direct_range_field_fold_partials(
    device const AkitaFp128 *input [[buffer(0)]],
    device AkitaFp128 *folded [[buffer(1)]],
    device const AkitaFp128 *e_first [[buffer(2)]],
    device const AkitaFp128 *e_second [[buffer(3)]],
    device AkitaFp128 *partials [[buffer(4)]],
    constant AkitaFp128 &challenge [[buffer(5)]],
    constant DirectRangeParams &params [[buffer(6)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup AkitaFp128 reduction_0[256];
    threadgroup AkitaFp128 reduction_2[256];
    threadgroup AkitaFp128 reduction_3[256];
    threadgroup AkitaFp128 reduction_4[256];
    AkitaFp128 sum_0 = akita_zero();
    AkitaFp128 sum_2 = akita_zero();
    AkitaFp128 sum_3 = akita_zero();
    AkitaFp128 sum_4 = akita_zero();
    ulong global_thread = (ulong)threadgroup_index.x * 256ul + (ulong)thread_index;
    ulong stride = params.workgroups * 256ul;
    for (ulong pair = global_thread; pair < params.pair_count; pair += stride) {
        AkitaFp128 values[2];
        for (uint side = 0u; side < 2u; ++side) {
            ulong output_index = pair * 2ul + (ulong)side;
            AkitaFp128 value = akita_zero();
            if (output_index < params.current_live_len) {
                ulong input_index = output_index * 2ul;
                AkitaFp128 left = input_index < params.input_live_len
                    ? input[input_index]
                    : akita_zero();
                AkitaFp128 right = input_index + 1ul < params.input_live_len
                    ? input[input_index + 1ul]
                    : akita_zero();
                value = akita_add(
                    left,
                    akita_mul(challenge, akita_sub(right, left)));
                folded[output_index] = value;
            }
            values[side] = value;
        }
        AkitaFp128 q1;
        AkitaFp128 q2;
        AkitaFp128 q3;
        AkitaFp128 q4;
        akita_direct_range_q_coefficients(
            values[0], values[1], (uint)params.basis, q1, q2, q3, q4);
        AkitaFp128 weight = akita_direct_range_eq_weight(
            e_first, e_second, pair, params);
        sum_0 = akita_add(sum_0, akita_mul(weight, q1));
        sum_2 = akita_add(sum_2, akita_mul(weight, q2));
        sum_3 = akita_add(sum_3, akita_mul(weight, q3));
        sum_4 = akita_add(sum_4, akita_mul(weight, q4));
    }
    reduction_0[thread_index] = sum_0;
    reduction_2[thread_index] = sum_2;
    reduction_3[thread_index] = sum_3;
    reduction_4[thread_index] = sum_4;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 128u; width != 0u; width >>= 1u) {
        if (thread_index < width) {
            reduction_0[thread_index] = akita_add(
                reduction_0[thread_index], reduction_0[thread_index + width]);
            reduction_2[thread_index] = akita_add(
                reduction_2[thread_index], reduction_2[thread_index + width]);
            reduction_3[thread_index] = akita_add(
                reduction_3[thread_index], reduction_3[thread_index + width]);
            reduction_4[thread_index] = akita_add(
                reduction_4[thread_index], reduction_4[thread_index + width]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index == 0u) {
        ulong output = (ulong)threadgroup_index.x * 4ul;
        partials[output] = reduction_0[0];
        partials[output + 1ul] = reduction_2[0];
        partials[output + 2ul] = reduction_3[0];
        partials[output + 3ul] = reduction_4[0];
    }
}

kernel void akita_fp128_direct_range_finalize(
    device const AkitaFp128 *input [[buffer(0)]],
    device AkitaFp128 *output [[buffer(1)]],
    constant AkitaFp128 &challenge [[buffer(2)]],
    constant ulong &input_live_len [[buffer(3)]])
{
    AkitaFp128 left = input_live_len != 0ul ? input[0] : akita_zero();
    AkitaFp128 right = input_live_len > 1ul ? input[1] : akita_zero();
    output[0] = akita_add(left, akita_mul(challenge, akita_sub(right, left)));
}

inline int akita_stage2_prefix_integer_point(
    int value_00,
    int value_10,
    int value_01,
    int value_11,
    uint point)
{
    switch (point) {
        case 0u: return value_00;
        case 1u: return value_01;
        case 2u: return value_01 - value_00;
        case 3u: return value_10;
        case 4u: return value_11;
        case 5u: return value_11 - value_10;
        case 6u: return value_10 - value_00;
        case 7u: return value_11 - value_01;
        default: return value_11 - value_10 - value_01 + value_00;
    }
}

inline AkitaFp128 akita_stage2_prefix_field_point(
    AkitaFp128 value_00,
    AkitaFp128 value_10,
    AkitaFp128 value_01,
    AkitaFp128 value_11,
    uint point)
{
    switch (point) {
        case 0u: return value_00;
        case 1u: return value_01;
        case 2u: return akita_sub(value_01, value_00);
        case 3u: return value_10;
        case 4u: return value_11;
        case 5u: return akita_sub(value_11, value_10);
        case 6u: return akita_sub(value_10, value_00);
        case 7u: return akita_sub(value_11, value_01);
        default: return akita_add(
            akita_sub(value_11, value_10),
            akita_sub(value_00, value_01));
    }
}

kernel void akita_fp128_direct_relation_two_round_prefix_partials(
    device const char *digits [[buffer(0)]],
    device const AkitaFp128 *equality_first [[buffer(1)]],
    device const AkitaFp128 *equality_second [[buffer(2)]],
    device const AkitaFp128 *alpha_points [[buffer(3)]],
    device const AkitaFp128 *lane_weights [[buffer(4)]],
    device const AkitaFp128 *linear_values [[buffer(5)]],
    device const uint *source_offsets [[buffer(6)]],
    device const uint *lane_offsets [[buffer(7)]],
    device const uint *lane_segments [[buffer(8)]],
    device const DirectRelationLinearSegment *segments [[buffer(9)]],
    device AkitaFp128 *partials [[buffer(10)]],
    constant DirectRelationTwoRoundPrefixParams &params [[buffer(11)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup AkitaFp128 simd_sums[8];
    uint point_slot = threadgroup_index.y;
    bool norm_point = point_slot < 8u;
    uint point = norm_point
        ? point_slot + (point_slot >= params.norm_omitted_corner ? 1u : 0u)
        : point_slot - 7u;
    ulong lanes_per_group = 256ul * params.lanes_per_thread;
    ulong lane_base = (ulong)threadgroup_index.x * lanes_per_group;
    AkitaFp128 sum = akita_zero();

    for (ulong lane_offset = 0ul;
         lane_offset < params.lanes_per_thread;
         ++lane_offset) {
        ulong lane = lane_base + (ulong)thread_index + lane_offset * 256ul;
        if (lane >= params.live_lane_count) {
            continue;
        }
        AkitaFp128 inner = akita_zero();
        for (ulong y_quad = 0ul; y_quad < params.y_quads; ++y_quad) {
            ulong digit_base = lane * params.coefficient_count + 4ul * y_quad;
            int value_00 = (int)digits[digit_base];
            int value_10 = (int)digits[digit_base + 1ul];
            int value_01 = (int)digits[digit_base + 2ul];
            int value_11 = (int)digits[digit_base + 3ul];
            int witness_point = akita_stage2_prefix_integer_point(
                value_00, value_10, value_01, value_11, point);
            if (norm_point) {
                int norm_value = (point == 0u || point == 1u || point == 3u || point == 4u)
                    ? witness_point * (witness_point + 1)
                    : witness_point * witness_point;
                ulong flat_quad = lane * params.y_quads + y_quad;
                AkitaFp128 equality_low = equality_first[
                    flat_quad & (params.equality_first_len - 1ul)];
                if (params.lanes_per_thread == 2ul) {
                    inner = akita_add(
                        inner, akita_mul_signed_i32(equality_low, norm_value));
                } else {
                    AkitaFp128 equality = akita_mul(
                        equality_low,
                        equality_second[flat_quad / params.equality_first_len]);
                    inner = akita_add(
                        inner, akita_mul_signed_i32(equality, norm_value));
                }
            } else {
                AkitaFp128 alpha_point = alpha_points[
                    (ulong)(point - 1u) * params.y_quads + y_quad];
                inner = akita_add(
                    inner, akita_mul_signed_i32(alpha_point, witness_point));
            }
        }

        if (!norm_point) {
            sum = akita_add(sum, akita_mul(lane_weights[lane], inner));
            if (params.linear_mode != 0ul) {
                uint begin = lane_offsets[lane];
                uint end = lane_offsets[lane + 1ul];
                for (uint cursor = begin; cursor < end; ++cursor) {
                    DirectRelationLinearSegment segment = segments[lane_segments[cursor]];
                    ulong lane_offset = (lane - (ulong)segment.target_lane_start)
                        / (ulong)segment.target_lane_stride;
                    ulong source_lane = (ulong)segment.source_lane_start
                        + lane_offset * (ulong)segment.source_lane_stride;
                    ulong source_base =
                        ((ulong)source_offsets[segment.source_index] + source_lane)
                        * params.coefficient_count;
                    AkitaFp128 segment_inner = akita_zero();
                    for (ulong y_quad = 0ul; y_quad < params.y_quads; ++y_quad) {
                        ulong offset = source_base + 4ul * y_quad;
                        AkitaFp128 source_point = akita_stage2_prefix_field_point(
                            linear_values[offset],
                            linear_values[offset + 1ul],
                            linear_values[offset + 2ul],
                            linear_values[offset + 3ul],
                            point);
                        ulong digit_base = lane * params.coefficient_count + 4ul * y_quad;
                        int witness_point = akita_stage2_prefix_integer_point(
                            (int)digits[digit_base],
                            (int)digits[digit_base + 1ul],
                            (int)digits[digit_base + 2ul],
                            (int)digits[digit_base + 3ul],
                            point);
                        segment_inner = akita_add(
                            segment_inner,
                            akita_mul_signed_i32(source_point, witness_point));
                    }
                    sum = akita_add(sum, akita_mul(segment.factor, segment_inner));
                }
            }
        } else {
            sum = akita_add(sum, inner);
        }
    }

    sum = akita_simd_sum_fp128(sum);
    uint simd_lane = thread_index & 31u;
    uint simdgroup = thread_index >> 5u;
    if (simd_lane == 0u) {
        simd_sums[simdgroup] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        AkitaFp128 group_sum = simd_lane < 8u ? simd_sums[simd_lane] : akita_zero();
        group_sum = akita_simd_sum_fp128(group_sum);
        if (simd_lane == 0u) {
            if (norm_point && params.lanes_per_thread == 2ul) {
                ulong equality_index =
                    lane_base * params.y_quads / params.equality_first_len;
                group_sum = akita_mul(group_sum, equality_second[equality_index]);
            }
            partials[(ulong)point_slot * params.workgroups + threadgroup_index.x] = group_sum;
        }
    }
}

kernel void akita_fp128_direct_relation_two_round_prefix_reduce(
    device const AkitaFp128 *partials [[buffer(0)]],
    device AkitaFp128 *output [[buffer(1)]],
    constant DirectRelationTwoRoundPrefixParams &params [[buffer(2)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup AkitaFp128 simd_sums[8];
    ulong point = threadgroup_index.x;
    AkitaFp128 sum = akita_zero();
    for (ulong group = (ulong)thread_index;
         group < params.workgroups;
         group += 256ul) {
        sum = akita_add(sum, partials[point * params.workgroups + group]);
    }
    sum = akita_simd_sum_fp128(sum);
    uint simd_lane = thread_index & 31u;
    uint simdgroup = thread_index >> 5u;
    if (simd_lane == 0u) {
        simd_sums[simdgroup] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        AkitaFp128 result = simd_lane < 8u ? simd_sums[simd_lane] : akita_zero();
        result = akita_simd_sum_fp128(result);
        if (simd_lane == 0u) {
            output[point] = result;
        }
    }
}

inline AkitaFp128 akita_direct_relation_compact_value(
    device const char *digits,
    device const AkitaFp128 *prefix_weights,
    ulong index,
    constant DirectRelationParams &params)
{
    ulong input_start = index * params.prefix_size;
    AkitaFp128 value = akita_zero();
    for (ulong prefix = 0ul; prefix < params.prefix_size; ++prefix) {
        ulong input_index = input_start + prefix;
        int digit = input_index < params.live_len ? (int)digits[input_index] : 0;
        AkitaFp128 weight;
        if (params.resident_challenges != 0ul) {
            weight = akita_from_u32(1u);
            ulong width = params.prefix_size;
            uint challenge_index = 0u;
            while (width > 1ul) {
                AkitaFp128 challenge = prefix_weights[challenge_index];
                AkitaFp128 factor = ((prefix >> challenge_index) & 1ul) != 0ul
                    ? challenge
                    : akita_sub(akita_from_u32(1u), challenge);
                weight = akita_mul(weight, factor);
                width >>= 1ul;
                challenge_index += 1u;
            }
        } else {
            weight = prefix_weights[prefix];
        }
        value = akita_add(value, akita_mul_signed_small(weight, (long)digit));
    }
    return value;
}

inline AkitaFp128 akita_direct_relation_factored_linear_value(
    ulong lane,
    ulong coefficient,
    ulong current_coeff_count,
    device const AkitaFp128 *linear_values,
    device const uint *source_offsets,
    device const uint *lane_offsets,
    device const uint *lane_segments,
    device const DirectRelationLinearSegment *segments)
{
    AkitaFp128 value = akita_zero();
    uint begin = lane_offsets[lane];
    uint end = lane_offsets[lane + 1ul];
    for (uint cursor = begin; cursor < end; ++cursor) {
        DirectRelationLinearSegment segment = segments[lane_segments[cursor]];
        ulong lane_offset = (lane - (ulong)segment.target_lane_start)
            / (ulong)segment.target_lane_stride;
        ulong source_lane = (ulong)segment.source_lane_start
            + lane_offset * (ulong)segment.source_lane_stride;
        ulong source_index = ((ulong)source_offsets[segment.source_index] + source_lane)
            * current_coeff_count + coefficient;
        value = akita_add(
            value,
            akita_mul(segment.factor, linear_values[source_index]));
    }
    return value;
}

inline AkitaFp128 akita_direct_relation_linear_value(
    ulong index,
    device const AkitaFp128 *linear_values,
    device const uint *source_offsets,
    device const uint *lane_offsets,
    device const uint *lane_segments,
    device const DirectRelationLinearSegment *segments,
    constant DirectRelationParams &params)
{
    if (params.linear_mode == 0ul) {
        return akita_zero();
    }
    ulong lane = index / params.current_coeff_count;
    if (lane >= params.live_lane_count) {
        return akita_zero();
    }
    if (params.linear_mode == 2ul) {
        return linear_values[lane];
    }
    ulong coefficient = index - lane * params.current_coeff_count;
    return akita_direct_relation_factored_linear_value(
        lane,
        coefficient,
        params.current_coeff_count,
        linear_values,
        source_offsets,
        lane_offsets,
        lane_segments,
        segments);
}

kernel void akita_fp128_direct_relation_linear_fold(
    device const AkitaFp128 *input [[buffer(0)]],
    device AkitaFp128 *output [[buffer(1)]],
    device const uint *source_offsets [[buffer(2)]],
    device const uint *lane_offsets [[buffer(3)]],
    device const uint *lane_segments [[buffer(4)]],
    device const DirectRelationLinearSegment *segments [[buffer(5)]],
    constant AkitaFp128 &challenge [[buffer(6)]],
    constant DirectRelationLinearFoldParams &params [[buffer(7)]],
    uint thread_index [[thread_position_in_grid]])
{
    ulong index = (ulong)thread_index;
    if (index >= params.output_len) {
        return;
    }
    if (params.mode == 1ul) {
        ulong next_coeff_count = params.current_coeff_count / 2ul;
        ulong source_lane = index / next_coeff_count;
        ulong coefficient = index - source_lane * next_coeff_count;
        ulong input_index = source_lane * params.current_coeff_count + 2ul * coefficient;
        AkitaFp128 left = input[input_index];
        output[index] = akita_add(
            left,
            akita_mul(challenge, akita_sub(input[input_index + 1ul], left)));
        return;
    }

    ulong left_lane = 2ul * index;
    AkitaFp128 left;
    AkitaFp128 right = akita_zero();
    if (params.mode == 2ul) {
        left = akita_direct_relation_factored_linear_value(
            left_lane, 0ul, 1ul, input, source_offsets, lane_offsets, lane_segments, segments);
        if (left_lane + 1ul < params.current_live_lane_count) {
            right = akita_direct_relation_factored_linear_value(
                left_lane + 1ul,
                0ul,
                1ul,
                input,
                source_offsets,
                lane_offsets,
                lane_segments,
                segments);
        }
    } else {
        left = input[left_lane];
        if (left_lane + 1ul < params.current_live_lane_count) {
            right = input[left_lane + 1ul];
        }
    }
    output[index] = akita_add(left, akita_mul(challenge, akita_sub(right, left)));
}

kernel void akita_fp128_direct_relation_alpha_fold(
    device const AkitaFp128 *input [[buffer(0)]],
    device AkitaFp128 *output [[buffer(1)]],
    constant AkitaFp128 &challenge [[buffer(2)]],
    constant ulong &output_len [[buffer(3)]],
    uint thread_index [[thread_position_in_grid]])
{
    ulong index = (ulong)thread_index;
    if (index >= output_len) {
        return;
    }
    AkitaFp128 left = input[2ul * index];
    output[index] = akita_add(
        left,
        akita_mul(challenge, akita_sub(input[2ul * index + 1ul], left)));
}

kernel void akita_fp128_direct_relation_scalar_advance(
    constant DirectRelationScalars &current [[buffer(0)]],
    constant AkitaFp128 &challenge [[buffer(1)]],
    constant AkitaFp128 &tau_current [[buffer(2)]],
    constant AkitaFp128 &tau_next [[buffer(3)]],
    device DirectRelationScalars *next [[buffer(4)]])
{
    AkitaFp128 one = akita_from_u32(1u);
    AkitaFp128 scalar = akita_add(current.l_at_0, current.l_at_1);
    AkitaFp128 bound = akita_add(
        akita_mul(tau_current, challenge),
        akita_mul(akita_sub(one, tau_current), akita_sub(one, challenge)));
    AkitaFp128 next_scalar = akita_mul(scalar, bound);
    AkitaFp128 next_at_one = akita_mul(next_scalar, tau_next);
    next[0].l_at_0 = akita_sub(next_scalar, next_at_one);
    next[0].l_at_1 = next_at_one;
    next[0].binary_batching = current.binary_batching;
}

inline void akita_direct_relation_bind_additional_pair(
    DirectRelationAdditionalPair pair,
    AkitaFp128 challenge,
    thread AkitaFp128 &linear,
    thread AkitaFp128 &binary)
{
    linear = akita_add(
        pair.linear[0],
        akita_mul(challenge, akita_sub(pair.linear[1], pair.linear[0])));
    binary = akita_add(
        pair.binary[0],
        akita_mul(challenge, akita_sub(pair.binary[1], pair.binary[0])));
}

kernel void akita_fp128_direct_relation_additional_fold(
    device const DirectRelationAdditionalPair *input [[buffer(0)]],
    device DirectRelationAdditionalPair *output [[buffer(1)]],
    device const DirectRelationAdditionalFoldMapping *mappings [[buffer(2)]],
    constant AkitaFp128 &challenge [[buffer(3)]],
    constant ulong &mapping_count [[buffer(4)]],
    uint thread_index [[thread_position_in_grid]])
{
    ulong index = (ulong)thread_index;
    if (index >= mapping_count) {
        return;
    }
    DirectRelationAdditionalFoldMapping mapping = mappings[index];
    DirectRelationAdditionalPair result;
    result.parent = mapping.parent;
    result.reserved = 0ul;
    result.linear[0] = akita_zero();
    result.linear[1] = akita_zero();
    result.binary[0] = akita_zero();
    result.binary[1] = akita_zero();
    if (mapping.left != 0xffffffffu) {
        akita_direct_relation_bind_additional_pair(
            input[mapping.left], challenge, result.linear[0], result.binary[0]);
    }
    if (mapping.right != 0xffffffffu) {
        akita_direct_relation_bind_additional_pair(
            input[mapping.right], challenge, result.linear[1], result.binary[1]);
    }
    output[index] = result;
}

inline void akita_emit_reduced_shift_sequence(
    threadgroup const AkitaFp128 *coefficients,
    device const AkitaFp128 *alpha_powers,
    device AkitaFp128 *output,
    ulong output_start,
    AkitaFp128 initial_evaluation,
    constant DirectRelationReducedSourceParams &params,
    uint thread_index)
{
    if (thread_index >= 32u) {
        return;
    }
    uint lane = thread_index;
    ulong chunk = params.ring_dimension / 32ul;
    ulong shift_start = (ulong)lane * chunk;
    AkitaFp128 multiplier = alpha_powers[chunk];
    AkitaFp128 bias = akita_zero();
    for (ulong local = 0ul; local < chunk; ++local) {
        ulong coefficient = params.ring_dimension - 1ul - shift_start - local;
        bias = akita_sub(
            akita_mul(params.alpha, bias),
            akita_mul(params.wrap_correction, coefficients[coefficient]));
    }

    AkitaFp128 prefix_multiplier = multiplier;
    AkitaFp128 prefix_bias = bias;
    for (uint offset = 1u; offset < 32u; offset <<= 1u) {
        uint source_lane = lane >= offset ? lane - offset : 0u;
        AkitaFp128 previous_multiplier =
            akita_simd_shuffle_fp128(prefix_multiplier, source_lane);
        AkitaFp128 previous_bias = akita_simd_shuffle_fp128(prefix_bias, source_lane);
        if (lane >= offset) {
            prefix_bias = akita_add(
                akita_mul(prefix_multiplier, previous_bias), prefix_bias);
            prefix_multiplier = akita_mul(prefix_multiplier, previous_multiplier);
        }
    }

    AkitaFp128 evaluation = initial_evaluation;
    uint previous_lane = lane == 0u ? 0u : lane - 1u;
    AkitaFp128 previous_multiplier =
        akita_simd_shuffle_fp128(prefix_multiplier, previous_lane);
    AkitaFp128 previous_bias = akita_simd_shuffle_fp128(prefix_bias, previous_lane);
    if (lane != 0u) {
        evaluation = akita_add(
            akita_mul(previous_multiplier, initial_evaluation), previous_bias);
    }
    for (ulong local = 0ul; local < chunk; ++local) {
        ulong shift = shift_start + local;
        output[output_start + shift] = evaluation;
        AkitaFp128 wrapped = akita_mul(
            params.wrap_correction,
            coefficients[params.ring_dimension - 1ul - shift]);
        evaluation = akita_sub(akita_mul(params.alpha, evaluation), wrapped);
    }
}

kernel void akita_fp128_direct_relation_setup_source(
    device const AkitaFp128 *matrix [[buffer(0)]],
    device const AkitaFp128 *row_weights [[buffer(1)]],
    device const AkitaFp128 *alpha_powers [[buffer(2)]],
    device AkitaFp128 *output [[buffer(3)]],
    constant DirectRelationReducedSourceParams &params [[buffer(4)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup AkitaFp128 coefficients[512];
    threadgroup AkitaFp128 reduction[256];
    ulong column = (ulong)threadgroup_index.x;
    for (ulong coefficient = (ulong)thread_index;
         coefficient < params.ring_dimension;
         coefficient += 256ul) {
        AkitaFp128 combined = akita_zero();
        for (ulong row = 0ul; row < params.row_count; ++row) {
            ulong index = ((row * params.item_count + column) * params.ring_dimension)
                + coefficient;
            combined = akita_add(combined, akita_mul(row_weights[row], matrix[index]));
        }
        coefficients[coefficient] = combined;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    AkitaFp128 sum = akita_zero();
    for (ulong coefficient = (ulong)thread_index;
         coefficient < params.ring_dimension;
         coefficient += 256ul) {
        sum = akita_add(sum, akita_mul(coefficients[coefficient], alpha_powers[coefficient]));
    }
    reduction[thread_index] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 128u; width != 0u; width >>= 1u) {
        if (thread_index < width) {
            reduction[thread_index] = akita_add(
                reduction[thread_index], reduction[thread_index + width]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    akita_emit_reduced_shift_sequence(
        coefficients,
        alpha_powers,
        output,
        column * params.ring_dimension,
        reduction[0],
        params,
        thread_index);
}

kernel void akita_fp128_direct_relation_sparse_source(
    device const uint *term_offsets [[buffer(0)]],
    device const uint *positions [[buffer(1)]],
    device const char *sparse_coefficients [[buffer(2)]],
    device const AkitaFp128 *alpha_powers [[buffer(3)]],
    device AkitaFp128 *output [[buffer(4)]],
    constant DirectRelationReducedSourceParams &params [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup AkitaFp128 coefficients[512];
    threadgroup AkitaFp128 reduction[256];
    ulong challenge = (ulong)threadgroup_index.x;
    for (ulong coefficient = (ulong)thread_index;
         coefficient < params.ring_dimension;
         coefficient += 256ul) {
        coefficients[coefficient] = akita_zero();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint start = term_offsets[challenge];
    uint end = term_offsets[challenge + 1ul];
    for (uint term = start + thread_index; term < end; term += 256u) {
        uint position = positions[term];
        coefficients[position] = akita_mul_signed_small(
            akita_from_u32(1u), (long)sparse_coefficients[term]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    AkitaFp128 sum = akita_zero();
    for (ulong coefficient = (ulong)thread_index;
         coefficient < params.ring_dimension;
         coefficient += 256ul) {
        sum = akita_add(sum, akita_mul(coefficients[coefficient], alpha_powers[coefficient]));
    }
    reduction[thread_index] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 128u; width != 0u; width >>= 1u) {
        if (thread_index < width) {
            reduction[thread_index] = akita_add(
                reduction[thread_index], reduction[thread_index + width]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    akita_emit_reduced_shift_sequence(
        coefficients,
        alpha_powers,
        output,
        challenge * params.ring_dimension,
        reduction[0],
        params,
        thread_index);
}

inline AkitaFp128 akita_direct_relation_p_value(
    ulong index,
    device const AkitaFp128 *alpha,
    device const AkitaFp128 *lane_weights,
    device const AkitaFp128 *linear_values,
    device const uint *source_offsets,
    device const uint *lane_offsets,
    device const uint *lane_segments,
    device const DirectRelationLinearSegment *segments,
    constant DirectRelationParams &params)
{
    ulong lane = index / params.current_coeff_count;
    ulong coefficient = index - lane * params.current_coeff_count;
    AkitaFp128 ordinary = akita_mul(alpha[coefficient], lane_weights[lane]);
    return akita_add(
        ordinary,
        akita_direct_relation_linear_value(
            index,
            linear_values,
            source_offsets,
            lane_offsets,
            lane_segments,
            segments,
            params));
}

inline void akita_direct_relation_coefficients(
    AkitaFp128 left,
    AkitaFp128 right,
    ulong pair,
    device const AkitaFp128 *e_first,
    device const AkitaFp128 *e_second,
    device const AkitaFp128 *alpha,
    device const AkitaFp128 *lane_weights,
    device const AkitaFp128 *linear_values,
    device const uint *source_offsets,
    device const uint *lane_offsets,
    device const uint *lane_segments,
    device const DirectRelationLinearSegment *segments,
    constant DirectRelationScalars &scalars,
    constant DirectRelationParams &params,
    thread AkitaFp128 &c0,
    thread AkitaFp128 &c2,
    thread AkitaFp128 &c3)
{
    AkitaFp128 delta = akita_sub(right, left);
    AkitaFp128 delta_squared = akita_mul(delta, delta);
    AkitaFp128 q0 = akita_mul(left, akita_add(left, akita_from_u32(1u)));
    AkitaFp128 q1 = akita_mul(
        delta,
        akita_add(akita_add(left, left), akita_from_u32(1u)));
    AkitaFp128 l_delta = akita_sub(scalars.l_at_1, scalars.l_at_0);
    ulong equality_low = pair & (params.num_first - 1ul);
    ulong equality_high = pair / params.num_first;
    AkitaFp128 equality = akita_mul(
        e_first[equality_low], e_second[equality_high]);
    AkitaFp128 p0 = akita_direct_relation_p_value(
        2ul * pair,
        alpha,
        lane_weights,
        linear_values,
        source_offsets,
        lane_offsets,
        lane_segments,
        segments,
        params);
    AkitaFp128 p1 = akita_direct_relation_p_value(
        2ul * pair + 1ul,
        alpha,
        lane_weights,
        linear_values,
        source_offsets,
        lane_offsets,
        lane_segments,
        segments,
        params);

    AkitaFp128 virtual_0 = akita_mul(
        equality,
        akita_mul(scalars.l_at_0, q0));
    AkitaFp128 virtual_2 = akita_mul(
        equality,
        akita_add(
            akita_mul(scalars.l_at_0, delta_squared),
            akita_mul(l_delta, q1)));
    AkitaFp128 virtual_3 = akita_mul(
        equality,
        akita_mul(l_delta, delta_squared));
    c0 = akita_add(c0, akita_add(virtual_0, akita_mul(left, p0)));
    c2 = akita_add(
        c2,
        akita_add(virtual_2, akita_mul(delta, akita_sub(p1, p0))));
    c3 = akita_add(c3, virtual_3);
}

inline void akita_direct_relation_store_partial(
    threadgroup AkitaFp128 *reduction_0,
    threadgroup AkitaFp128 *reduction_2,
    threadgroup AkitaFp128 *reduction_3,
    threadgroup AkitaFp128 *reduction_4,
    AkitaFp128 sum_0,
    AkitaFp128 sum_2,
    AkitaFp128 sum_3,
    uint thread_index,
    uint threadgroup_index,
    device AkitaFp128 *partials)
{
    reduction_0[thread_index] = sum_0;
    reduction_2[thread_index] = sum_2;
    reduction_3[thread_index] = sum_3;
    reduction_4[thread_index] = akita_zero();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 128u; width != 0u; width >>= 1u) {
        if (thread_index < width) {
            reduction_0[thread_index] = akita_add(
                reduction_0[thread_index], reduction_0[thread_index + width]);
            reduction_2[thread_index] = akita_add(
                reduction_2[thread_index], reduction_2[thread_index + width]);
            reduction_3[thread_index] = akita_add(
                reduction_3[thread_index], reduction_3[thread_index + width]);
            reduction_4[thread_index] = akita_add(
                reduction_4[thread_index], reduction_4[thread_index + width]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index == 0u) {
        ulong output = (ulong)threadgroup_index * 4ul;
        partials[output] = reduction_0[0];
        partials[output + 1ul] = reduction_2[0];
        partials[output + 2ul] = reduction_3[0];
        partials[output + 3ul] = reduction_4[0];
    }
}

kernel void akita_fp128_direct_relation_initial_partials(
    device const char *digits [[buffer(0)]],
    device const AkitaFp128 *e_first [[buffer(1)]],
    device const AkitaFp128 *e_second [[buffer(2)]],
    device const AkitaFp128 *alpha [[buffer(3)]],
    device const AkitaFp128 *lane_weights [[buffer(4)]],
    device const AkitaFp128 *linear_values [[buffer(5)]],
    device const uint *source_offsets [[buffer(6)]],
    device const uint *lane_offsets [[buffer(7)]],
    device const uint *lane_segments [[buffer(8)]],
    device const DirectRelationLinearSegment *segments [[buffer(9)]],
    device AkitaFp128 *partials [[buffer(10)]],
    device const AkitaFp128 *prefix_weights [[buffer(11)]],
    constant DirectRelationScalars &scalars [[buffer(12)]],
    constant DirectRelationParams &params [[buffer(13)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup AkitaFp128 reduction_0[256];
    threadgroup AkitaFp128 reduction_2[256];
    threadgroup AkitaFp128 reduction_3[256];
    threadgroup AkitaFp128 reduction_4[256];
    AkitaFp128 sum_0 = akita_zero();
    AkitaFp128 sum_2 = akita_zero();
    AkitaFp128 sum_3 = akita_zero();
    ulong global_thread = (ulong)threadgroup_index.x * 256ul + (ulong)thread_index;
    ulong stride = params.workgroups * 256ul;
    for (ulong pair = global_thread; pair < params.pair_count; pair += stride) {
        AkitaFp128 left = akita_direct_relation_compact_value(
            digits, prefix_weights, 2ul * pair, params);
        AkitaFp128 right = akita_direct_relation_compact_value(
            digits, prefix_weights, 2ul * pair + 1ul, params);
        akita_direct_relation_coefficients(
            left, right, pair, e_first, e_second, alpha, lane_weights,
            linear_values, source_offsets, lane_offsets, lane_segments, segments,
            scalars, params, sum_0, sum_2, sum_3);
    }
    akita_direct_relation_store_partial(
        reduction_0, reduction_2, reduction_3, reduction_4,
        sum_0, sum_2, sum_3, thread_index, threadgroup_index.x, partials);
}

kernel void akita_fp128_direct_relation_compact_fold_partials(
    device const char *digits [[buffer(0)]],
    device AkitaFp128 *folded [[buffer(1)]],
    device const AkitaFp128 *e_first [[buffer(2)]],
    device const AkitaFp128 *e_second [[buffer(3)]],
    device const AkitaFp128 *alpha [[buffer(4)]],
    device const AkitaFp128 *lane_weights [[buffer(5)]],
    device const AkitaFp128 *linear_values [[buffer(6)]],
    device const uint *source_offsets [[buffer(7)]],
    device const uint *lane_offsets [[buffer(8)]],
    device const uint *lane_segments [[buffer(9)]],
    device const DirectRelationLinearSegment *segments [[buffer(10)]],
    device AkitaFp128 *partials [[buffer(11)]],
    device const AkitaFp128 *prefix_weights [[buffer(12)]],
    constant DirectRelationScalars &scalars [[buffer(13)]],
    constant DirectRelationParams &params [[buffer(14)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup AkitaFp128 reduction_0[256];
    threadgroup AkitaFp128 reduction_2[256];
    threadgroup AkitaFp128 reduction_3[256];
    threadgroup AkitaFp128 reduction_4[256];
    AkitaFp128 sum_0 = akita_zero();
    AkitaFp128 sum_2 = akita_zero();
    AkitaFp128 sum_3 = akita_zero();
    ulong global_thread = (ulong)threadgroup_index.x * 256ul + (ulong)thread_index;
    ulong stride = params.workgroups * 256ul;
    for (ulong pair = global_thread; pair < params.pair_count; pair += stride) {
        ulong left_index = 2ul * pair;
        AkitaFp128 left = left_index < params.current_live_len
            ? akita_direct_relation_compact_value(
                digits, prefix_weights, left_index, params)
            : akita_zero();
        AkitaFp128 right = left_index + 1ul < params.current_live_len
            ? akita_direct_relation_compact_value(
                digits, prefix_weights, left_index + 1ul, params)
            : akita_zero();
        if (params.materialize_prefix != 0ul) {
            if (left_index < params.current_live_len) {
                folded[left_index] = left;
            }
            if (left_index + 1ul < params.current_live_len) {
                folded[left_index + 1ul] = right;
            }
        }
        akita_direct_relation_coefficients(
            left, right, pair, e_first, e_second, alpha, lane_weights,
            linear_values, source_offsets, lane_offsets, lane_segments, segments,
            scalars, params, sum_0, sum_2, sum_3);
    }
    akita_direct_relation_store_partial(
        reduction_0, reduction_2, reduction_3, reduction_4,
        sum_0, sum_2, sum_3, thread_index, threadgroup_index.x, partials);
}

kernel void akita_fp128_direct_relation_field_fold_partials(
    device const AkitaFp128 *input [[buffer(0)]],
    device AkitaFp128 *folded [[buffer(1)]],
    device const AkitaFp128 *e_first [[buffer(2)]],
    device const AkitaFp128 *e_second [[buffer(3)]],
    device const AkitaFp128 *alpha [[buffer(4)]],
    device AkitaFp128 *lane_weights [[buffer(5)]],
    device const AkitaFp128 *linear_values [[buffer(6)]],
    device const uint *source_offsets [[buffer(7)]],
    device const uint *lane_offsets [[buffer(8)]],
    device const uint *lane_segments [[buffer(9)]],
    device const DirectRelationLinearSegment *segments [[buffer(10)]],
    device AkitaFp128 *partials [[buffer(11)]],
    constant AkitaFp128 &challenge [[buffer(12)]],
    constant DirectRelationScalars &scalars [[buffer(13)]],
    constant DirectRelationParams &params [[buffer(14)]],
    device const AkitaFp128 *lane_weights_input [[buffer(15)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup AkitaFp128 reduction_0[256];
    threadgroup AkitaFp128 reduction_2[256];
    threadgroup AkitaFp128 reduction_3[256];
    threadgroup AkitaFp128 reduction_4[256];
    AkitaFp128 sum_0 = akita_zero();
    AkitaFp128 sum_2 = akita_zero();
    AkitaFp128 sum_3 = akita_zero();
    ulong global_thread = (ulong)threadgroup_index.x * 256ul + (ulong)thread_index;
    ulong stride = params.workgroups * 256ul;
    for (ulong pair = global_thread; pair < params.pair_count; pair += stride) {
        AkitaFp128 values[2];
        for (uint side = 0u; side < 2u; ++side) {
            ulong output_index = 2ul * pair + (ulong)side;
            AkitaFp128 value = akita_zero();
            if (output_index < params.current_live_len) {
                ulong input_index = 2ul * output_index;
                AkitaFp128 left = input_index < params.input_live_len
                    ? input[input_index]
                    : akita_zero();
                AkitaFp128 right = input_index + 1ul < params.input_live_len
                    ? input[input_index + 1ul]
                    : akita_zero();
                value = akita_add(
                    left, akita_mul(challenge, akita_sub(right, left)));
                folded[output_index] = value;
            }
            values[side] = value;
            if (params.fold_lane_weights != 0ul) {
                ulong lane_input_index = 2ul * output_index;
                AkitaFp128 lane_left = lane_weights_input[lane_input_index];
                lane_weights[output_index] = akita_add(
                    lane_left,
                    akita_mul(
                        challenge,
                        akita_sub(lane_weights_input[lane_input_index + 1ul], lane_left)));
            }
        }
        akita_direct_relation_coefficients(
            values[0], values[1], pair, e_first, e_second, alpha, lane_weights,
            linear_values, source_offsets, lane_offsets, lane_segments, segments,
            scalars, params, sum_0, sum_2, sum_3);
    }
    akita_direct_relation_store_partial(
        reduction_0, reduction_2, reduction_3, reduction_4,
        sum_0, sum_2, sum_3, thread_index, threadgroup_index.x, partials);
}

inline void akita_direct_relation_additional_coefficients(
    AkitaFp128 left,
    AkitaFp128 right,
    DirectRelationAdditionalPair pair,
    constant DirectRelationScalars &scalars,
    thread AkitaFp128 &c0,
    thread AkitaFp128 &c2,
    thread AkitaFp128 &c3)
{
    AkitaFp128 delta = akita_sub(right, left);
    AkitaFp128 linear_delta = akita_sub(pair.linear[1], pair.linear[0]);
    AkitaFp128 binary_delta = akita_sub(pair.binary[1], pair.binary[0]);
    AkitaFp128 square_0 = akita_mul(left, akita_add(left, akita_from_u32(1u)));
    AkitaFp128 square_1 = akita_mul(
        delta, akita_add(akita_add(left, left), akita_from_u32(1u)));
    AkitaFp128 square_2 = akita_mul(delta, delta);
    AkitaFp128 binary_0 = akita_mul(scalars.binary_batching, pair.binary[0]);
    AkitaFp128 binary_delta_batched = akita_mul(scalars.binary_batching, binary_delta);
    c0 = akita_add(c0, akita_add(
        akita_mul(left, pair.linear[0]),
        akita_mul(binary_0, square_0)));
    c2 = akita_add(c2, akita_add(
        akita_mul(delta, linear_delta),
        akita_add(
            akita_mul(binary_0, square_2),
            akita_mul(binary_delta_batched, square_1))));
    c3 = akita_add(c3, akita_mul(binary_delta_batched, square_2));
}

kernel void akita_fp128_direct_relation_additional_compact_partials(
    device const char *digits [[buffer(0)]],
    device const AkitaFp128 *prefix_weights [[buffer(1)]],
    device const DirectRelationAdditionalPair *pairs [[buffer(2)]],
    device AkitaFp128 *partials [[buffer(3)]],
    constant DirectRelationScalars &scalars [[buffer(4)]],
    constant DirectRelationParams &params [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup AkitaFp128 reduction_0[256];
    threadgroup AkitaFp128 reduction_2[256];
    threadgroup AkitaFp128 reduction_3[256];
    threadgroup AkitaFp128 reduction_4[256];
    AkitaFp128 sum_0 = akita_zero();
    AkitaFp128 sum_2 = akita_zero();
    AkitaFp128 sum_3 = akita_zero();
    ulong global_thread = (ulong)threadgroup_index.x * 256ul + (ulong)thread_index;
    ulong stride = params.additional_workgroups * 256ul;
    for (ulong index = global_thread; index < params.additional_pair_count; index += stride) {
        DirectRelationAdditionalPair pair = pairs[index];
        AkitaFp128 left = akita_direct_relation_compact_value(
            digits, prefix_weights, 2ul * pair.parent, params);
        AkitaFp128 right = akita_direct_relation_compact_value(
            digits, prefix_weights, 2ul * pair.parent + 1ul, params);
        akita_direct_relation_additional_coefficients(
            left, right, pair, scalars, sum_0, sum_2, sum_3);
    }
    akita_direct_relation_store_partial(
        reduction_0, reduction_2, reduction_3, reduction_4,
        sum_0, sum_2, sum_3, thread_index, threadgroup_index.x, partials);
}

kernel void akita_fp128_direct_relation_additional_field_partials(
    device const AkitaFp128 *input [[buffer(0)]],
    device const DirectRelationAdditionalPair *pairs [[buffer(1)]],
    device AkitaFp128 *partials [[buffer(2)]],
    constant DirectRelationScalars &scalars [[buffer(3)]],
    constant DirectRelationParams &params [[buffer(4)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup AkitaFp128 reduction_0[256];
    threadgroup AkitaFp128 reduction_2[256];
    threadgroup AkitaFp128 reduction_3[256];
    threadgroup AkitaFp128 reduction_4[256];
    AkitaFp128 sum_0 = akita_zero();
    AkitaFp128 sum_2 = akita_zero();
    AkitaFp128 sum_3 = akita_zero();
    ulong global_thread = (ulong)threadgroup_index.x * 256ul + (ulong)thread_index;
    ulong stride = params.additional_workgroups * 256ul;
    for (ulong index = global_thread; index < params.additional_pair_count; index += stride) {
        DirectRelationAdditionalPair pair = pairs[index];
        ulong left_index = 2ul * pair.parent;
        AkitaFp128 left = left_index < params.current_live_len
            ? input[left_index]
            : akita_zero();
        AkitaFp128 right = left_index + 1ul < params.current_live_len
            ? input[left_index + 1ul]
            : akita_zero();
        akita_direct_relation_additional_coefficients(
            left, right, pair, scalars, sum_0, sum_2, sum_3);
    }
    akita_direct_relation_store_partial(
        reduction_0, reduction_2, reduction_3, reduction_4,
        sum_0, sum_2, sum_3, thread_index, threadgroup_index.x, partials);
}

inline uint4 akita_fp128_d512_gather_word(
    threadgroup const uint *matrix,
    uint word,
    uint matrix_base,
    uint4 sources)
{
    uint plane_base = word * PACKED_FP128_D512_PANEL_TILE_ELEMENTS;
    return uint4(
        matrix[plane_base + matrix_base + sources[0]],
        matrix[plane_base + matrix_base + sources[1]],
        matrix[plane_base + matrix_base + sources[2]],
        matrix[plane_base + matrix_base + sources[3]]);
}

inline void akita_fp128_d512_accumulate_value(
    thread AkitaTransposedFp128Accumulator &accumulator,
    uint4 value_0,
    uint4 value_1,
    uint4 value_2,
    uint4 value_3,
    bool4 positive)
{
    uint4 carry = select(uint4(1u), uint4(0u), positive);
    accumulator.word_0 = akita_add_transposed_word(
        accumulator.word_0, select(~value_0, value_0, positive), carry);
    accumulator.word_1 = akita_add_transposed_word(
        accumulator.word_1, select(~value_1, value_1, positive), carry);
    accumulator.word_2 = akita_add_transposed_word(
        accumulator.word_2, select(~value_2, value_2, positive), carry);
    accumulator.word_3 = akita_add_transposed_word(
        accumulator.word_3, select(~value_3, value_3, positive), carry);
    int4 carry_words = int4(carry);
    accumulator.wraps += select(carry_words - int4(1), carry_words, positive);
}

inline void akita_fp128_d512_accumulate_positive(
    thread AkitaTransposedFp128Accumulator &accumulator,
    threadgroup const uint *matrix,
    uint matrix_base,
    uint4 sources)
{
    uint4 carry = uint4(0u);
    accumulator.word_0 = akita_add_transposed_word(
        accumulator.word_0,
        akita_fp128_d512_gather_word(matrix, 0u, matrix_base, sources), carry);
    accumulator.word_1 = akita_add_transposed_word(
        accumulator.word_1,
        akita_fp128_d512_gather_word(matrix, 1u, matrix_base, sources), carry);
    accumulator.word_2 = akita_add_transposed_word(
        accumulator.word_2,
        akita_fp128_d512_gather_word(matrix, 2u, matrix_base, sources), carry);
    accumulator.word_3 = akita_add_transposed_word(
        accumulator.word_3,
        akita_fp128_d512_gather_word(matrix, 3u, matrix_base, sources), carry);
    accumulator.wraps += int4(carry);
}

inline void akita_fp128_d512_accumulate_negative(
    thread AkitaTransposedFp128Accumulator &accumulator,
    threadgroup const uint *matrix,
    uint matrix_base,
    uint4 sources)
{
    uint4 carry = uint4(1u);
    accumulator.word_0 = akita_add_transposed_word(
        accumulator.word_0,
        ~akita_fp128_d512_gather_word(matrix, 0u, matrix_base, sources), carry);
    accumulator.word_1 = akita_add_transposed_word(
        accumulator.word_1,
        ~akita_fp128_d512_gather_word(matrix, 1u, matrix_base, sources), carry);
    accumulator.word_2 = akita_add_transposed_word(
        accumulator.word_2,
        ~akita_fp128_d512_gather_word(matrix, 2u, matrix_base, sources), carry);
    accumulator.word_3 = akita_add_transposed_word(
        accumulator.word_3,
        ~akita_fp128_d512_gather_word(matrix, 3u, matrix_base, sources), carry);
    accumulator.wraps += int4(carry) - int4(1);
}

inline void akita_fp128_d512_accumulate_mixed(
    thread AkitaTransposedFp128Accumulator &accumulator,
    threadgroup const uint *matrix,
    uint matrix_base,
    uint4 sources,
    bool4 positive)
{
    akita_fp128_d512_accumulate_value(
        accumulator,
        akita_fp128_d512_gather_word(matrix, 0u, matrix_base, sources),
        akita_fp128_d512_gather_word(matrix, 1u, matrix_base, sources),
        akita_fp128_d512_gather_word(matrix, 2u, matrix_base, sources),
        akita_fp128_d512_gather_word(matrix, 3u, matrix_base, sources),
        positive);
}

inline void akita_fp128_d512_accumulate_pair(
    thread AkitaTransposedFp128Accumulator &accumulator_0,
    thread AkitaTransposedFp128Accumulator &accumulator_1,
    threadgroup const uint *matrix,
    uint simd_lane,
    uint coefficient_band,
    uint local_position,
    uint local_shift,
    bool odd_row)
{
    uint coefficient_base = coefficient_band * 256u;
    uint4 coefficients_0 = uint4(
        simd_lane + coefficient_base,
        simd_lane + coefficient_base + 32u,
        simd_lane + coefficient_base + 64u,
        simd_lane + coefficient_base + 96u);
    uint4 coefficients_1 = coefficients_0 + uint4(128u);
    uint matrix_base = local_position * 512u;
    if (coefficient_band == 0u) {
        if (odd_row) {
            uint4 shift = uint4(256u - local_shift);
            akita_fp128_d512_accumulate_negative(
                accumulator_0, matrix, matrix_base, coefficients_0 + shift);
            akita_fp128_d512_accumulate_negative(
                accumulator_1, matrix, matrix_base, coefficients_1 + shift);
        } else {
            uint4 shift = uint4(local_shift);
            akita_fp128_d512_accumulate_mixed(
                accumulator_0, matrix, matrix_base,
                (coefficients_0 - shift) & uint4(511u), coefficients_0 >= shift);
            akita_fp128_d512_accumulate_mixed(
                accumulator_1, matrix, matrix_base,
                (coefficients_1 - shift) & uint4(511u), coefficients_1 >= shift);
        }
    } else if (odd_row) {
        uint4 shift = uint4(256u + local_shift);
        akita_fp128_d512_accumulate_mixed(
            accumulator_0, matrix, matrix_base,
            (coefficients_0 - shift) & uint4(511u), coefficients_0 >= shift);
        akita_fp128_d512_accumulate_mixed(
            accumulator_1, matrix, matrix_base,
            (coefficients_1 - shift) & uint4(511u), coefficients_1 >= shift);
    } else {
        uint4 shift = uint4(local_shift);
        akita_fp128_d512_accumulate_positive(
            accumulator_0, matrix, matrix_base, coefficients_0 - shift);
        akita_fp128_d512_accumulate_positive(
            accumulator_1, matrix, matrix_base, coefficients_1 - shift);
    }
}

inline void akita_fp128_d512_accumulate_shift(
    thread AkitaTransposedFp128Accumulator &accumulator_0,
    thread AkitaTransposedFp128Accumulator &accumulator_1,
    threadgroup const uint *matrix,
    uint simd_lane,
    uint coefficient_band,
    uint local_position,
    uint local_shift)
{
    uint coefficient_base = coefficient_band * 256u;
    uint4 coefficients_0 = uint4(
        simd_lane + coefficient_base,
        simd_lane + coefficient_base + 32u,
        simd_lane + coefficient_base + 64u,
        simd_lane + coefficient_base + 96u);
    uint4 coefficients_1 = coefficients_0 + uint4(128u);
    uint4 shift = uint4(local_shift);
    uint matrix_base = local_position * 512u;
    akita_fp128_d512_accumulate_mixed(
        accumulator_0, matrix, matrix_base,
        (coefficients_0 - shift) & uint4(511u), coefficients_0 >= shift);
    akita_fp128_d512_accumulate_mixed(
        accumulator_1, matrix, matrix_base,
        (coefficients_1 - shift) & uint4(511u), coefficients_1 >= shift);
}

inline void akita_store_fp128_d512_group(
    device AkitaFp128 *partials,
    AkitaTransposedFp128Accumulator accumulator,
    uint coefficient_group,
    uint column,
    uint block_in_column,
    uint blocks_per_column,
    uint n_a,
    uint a_row,
    uint position_partial,
    uint output_coefficients,
    uint simd_lane)
{
    uint block = column * blocks_per_column + block_in_column;
    uint output_base = (block * n_a + a_row) * 512u + coefficient_group * 128u;
    uint partial_base = position_partial * output_coefficients + output_base;
    partials[partial_base + simd_lane] =
        akita_reduce_transposed_fp128(accumulator, 0u);
    partials[partial_base + simd_lane + 32u] =
        akita_reduce_transposed_fp128(accumulator, 1u);
    partials[partial_base + simd_lane + 64u] =
        akita_reduce_transposed_fp128(accumulator, 2u);
    partials[partial_base + simd_lane + 96u] =
        akita_reduce_transposed_fp128(accumulator, 3u);
}

kernel void akita_packed_onehot_commit_fp128_d512_panels(
    device const AkitaFp128 *matrix [[buffer(0)]],
    device const uchar *lanes [[buffer(1)]],
    device AkitaFp128 *partials [[buffer(2)]],
    constant PackedOneHotCommitParams &params [[buffer(3)]],
    device const ulong *active_zero_rows [[buffer(4)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup uint shared_matrix[PACKED_FP128_D512_PANEL_TILE_ELEMENTS * 4];

    constexpr uint tasks_per_stream = 32u;
    constexpr uint threads_per_threadgroup = 1024u;
    constexpr uint positions_per_tile = 4u;
    uint live_columns = (uint)params.num_columns;
    uint onehot_k = (uint)params.onehot_k;
    uint rows_per_position = 512u / onehot_k;
    uint rows_per_tile = positions_per_tile * rows_per_position;
    uint num_tasks = (uint)params.dispatch_tasks;
    uint blocks_per_column = (uint)params.blocks_per_column;
    uint streams = (num_tasks + tasks_per_stream - 1u) / tasks_per_stream;
    uint simd_lane = thread_index & 31u;
    uint simdgroup = thread_index >> 5u;
    uint position_partials = (uint)params.position_partials_per_block;
    uint groups_per_band = streams * position_partials * (uint)params.n_a;
    uint coefficient_band = threadgroup_index.x / groups_per_band;
    uint group = threadgroup_index.x % groups_per_band;
    uint stream = group % streams;
    uint partial_group = group / streams;
    uint position_partial = partial_group % position_partials;
    uint a_row = partial_group / position_partials;
    uint positions_per_partial = (uint)params.positions_per_partial;
    uint partial_start = position_partial * positions_per_partial;
    uint rows_per_partial = positions_per_partial * rows_per_position;
    uint rows_per_block = (uint)params.positions_per_block * rows_per_position;
    uint output_coefficients = (uint)params.output_coefficients;
    uint dispatch_task = stream * tasks_per_stream + simdgroup;
    bool simdgroup_active = dispatch_task < num_tasks;
    uint global_task = (uint)params.task_offset + dispatch_task;
    uint task_block = global_task / live_columns;
    uint task_column = global_task % live_columns;
    ulong matrix_cursor =
        ((ulong)a_row * params.positions_per_block + (ulong)partial_start) * 512ul;

    AkitaTransposedFp128Accumulator accumulator_0 = akita_transposed_fp128_zero();
    AkitaTransposedFp128Accumulator accumulator_1 = akita_transposed_fp128_zero();
    uint coefficient_group_0 = coefficient_band * 2u;
    uint coefficient_group_1 = coefficient_group_0 + 1u;

    uint tile_count = positions_per_partial / positions_per_tile;
    for (uint tile = 0u; tile < tile_count; ++tile) {
        for (uint shared_index = thread_index;
             shared_index < PACKED_FP128_D512_PANEL_TILE_ELEMENTS;
             shared_index += threads_per_threadgroup) {
            AkitaFp128 value = matrix[matrix_cursor + (ulong)shared_index];
            shared_matrix[shared_index] = value.limb[0];
            shared_matrix[PACKED_FP128_D512_PANEL_TILE_ELEMENTS + shared_index] =
                value.limb[1];
            shared_matrix[PACKED_FP128_D512_PANEL_TILE_ELEMENTS * 2u + shared_index] =
                value.limb[2];
            shared_matrix[PACKED_FP128_D512_PANEL_TILE_ELEMENTS * 3u + shared_index] =
                value.limb[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (onehot_k == 256u) {
            uint local_hot = 0u;
            bool local_selected = false;
            if (simdgroup_active && simd_lane < rows_per_tile) {
                ulong trace_row = (ulong)task_block * (ulong)rows_per_block
                    + (ulong)position_partial * (ulong)rows_per_partial
                    + (ulong)tile * (ulong)rows_per_tile
                    + (ulong)simd_lane;
                local_hot = (uint)lanes[
                    (trace_row - params.lane_row_offset) * params.lane_stride
                        + (ulong)task_column];
                local_selected = local_hot != 0u;
                if (!local_selected
                    && ((params.zero_column_mask >> task_column) & 1ul) != 0ul) {
                    ulong active_word = active_zero_rows[trace_row >> 6ul];
                    local_selected = ((active_word >> (trace_row & 63ul)) & 1ul) != 0ul;
                }
            }
            uint selected = uint(
                simd_ballot(local_selected).operator unsigned long());
            while (selected != 0u) {
                uint selected_lane = ctz(selected);
                uint selected_hot = simd_shuffle(local_hot, selected_lane);
                uint local_position = selected_lane >> 1u;
                bool odd_row = (selected_lane & 1u) != 0u;
                akita_fp128_d512_accumulate_pair(
                    accumulator_0, accumulator_1, shared_matrix, simd_lane,
                    coefficient_band, local_position, selected_hot, odd_row);
                selected &= selected - 1u;
            }
        } else {
            for (uint local_position = 0u;
                 local_position < positions_per_tile;
                 ++local_position) {
                ulong trace_row = (ulong)task_block * (ulong)rows_per_block
                    + (ulong)position_partial * (ulong)rows_per_partial
                    + (ulong)tile * (ulong)rows_per_tile
                    + (ulong)local_position * (ulong)rows_per_position
                    + (ulong)simd_lane;
                uint local_hot = 0u;
                bool local_selected = false;
                if (simdgroup_active) {
                    local_hot = (uint)lanes[
                        (trace_row - params.lane_row_offset) * params.lane_stride
                            + (ulong)task_column];
                    local_selected = local_hot != 0u;
                    if (!local_selected
                        && ((params.zero_column_mask >> task_column) & 1ul) != 0ul) {
                        ulong active_word = active_zero_rows[trace_row >> 6ul];
                        local_selected =
                            ((active_word >> (trace_row & 63ul)) & 1ul) != 0ul;
                    }
                }
                uint selected = uint(
                    simd_ballot(local_selected).operator unsigned long());
                while (selected != 0u) {
                    uint selected_lane = ctz(selected);
                    uint selected_hot = simd_shuffle(local_hot, selected_lane);
                    uint local_shift = selected_lane * onehot_k + selected_hot;
                    akita_fp128_d512_accumulate_shift(
                        accumulator_0, accumulator_1, shared_matrix, simd_lane,
                        coefficient_band, local_position, local_shift);
                    selected &= selected - 1u;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        matrix_cursor += (ulong)PACKED_FP128_D512_PANEL_TILE_ELEMENTS;
    }

    if (simdgroup_active) {
        akita_store_fp128_d512_group(
            partials, accumulator_0, coefficient_group_0, task_column, task_block,
            blocks_per_column, (uint)params.n_a, a_row, position_partial,
            output_coefficients, simd_lane);
        akita_store_fp128_d512_group(
            partials, accumulator_1, coefficient_group_1, task_column, task_block,
            blocks_per_column, (uint)params.n_a, a_row, position_partial,
            output_coefficients, simd_lane);
    }
}

kernel void akita_onehot_commit_block_batched(
    device const AkitaFp128 *matrix [[buffer(0)]],
    device const ushort *hot_indices [[buffer(1)]],
    device AkitaFp128 *output [[buffer(2)]],
    constant OneHotCommitParams &params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    uint ring_d = (uint)params.ring_d;
    uint log_ring_d = (uint)params.log_ring_d;
    uint onehot_k = (uint)params.onehot_k;
    uint log_onehot_k = (uint)params.log_onehot_k;
    uint positions_per_block = (uint)params.positions_per_block;
    uint blocks_per_threadgroup = (uint)params.blocks_per_threadgroup;
    uint num_blocks = (uint)params.num_blocks;
    uint n_a = (uint)params.n_a;
    uint coefficient = thread_index & (ring_d - 1u);
    uint block_lane = thread_index >> log_ring_d;
    uint block_groups =
        (num_blocks + blocks_per_threadgroup - 1u) / blocks_per_threadgroup;
    uint group = threadgroup_index.x;
    uint block_group = group % block_groups;
    uint source_row = group / block_groups;
    uint row = source_row % n_a;
    uint source = source_row / n_a;
    uint block = block_group * blocks_per_threadgroup + block_lane;
    if (source >= (uint)params.num_sources || block >= num_blocks) {
        return;
    }

    uint total_field_elements = (uint)params.total_field_elements;
    uint block_field_start = block * positions_per_block * ring_d;
    uint block_field_end = min(
        block_field_start + positions_per_block * ring_d,
        total_field_elements);
    uint chunk_start = block_field_start >> log_onehot_k;
    uint chunk_end =
        (block_field_end + onehot_k - 1u) >> log_onehot_k;
    uint source_chunk_base = source * (uint)params.chunks_per_source;
    uint active_a_cols = positions_per_block * (uint)params.num_digits_inner;
    uint block_ring_start = block * positions_per_block;

    AkitaFp128 accumulator = akita_zero();
    AkitaWideAccumulator deferred = akita_wide_zero();
    uint deferred_count = 0u;
    bool has_reduced_segment = false;
    for (uint chunk = chunk_start; chunk < chunk_end; ++chunk) {
        uint hot = hot_indices[source_chunk_base + chunk];
        if (hot == NONE_INDEX) {
            continue;
        }
        uint field_position = (chunk << log_onehot_k) + hot;
        uint ring_index = field_position >> log_ring_d;
        if (ring_index < block_ring_start
            || ring_index >= block_ring_start + positions_per_block) {
            continue;
        }
        uint position = ring_index - block_ring_start;
        uint shift = field_position & (ring_d - 1u);
        bool positive = coefficient >= shift;
        uint source_coefficient = (coefficient - shift) & (ring_d - 1u);
        uint column = position * (uint)params.num_digits_inner;
        uint matrix_index =
            ((row * active_a_cols + column) << log_ring_d)
            + source_coefficient;
        AkitaFp128 value = matrix[matrix_index];
        akita_wide_accumulate(deferred, value, positive);
        ++deferred_count;
        if (deferred_count == MAX_DEFERRED_ACCUMULATIONS) {
            AkitaFp128 segment = akita_reduce_wide(deferred);
            accumulator = has_reduced_segment
                ? akita_add(accumulator, segment)
                : segment;
            deferred = akita_wide_zero();
            deferred_count = 0u;
            has_reduced_segment = true;
        }
    }

    if (deferred_count != 0u) {
        AkitaFp128 segment = akita_reduce_wide(deferred);
        accumulator = has_reduced_segment
            ? akita_add(accumulator, segment)
            : segment;
    }

    uint output_index =
        ((source * num_blocks + block) * n_a + row) * ring_d
        + coefficient;
    output[output_index] = accumulator;
}

kernel void akita_onehot_commit_gather(
    device const AkitaFp128 *matrix [[buffer(0)]],
    device const ushort *hot_indices [[buffer(1)]],
    device AkitaFp128 *output [[buffer(2)]],
    constant OneHotCommitParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    ulong output_index = (ulong)gid;
    if (output_index >= params.output_coefficients) {
        return;
    }

    ulong outputs_per_source = params.num_blocks * params.n_a * params.ring_d;
    ulong source = output_index / outputs_per_source;
    ulong source_local = output_index - source * outputs_per_source;
    ulong block = source_local / (params.n_a * params.ring_d);
    ulong row_coefficient = source_local - block * params.n_a * params.ring_d;
    ulong row = row_coefficient / params.ring_d;
    ulong coefficient = row_coefficient - row * params.ring_d;

    ulong block_field_start = block * params.positions_per_block * params.ring_d;
    ulong block_field_end = min(
        block_field_start + params.positions_per_block * params.ring_d,
        params.total_field_elements);
    ulong chunk_start = block_field_start / params.onehot_k;
    ulong chunk_end = (block_field_end + params.onehot_k - 1ul) / params.onehot_k;

    AkitaFp128 accumulator = akita_zero();
    ulong source_chunk_base = source * params.chunks_per_source;
    ulong active_a_cols = params.positions_per_block * params.num_digits_inner;
    for (ulong chunk = chunk_start; chunk < chunk_end; ++chunk) {
        uint hot = hot_indices[source_chunk_base + chunk];
        if (hot == NONE_INDEX) {
            continue;
        }

        ulong field_position = chunk * params.onehot_k + (ulong)hot;
        ulong ring_index = field_position / params.ring_d;
        if (ring_index / params.positions_per_block != block) {
            continue;
        }
        ulong position = ring_index % params.positions_per_block;
        ulong shift = field_position % params.ring_d;
        bool positive = coefficient >= shift;
        ulong source_coefficient = positive
            ? coefficient - shift
            : params.ring_d + coefficient - shift;
        ulong column = position * params.num_digits_inner;
        ulong matrix_index =
            ((row * active_a_cols + column) * params.ring_d) + source_coefficient;
        AkitaFp128 value = matrix[matrix_index];
        accumulator = positive
            ? akita_add(accumulator, value)
            : akita_sub(accumulator, value);
    }
    output[output_index] = accumulator;
}

// Packed fp128 D128 rank-3 root commitment.
//
// Mapping for K = 256, D = 128: local field = row * 256 + lane, position =
// field / 128 (one trace row spans two ring positions), shift = field mod 128.
// Every hot entry adds the negacyclic rotation of one 128-coefficient row of
// each of the n_a = 3 matrix elements into the (column, block, element)
// accumulator. A threadgroup owns one matrix element: it streams that
// element's rows for eight positions per 20 KiB tile, decoding radix26 digits
// once for all tasks sharing the tile. Each SIMD group accumulates two tasks
// with four coefficients per lane.
#define PACKED_FP128_D128_RANK3_D 128u
#define PACKED_FP128_D128_RANK3_TILE_POSITIONS 8u
#define PACKED_FP128_D128_RANK3_ROWS_PER_TILE 4u
#define PACKED_FP128_D128_RANK3_TILE_ELEMENTS 1024u
#define PACKED_FP128_D128_RANK3_TASKS_PER_SIMDGROUP 2u
#define PACKED_FP128_D128_RANK3_TASKS_PER_STREAM 64u

static_assert(
    PACKED_FP128_D128_RANK3_TILE_POSITIONS * PACKED_FP128_D128_RANK3_D
        == PACKED_FP128_D128_RANK3_TILE_ELEMENTS,
    "D128 rank-3 decoded tile plane geometry");
static_assert(PACKED_FP128_D128_RANK3_ROWS_PER_TILE * 4u == 16u,
    "four tiles must contain at most sixteen selected contributions");
static_assert(16u % PACKED_FP128_D128_RANK3_TILE_POSITIONS == 0u,
    "internal tiles divide the registered position-partial alignment");
static_assert(17ul * ((1ul << 32ul) - ulong(AKITA_OFFSET)) < (1ul << 26ul),
    "radix26 folded low digit retains the carry bound");
static_assert(17ul * ((1ul << 26ul) + (1ul << 32ul) - ulong(AKITA_OFFSET))
    < (1ul << 31ul), "radix26 signed intermediates fit i32");

struct AkitaRadix26Accumulator {
    int4 d0, d1, d2, d3, d4;
};

inline AkitaRadix26Accumulator akita_radix26_zero() {
    AkitaRadix26Accumulator result;
    result.d0 = result.d1 = result.d2 = result.d3 = result.d4 = int4(0);
    return result;
}

// S=sum(d_i*2^(26i)) mod p. Four tiles add at most sixteen signed inputs.
// Propagated carries and the top quotient stay in [-17,17]; every i32
// intermediate has magnitude <=17*(2^26+22537)<2^31. The top fold uses
// 2^128 = AKITA_OFFSET = 64*2^26-22537 mod p. After folding, d0 lies in
// [-17*22537,2^26-1+17*22537], d1 in [-17*64,2^26-1+17*64], the next two
// digits in [0,2^26), and d4 in [0,2^24), restoring the induction invariant.
inline void akita_radix26_normalize(thread AkitaRadix26Accumulator &accumulator) {
    constexpr int mask = (1 << 26) - 1;
    int4 carry = accumulator.d0 >> 26;
    accumulator.d0 &= int4(mask);
    accumulator.d1 += carry;
    carry = accumulator.d1 >> 26;
    accumulator.d1 &= int4(mask);
    accumulator.d2 += carry;
    carry = accumulator.d2 >> 26;
    accumulator.d2 &= int4(mask);
    accumulator.d3 += carry;
    carry = accumulator.d3 >> 26;
    accumulator.d3 &= int4(mask);
    accumulator.d4 += carry;
    int4 high = accumulator.d4 >> 24;
    accumulator.d4 &= int4((1 << 24) - 1);
    constexpr int correction_low = int((1ul << 32ul) - ulong(AKITA_OFFSET));
    accumulator.d0 -= high * int4(correction_low);
    accumulator.d1 += high * int4(64);
}

inline int4 akita_radix26_gather(
    threadgroup const uint *matrix, uint digit, uint matrix_base, uint4 sources)
{
    uint base = digit * PACKED_FP128_D128_RANK3_TILE_ELEMENTS + matrix_base;
    return int4(matrix[base + sources[0]], matrix[base + sources[1]],
        matrix[base + sources[2]], matrix[base + sources[3]]);
}

inline void akita_radix26_accumulate(
    thread AkitaRadix26Accumulator &accumulator,
    threadgroup const uint *matrix,
    uint matrix_base,
    uint4 sources,
    bool4 positive)
{
    int4 sign = select(int4(-1), int4(1), positive);
    accumulator.d0 += sign * akita_radix26_gather(matrix, 0u, matrix_base, sources);
    accumulator.d1 += sign * akita_radix26_gather(matrix, 1u, matrix_base, sources);
    accumulator.d2 += sign * akita_radix26_gather(matrix, 2u, matrix_base, sources);
    accumulator.d3 += sign * akita_radix26_gather(matrix, 3u, matrix_base, sources);
    accumulator.d4 += sign * akita_radix26_gather(matrix, 4u, matrix_base, sources);
}

inline AkitaFp128 akita_reduce_radix26(AkitaRadix26Accumulator accumulator, uint component) {
    constexpr int mask = (1 << 26) - 1;
    int d0 = accumulator.d0[component];
    int d1 = accumulator.d1[component] + (d0 >> 26);
    d0 &= mask;
    int d2 = accumulator.d2[component] + (d1 >> 26);
    d1 &= mask;
    int d3 = accumulator.d3[component] + (d2 >> 26);
    d2 &= mask;
    int d4 = accumulator.d4[component] + (d3 >> 26);
    d3 &= mask;
    uint w0 = uint(d0) | (uint(d1) << 26u);
    uint w1 = (uint(d1) >> 6u) | (uint(d2) << 20u);
    uint w2 = (uint(d2) >> 12u) | (uint(d3) << 14u);
    uint w3 = (uint(d3) >> 18u) | (uint(d4) << 8u);
    AkitaWideAccumulator digits;
    digits.low_digits = int4(w0 & 65535u, w1 & 65535u, w2 & 65535u, w3 & 65535u);
    // Retain the signed quotient above bit128 for the canonical field reducer.
    digits.high_digits = int4(w0 >> 16u, w1 >> 16u, w2 >> 16u, d4 >> 8);
    return akita_reduce_wide(digits);
}

inline void akita_fp128_d128_rank3_accumulate_task_tile(
    thread AkitaRadix26Accumulator &accumulator,
    threadgroup const uint *shared_matrix,
    device const uchar *lanes,
    device const ulong *active_zero_rows,
    constant PackedOneHotCommitParams &params,
    ulong tile_row_base,
    uint task_column,
    uint simd_lane)
{
    uint local_hot = 0u;
    bool local_selected = false;
    if (simd_lane < PACKED_FP128_D128_RANK3_ROWS_PER_TILE) {
        ulong trace_row = tile_row_base + (ulong)simd_lane;
        local_hot = (uint)lanes[
            (trace_row - params.lane_row_offset) * params.lane_stride + (ulong)task_column];
        local_selected = local_hot != 0u;
        if (!local_selected
            && ((params.zero_column_mask >> task_column) & 1ul) != 0ul) {
            ulong active_word = active_zero_rows[trace_row >> 6ul];
            local_selected = ((active_word >> (trace_row & 63ul)) & 1ul) != 0ul;
        }
    }
    uint selected = uint(simd_ballot(local_selected).operator unsigned long());
    uint4 coefficients = uint4(simd_lane, simd_lane + 32u, simd_lane + 64u, simd_lane + 96u);
    while (selected != 0u) {
        uint selected_lane = ctz(selected);
        uint selected_hot = simd_shuffle(local_hot, selected_lane);
        uint local_position = 2u * selected_lane + (selected_hot >> 7u);
        uint4 shift = uint4(selected_hot & 127u);
        akita_radix26_accumulate(
            accumulator, shared_matrix, local_position * PACKED_FP128_D128_RANK3_D,
            (coefficients - shift) & uint4(127u), coefficients >= shift);
        selected &= selected - 1u;
    }
}

inline void akita_store_fp128_d128_rank3(
    device AkitaFp128 *partials,
    AkitaRadix26Accumulator accumulator,
    constant PackedOneHotCommitParams &params,
    uint task_column,
    uint task_block,
    uint element,
    uint position_partial,
    uint simd_lane)
{
    ulong block = (ulong)task_column * params.blocks_per_column + (ulong)task_block;
    ulong output_base = (block * params.n_a + (ulong)element) * (ulong)PACKED_FP128_D128_RANK3_D;
    ulong partial_base = (ulong)position_partial * params.output_coefficients + output_base;
    partials[partial_base + simd_lane] = akita_reduce_radix26(accumulator, 0u);
    partials[partial_base + simd_lane + 32ul] = akita_reduce_radix26(accumulator, 1u);
    partials[partial_base + simd_lane + 64ul] = akita_reduce_radix26(accumulator, 2u);
    partials[partial_base + simd_lane + 96ul] = akita_reduce_radix26(accumulator, 3u);
}

kernel void akita_packed_onehot_commit_fp128_d128_rank3(
    device const AkitaFp128 *matrix [[buffer(0)]],
    device const uchar *lanes [[buffer(1)]],
    device AkitaFp128 *partials [[buffer(2)]],
    constant PackedOneHotCommitParams &params [[buffer(3)]],
    device const ulong *active_zero_rows [[buffer(4)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup uint shared_matrix[PACKED_FP128_D128_RANK3_TILE_ELEMENTS * 5];

    constexpr uint tasks_per_stream = PACKED_FP128_D128_RANK3_TASKS_PER_STREAM;
    constexpr uint threads_per_threadgroup = 1024u;
    uint num_tasks = (uint)params.dispatch_tasks;
    uint streams = (num_tasks + tasks_per_stream - 1u) / tasks_per_stream;
    uint simd_lane = thread_index & 31u;
    uint simdgroup = thread_index >> 5u;
    uint position_partials = (uint)params.position_partials_per_block;
    uint stream = threadgroup_index.x % streams;
    uint partial_group = threadgroup_index.x / streams;
    uint position_partial = partial_group % position_partials;
    uint element = partial_group / position_partials;
    uint positions_per_partial = (uint)params.positions_per_partial;
    uint partial_start = position_partial * positions_per_partial;
    ulong rows_per_partial = (ulong)positions_per_partial / 2ul;
    ulong rows_per_block = params.positions_per_block / 2ul;
    uint live_columns = (uint)params.num_columns;
    uint dispatch_task_0 = stream * tasks_per_stream
        + simdgroup * PACKED_FP128_D128_RANK3_TASKS_PER_SIMDGROUP;
    bool active_0 = dispatch_task_0 < num_tasks;
    bool active_1 = dispatch_task_0 + 1u < num_tasks;
    uint global_0 = (uint)params.task_offset + dispatch_task_0;
    uint global_1 = global_0 + 1u;
    uint block_0 = global_0 / live_columns;
    uint column_0 = global_0 % live_columns;
    uint block_1 = global_1 / live_columns;
    uint column_1 = global_1 % live_columns;
    ulong matrix_cursor =
        ((ulong)element * params.positions_per_block + (ulong)partial_start)
        * (ulong)PACKED_FP128_D128_RANK3_D;

    AkitaRadix26Accumulator accumulator_0 = akita_radix26_zero();
    AkitaRadix26Accumulator accumulator_1 = akita_radix26_zero();

    uint tile_count = positions_per_partial / PACKED_FP128_D128_RANK3_TILE_POSITIONS;
    for (uint tile = 0u; tile < tile_count; ++tile) {
        for (uint shared_index = thread_index;
             shared_index < PACKED_FP128_D128_RANK3_TILE_ELEMENTS;
             shared_index += threads_per_threadgroup) {
            AkitaFp128 value = matrix[matrix_cursor + (ulong)shared_index];
            constexpr uint mask = (1u << 26u) - 1u;
            shared_matrix[shared_index] = value.limb[0] & mask;
            shared_matrix[PACKED_FP128_D128_RANK3_TILE_ELEMENTS + shared_index] =
                ((value.limb[0] >> 26u) | (value.limb[1] << 6u)) & mask;
            shared_matrix[PACKED_FP128_D128_RANK3_TILE_ELEMENTS * 2u + shared_index] =
                ((value.limb[1] >> 20u) | (value.limb[2] << 12u)) & mask;
            shared_matrix[PACKED_FP128_D128_RANK3_TILE_ELEMENTS * 3u + shared_index] =
                ((value.limb[2] >> 14u) | (value.limb[3] << 18u)) & mask;
            shared_matrix[PACKED_FP128_D128_RANK3_TILE_ELEMENTS * 4u + shared_index] =
                value.limb[3] >> 8u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ulong tile_rows = (ulong)position_partial * rows_per_partial
            + (ulong)tile * (ulong)PACKED_FP128_D128_RANK3_ROWS_PER_TILE;
        if (active_0) {
            akita_fp128_d128_rank3_accumulate_task_tile(
                accumulator_0, shared_matrix, lanes, active_zero_rows, params,
                (ulong)block_0 * rows_per_block + tile_rows, column_0, simd_lane);
        }
        if (active_1) {
            akita_fp128_d128_rank3_accumulate_task_tile(
                accumulator_1, shared_matrix, lanes, active_zero_rows, params,
                (ulong)block_1 * rows_per_block + tile_rows, column_1, simd_lane);
        }
        if ((tile & 3u) == 3u) {
            if (active_0) akita_radix26_normalize(accumulator_0);
            if (active_1) akita_radix26_normalize(accumulator_1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        matrix_cursor += (ulong)PACKED_FP128_D128_RANK3_TILE_ELEMENTS;
    }

    if (active_0) {
        akita_store_fp128_d128_rank3(
            partials, accumulator_0, params, column_0, block_0, element, position_partial,
            simd_lane);
    }
    if (active_1) {
        akita_store_fp128_d128_rank3(
            partials, accumulator_1, params, column_1, block_1, element, position_partial,
            simd_lane);
    }
}

// Packed decompose-fold for the D128 rank-3 row. One threadgroup of 128
// threads owns one ring position: at K = 256, D = 128 a trace row spans two
// positions, so position `p` of a block reads row `p / 2` and keeps the hot
// lanes whose top bit equals `p & 1`; committed-zero lanes (hot = 0) belong to
// the even position with coefficient 0.
kernel void akita_fp128_d128_decompose_fold(
    device const uchar *lanes [[buffer(0)]],
    device const ushort *challenge_positions [[buffer(1)]],
    device const char *challenge_coefficients [[buffer(2)]],
    device int *output [[buffer(3)]],
    constant PackedDecomposeFoldParams &params [[buffer(4)]],
    device const ulong *active_zero_rows [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup atomic_int accumulators[128];

    uint local_position = threadgroup_index.x;
    ulong position = params.position_start + (ulong)local_position;
    atomic_store_explicit(&accumulators[thread_index], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong tasks_per_position = params.blocks_per_column * params.num_columns;
    for (ulong task = (ulong)thread_index; task < tasks_per_position; task += 128ul) {
        ulong trace_block = task / params.num_columns;
        ulong column = task % params.num_columns;
        ulong ring = trace_block * params.num_positions + position;
        ulong row = ring >> 1ul;
        uint ring_half = (uint)(ring & 1ul);
        uchar hot = lanes[row * params.lane_stride + column];
        bool committed = hot != 0u;
        if (!committed && column < 64ul
            && ((params.zero_column_mask >> column) & 1ul) != 0ul) {
            ulong active_word = active_zero_rows[row >> 6ul];
            committed = ((active_word >> (row & 63ul)) & 1ul) != 0ul;
        }
        if (!committed || ((uint)hot >> 7u) != ring_half) {
            continue;
        }

        uint source_coefficient = (uint)hot & 127u;
        ulong challenge = column * params.blocks_per_column + trace_block;
        ulong challenge_start = challenge * params.challenge_weight;
        for (ulong term = 0ul; term < params.challenge_weight; ++term) {
            uint destination = source_coefficient
                + (uint)challenge_positions[challenge_start + term];
            int value = (int)challenge_coefficients[challenge_start + term];
            if (destination >= 128u) {
                destination -= 128u;
                value = -value;
            }
            atomic_fetch_add_explicit(
                &accumulators[destination], value, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong output_base = (ulong)local_position * 128ul;
    output[output_base + (ulong)thread_index] = atomic_load_explicit(
        &accumulators[thread_index], memory_order_relaxed);
}
