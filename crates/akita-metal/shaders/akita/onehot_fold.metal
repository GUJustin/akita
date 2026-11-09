// Each group partitions monomials by exponent modulo eight, then evaluates
// two 64-subring coefficients per lane without integer atomics.
namespace akita {
struct OneHotFoldParams {
    ulong prefix_rows;
    uint positions, columns, blocks_per_column, first_block, chunks, padding;
};

[[host_name("akita_k16_subring64_fold")]]
kernel void akita_k16_subring64_fold(
    device const uchar* hot_rows [[buffer(0)]],
    device const char* challenges [[buffer(1)]],
    device int* output [[buffer(2)]],
    constant OneHotFoldParams& p [[buffer(3)]],
    device const uint2* ranges [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    threadgroup uint partitioned[8 * 8 * 32];
    threadgroup uint counts[8 * 8];
    uint lane = tid & 31u, simdgroup = tid >> 5u;
    uint position = group / p.chunks, chunk = group % p.chunks;
    uint2 range = ranges[chunk];
    uint tasks = (range.y - range.x) * p.columns * 32u;
    int low_sum = 0, high_sum = 0;
    for (uint base = 0; base < tasks; base += 256u) {
        uint task = base + tid;
        bool valid = task < tasks;
        uint hot = 255u, source_high = 0, challenge = 0;
        if (valid) {
            uint block = range.x + task / (p.columns * 32u);
            uint column = (task >> 5u) % p.columns;
            uint row_in_ring = task & 31u;
            ulong row = 32ul * (ulong(block - p.first_block) * p.positions + position) + row_in_ring;
            if (row < p.prefix_rows) hot = hot_rows[row * p.columns + column];
            valid = hot != 255u;
            source_high = 2u * row_in_ring + (hot >> 3u);
            challenge = column * p.blocks_per_column + block;
        }
        for (uint low = 0; low < 8u; ++low) {
            bool selected = valid && ((hot & 7u) == low);
            uint mask = uint(metal::simd_ballot(selected).operator unsigned long());
            uint partition = simdgroup * 8u + low;
            if (lane == 0) counts[partition] = metal::popcount(mask);
            if (selected) {
                uint preceding = lane == 0 ? 0u : ((1u << lane) - 1u);
                uint rank = metal::popcount(mask & preceding);
                partitioned[partition * 32u + rank] = (challenge << 6u) | source_high;
            }
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        for (uint producer = 0; producer < 8u; ++producer) {
            uint partition = producer * 8u + simdgroup;
            for (uint index = 0; index < counts[partition]; ++index) {
                uint packed = partitioned[partition * 32u + index];
                uint source = packed & 63u, challenge_index = packed >> 6u;
                uint offset = challenge_index * 64u;
                int low = int(challenges[offset + ((lane + 64u - source) & 63u)]);
                int high = int(challenges[offset + ((lane + 96u - source) & 63u)]);
                low_sum += lane < source ? -low : low;
                high_sum += lane + 32u < source ? -high : high;
            }
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    uint offset = group * 512u + simdgroup + 8u * lane;
    output[offset] += low_sum;
    output[offset + 256u] += high_sum;
}
} // namespace akita
