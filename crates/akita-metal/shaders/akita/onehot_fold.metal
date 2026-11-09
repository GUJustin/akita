// Each SIMD group accumulates sparse monomial products into its own ring.
// Threadgroup atomics resolve overlapping terms without global contention.
namespace akita {
struct OneHotFoldParams {
    ulong prefix_rows;
    uint positions, columns, blocks_per_column, first_block, chunks, padding;
};
struct OneHotFoldTerm { ushort position; short coefficient; };

[[host_name("akita_k16_sparse_fold")]]
kernel void akita_k16_sparse_fold(
    device const uchar* hot_rows [[buffer(0)]],
    device const OneHotFoldTerm* terms [[buffer(1)]],
    device int* output [[buffer(2)]],
    constant OneHotFoldParams& p [[buffer(3)]],
    device const uint2* ranges [[buffer(4)]],
    device const uint* offsets [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    threadgroup metal::atomic_int partials[8 * 512];
    for (uint i = tid; i < 8 * 512; i += 256)
        metal::atomic_store_explicit(&partials[i], 0, metal::memory_order_relaxed);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    uint lane = tid & 31u, simdgroup = tid >> 5u;
    uint position = group / p.chunks, chunk = group % p.chunks;
    uint2 range = ranges[chunk];
    uint rings = (range.y - range.x) * p.columns;
    for (uint ring = simdgroup; ring < rings; ring += 8u) {
        uint block = range.x + ring / p.columns;
        uint column = ring % p.columns;
        ulong row = 32ul * (ulong(block - p.first_block) * p.positions + position) + lane;
        uint hot = row < p.prefix_rows ? hot_rows[row * p.columns + column] : 255u;
        if (hot == 255u) continue;
        uint challenge = column * p.blocks_per_column + block;
        uint exponent = 16u * lane + hot;
        for (uint i = offsets[challenge]; i < offsets[challenge + 1u]; ++i) {
            OneHotFoldTerm term = terms[i];
            uint target = exponent + uint(term.position);
            int value = int(term.coefficient);
            if (target >= 512u) value = -value;
            metal::atomic_fetch_add_explicit(&partials[simdgroup * 512u + (target & 511u)], value, metal::memory_order_relaxed);
        }
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    for (uint coefficient = tid; coefficient < 512u; coefficient += 256u) {
        int sum = 0;
        for (uint producer = 0; producer < 8u; ++producer)
            sum += metal::atomic_load_explicit(&partials[producer * 512u + coefficient], metal::memory_order_relaxed);
        output[group * 512u + coefficient] += sum;
    }
}
} // namespace akita
