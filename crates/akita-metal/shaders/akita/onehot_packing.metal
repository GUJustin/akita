// Base-field coefficient packing. Each lane owns a coefficient; nearby lanes
// compare the same compact selector against the sixteen possible hot values.
namespace akita {
struct OneHotPackingParams {
    ulong prefix_rows;
    uint positions, columns, blocks_per_column, subring;
    uint tiles, local_block, output_block, logical_rings;
};

template <uint C>
kernel void akita_onehot_packing_gather(
    device const uchar* hot [[buffer(0)]],
    device const jolt::Fp128<C>* weights [[buffer(1)]],
    device jolt::Fp128<C>* partials [[buffer(2)]],
    constant OneHotPackingParams& p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    using F = jolt::Fp128<C>;
    uint coefficient = index % 512;
    uint tile = (index / 512) % p.tiles;
    uint column = index / (512 * p.tiles);
    if (column >= p.columns) return;
    uint begin = tile * 1024;
    uint end = metal::min(p.positions, begin + 1024);
    uint row_offset = coefficient / 16, digit = coefficient % 16;
    F sum = F::zero();
    for (uint position = begin; position < end; position++) {
        ulong ring = ulong(p.local_block) * p.positions + position;
        ulong row = ring * 32 + row_offset;
        if (ring >= p.logical_rings || row >= p.prefix_rows) break;
        if (uint(hot[row * p.columns + column]) == digit)
            sum = sum + weights[position];
    }
    // Complete write, including zero/omitted source suffixes.
    partials[index] = sum;
}

template <uint C>
kernel void akita_onehot_packing_combine(
    device const jolt::Fp128<C>* partials [[buffer(0)]],
    device const jolt::Fp128<C>* weights [[buffer(1)]],
    device jolt::Fp128<C>* output [[buffer(2)]],
    constant OneHotPackingParams& p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    using F = jolt::Fp128<C>;
    uint subring_coefficient = index % p.subring, column = index / p.subring;
    if (column >= p.columns) return;
    uint stride = 512 / p.subring;
    F packed = F::zero();
    for (uint low = 0; low < stride; low++) {
        uint coefficient = subring_coefficient * stride + low;
        F sum = F::zero();
        for (uint tile = 0; tile < p.tiles; tile++)
            sum = sum + partials[(column * p.tiles + tile) * 512 + coefficient];
        packed = packed + sum * weights[low];
    }
    output[(ulong(column) * p.blocks_per_column + p.output_block) * p.subring + subring_coefficient] = packed;
}
} // namespace akita
