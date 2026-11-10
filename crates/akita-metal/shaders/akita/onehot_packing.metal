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

namespace akita {
// The same canonical tensor-product weights, prepared once for the four low
// coefficients of a D512/S128 packing bucket. No trace data is materialized.
template <uint C>
kernel void akita_onehot_packing_weights(
    device const jolt::Fp128<C>* positions [[buffer(0)]],
    device const jolt::Fp128<C>* packing [[buffer(1)]],
    device jolt::Fp128<C>* combined [[buffer(2)]],
    constant uint& count [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    uint position = index / 4, low = index % 4;
    if (position < count) combined[index] = positions[position] * packing[low];
}

template <uint C>
kernel void akita_onehot_packing_bins(
    device const uchar* hot [[buffer(0)]],
    device const jolt::Fp128<C>* combined [[buffer(1)]],
    device jolt::Fp128<C>* partials [[buffer(2)]],
    constant OneHotPackingParams& p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    using F = jolt::Fp128<C>;
    uint row_offset = index % 32;
    uint tile = (index / 32) % p.tiles;
    uint column = index / (32 * p.tiles);
    if (column >= p.columns) return;
    F bins[4] = {F::zero(), F::zero(), F::zero(), F::zero()};
    uint begin = tile * 1024, end = metal::min(p.positions, begin + 1024);
    for (uint position = begin; position < end; position++) {
        ulong ring = ulong(p.local_block) * p.positions + position;
        ulong row = ring * 32 + row_offset;
        if (ring >= p.logical_rings || row >= p.prefix_rows) break;
        uint digit = hot[row * p.columns + column];
        if (digit < 16) {
            uint bin = digit >> 2;
            bins[bin] = bins[bin] + combined[ulong(position) * 4 + (digit & 3)];
        }
    }
    ulong base = (ulong(column) * p.tiles + tile) * 128 + row_offset * 4;
    for (uint bin = 0; bin < 4; bin++) partials[base + bin] = bins[bin];
}

template <uint C>
kernel void akita_onehot_packing_bins_combine(
    device const jolt::Fp128<C>* partials [[buffer(0)]],
    device jolt::Fp128<C>* output [[buffer(1)]],
    constant OneHotPackingParams& p [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    using F = jolt::Fp128<C>;
    uint coefficient = index % 128, column = index / 128;
    if (column >= p.columns) return;
    F sum = F::zero();
    for (uint tile = 0; tile < p.tiles; tile++)
        sum = sum + partials[(ulong(column) * p.tiles + tile) * 128 + coefficient];
    output[(ulong(column) * p.blocks_per_column + p.output_block) * 128 + coefficient] = sum;
}
} // namespace akita
