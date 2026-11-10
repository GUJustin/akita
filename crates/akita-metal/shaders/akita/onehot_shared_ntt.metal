namespace akita {
// Canonical uint spectra, not the balanced-byte representation used by TensorOps.
// A SIMD group owns one source ring/position and stores consecutive frequencies.
template <uint MODULUS>
kernel void akita_onehot_shared_source(
    device const uchar* hot [[buffer(0)]],
    device const uchar* kinds [[buffer(1)]],
    device const uint* tables [[buffer(2)]],
    device const OneHotLimbSource* sources [[buffer(3)]],
    device const uint* source_for_ring [[buffer(4)]],
    device uint* spectra [[buffer(5)]],
    constant OneHotLimbParams& p [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    uint lane = index % 32, task = index / 32;
    uint ring = task % p.rings;
    uint position_in_part = (task / p.rings) % p.reconstruction_positions;
    uint part = task / (p.rings * p.reconstruction_positions);
    if (part >= p.parts) return;
    uint position = (p.first_chunk + part) * p.reconstruction_positions + position_in_part;
    OneHotLimbSource source = sources[source_for_ring[ring]];
    uint block = ring - source.first_ring; // admission requires rank one
    ulong first = (ulong(block) * p.positions + position) * 32;
    ulong live = metal::min(source.chunks, source.fields / 16);
    uint a[16] = {};
    // Four SIMD groups own disjoint sets of sixteen bins per lane (8 KiB).
    // Each lane initializes, updates and reads only its own words, so there
    // are no cross-lane consumers and no additional barrier is needed.
    threadgroup uint bins[2048];
    uint bin_base = ((index / 32) % 4) * 512 + lane;
    NttPrime prime = p.prime;
    prime.p = int(MODULUS);
    if (position < p.positions && first < live) {
        uint present = uint(metal::min(32ul, live - first));
        uint kind = kinds[source.hot_offset + (first / 32) * source.hot_stride];
        if (kind < 16) {
            for (uint v = 0; v < 16; v++)
                a[v] = tables[17936 + kind * 512 + v * 32 + lane];
        } else if (kind != 255) {
            for (uint v = 0; v < 16; v++) bins[bin_base + v * 32] = 0;
            for (uint row = 0; row < present; row++) {
                uint digit = hot[source.hot_offset + (first + row) * source.hot_stride];
                if (digit != 255) {
                    uint root = tables[(digit * 32 + row) * 32 + lane];
                    uint offset = bin_base + digit * 32;
                    bins[offset] = uint(csubp(prime, int(bins[offset] + root)));
                }
            }
            for (uint v = 0; v < 16; v++) a[v] = bins[bin_base + v * 32];
            onehot_ntt16(a, prime, tables + 16384, tables + 16392, MODULUS);
            for (uint v = 0; v < 16; v++) {
                uint rev = ((v & 1) << 3) | ((v & 2) << 1) | ((v & 4) >> 1) | ((v & 8) >> 3);
                if (rev > v) { uint tmp = a[v]; a[v] = a[rev]; a[rev] = tmp; }
            }
        }
    }
    // All allocated slots are written, including the final span's zero suffix.
    for (uint v = 0; v < 16; v++) {
        uint frequency = v * 32 + lane;
        ulong out = (((ulong(part) * 8 + frequency / 64) * p.reconstruction_positions
            + position_in_part) * p.rings + ring) * 64 + frequency % 64;
        spectra[out] = a[v];
    }
}

template <uint MODULUS, uint DOT_BATCH>
kernel void akita_onehot_shared_dot(
    device const uint* spectra [[buffer(0)]],
    device const uint* key [[buffer(1)]],
    device uint* residues [[buffer(2)]],
    constant OneHotLimbParams& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    static_assert(DOT_BATCH == 6, "the shared tile has six key positions");
    uint tiles = (p.rings + 15) / 16;
    uint tile = group % tiles, frequency_tile = (group / tiles) % 8;
    uint part = group / (tiles * 8);
    uint ring = tile * 16 + tid / 32, lane = tid % 32;
    uint begin = (p.first_chunk + part) * p.reconstruction_positions;
    uint end = metal::min(p.positions, begin + p.reconstruction_positions);
    NttPrime prime = p.prime;
    prime.p = int(MODULUS);
    // 12 KiB, with no other static arrays; 24 KiB with validation doubling.
    threadgroup uint key_tile[6 * 8 * 64];
    uint acc[16] = {};
    for (uint first = begin; first < end; first += DOT_BATCH) {
        for (uint i = tid; i < 6 * 8 * 64; i += 512) {
            uint position = first + i / 512;
            uint limb = (i / 64) % 8, frequency = frequency_tile * 64 + i % 64;
            key_tile[i] = position < end
                ? key[ulong(position) * 4096 + limb * 512 + frequency] : 0;
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        if (ring < p.rings) {
            for (uint f = 0; f < 2; f++) {
                uint frequency = lane + f * 32;
                uint source[6] = {};
                for (uint r = 0; r < DOT_BATCH; r++) if (first + r < end) {
                    ulong offset = (((ulong(part) * 8 + frequency_tile) * p.reconstruction_positions
                        + first - begin + r) * p.rings + ring) * 64 + frequency;
                    source[r] = spectra[offset];
                }
                for (uint limb = 0; limb < 8; limb++) {
                    ulong sum = 0;
                    for (uint r = 0; r < DOT_BATCH; r++)
                        sum += ulong(key_tile[(r * 8 + limb) * 64 + frequency]) * source[r];
                    uint value = onehot_limb_dot_reduce(sum, prime, MODULUS);
                    uint i = f * 8 + limb;
                    acc[i] = uint(csubp(prime, int(acc[i] + value)));
                }
            }
        }
        // Inactive rounded output tasks still synchronize before tile reuse.
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    if (ring < p.rings) {
        for (uint f = 0; f < 2; f++)
            for (uint limb = 0; limb < 8; limb++)
                residues[((ulong(part) * p.rings + ring) * 8 + limb) * 512
                    + frequency_tile * 64 + f * 32 + lane] = acc[f * 8 + limb];
    }
}

template <uint C, uint MODULUS>
kernel void akita_onehot_shared_inverse(
    device const uint* residues [[buffer(0)]],
    device const uint* tables [[buffer(1)]],
    device jolt::Fp128<C>* partials [[buffer(2)]],
    constant OneHotLimbParams& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint work[4096];
    for (uint i = tid; i < 4096; i += 256)
        work[(i / 512) * 512 + (metal::reverse_bits(i % 512) >> 23)] = residues[ulong(group) * 4096 + i];
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    onehot_limb_inverse<C, MODULUS>(work, tables, partials, p, group, tid);
}
} // namespace akita
