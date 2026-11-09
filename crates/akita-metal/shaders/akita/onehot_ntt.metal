// Exact K16/D512 convolutions of eight 16-bit key limbs. The host derives
// the reconstruction span from the prime so span*32*65535 < p/2. Each
// batch reconstructs into Fp128 before its results are accumulated further.
namespace akita {
struct OneHotLimbSource {
    ulong fields, chunks, hot_offset;
    uint first_ring, hot_stride;
};
struct OneHotLimbParams {
    NttPrime prime;
    uint positions, columns, digits, rows, rings, reconstruction_positions;
    uint first_chunk, parts, direct_output;
};

template <uint C, uint DOT_BATCH, uint MODULUS, typename KeyIndex>
kernel void akita_onehot_limb_commit(
    device const uchar* hot [[buffer(0)]],
    device const uint* key [[buffer(1)]],
    device const uint* tables [[buffer(2)]],
    device const OneHotLimbSource* sources [[buffer(3)]],
    device const uint* source_for_ring [[buffer(4)]],
    device jolt::Fp128<C>* out [[buffer(5)]],
    constant OneHotLimbParams& p [[buffer(6)]],
    device const uchar* ring_kinds [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    static_assert(DOT_BATCH > 0 && DOT_BATCH <= 8, "dot products fit u64 and the shared tile");
    using F = jolt::Fp128<C>;
    uint task = group % p.rings;
    uint part = group / p.rings;
    OneHotLimbSource source = sources[source_for_ring[task]];
    uint local_task = task - source.first_ring;
    uint block = local_task / p.rows, row_a = local_task % p.rows;
    uint begin = (p.first_chunk + part) * p.reconstruction_positions;
    uint end = metal::min(p.positions, begin + p.reconstruction_positions);
    uint modulus = MODULUS;
    NttPrime prime = p.prime;
    prime.p = int(MODULUS);
    // Geometry requires a whole number of D512 rings. Logical length is
    // therefore divisible by 16, and validated present digits are < 16.
    // Bounding chunks is equivalent to testing chunk*16+digit < fields.
    ulong live_chunks = metal::min(source.chunks, source.fields / 16);
    ulong source_base = source.hot_offset + ulong(block) * p.positions * 32 * source.hot_stride;
    ulong block_chunks = ulong(block) * p.positions * 32;
    // Short device backing denotes absent entries, including trimmed tails.
    // Whole zero chunks still overwrite their partials (or the first output),
    // but require neither key reads nor transforms.
    if (block_chunks + ulong(begin) * 32 >= live_chunks) {
        for (uint coeff = tid; coeff < 512; coeff += 256) {
            if (!p.direct_output || p.first_chunk == 0)
                out[ulong(group) * 512 + coeff] = F::zero();
        }
        return;
    }
    KeyIndex key_row = KeyIndex(row_a) * p.columns * 4096;
    KeyIndex key_stride = KeyIndex(p.digits) * 4096;
    device const uint* tw = tables + 16384;
    device const uint* twq = tw + 8;
    device const uint* invw = twq + 8;
    device const uint* invwq = invw + 256;
    device const uint* untwist = invwq + 256;
    device const uint* untwistq = untwist + 512;
    device const uint* uniform_spectra = untwistq + 512;
    threadgroup uint work[4096];
    uint acc[16] = {};
    for (uint first = begin; first < end; first += DOT_BATCH) {
        KeyIndex first_key = key_row + KeyIndex(first) * key_stride;
        if (tid < DOT_BATCH * 32) {
            uint r = tid / 32, c = tid % 32, pos = first + r;
            for (uint v = 0; v < 16; v++) work[r * 512 + v * 32 + c] = 0;
            if (pos < end) {
                ulong chunk = block_chunks + ulong(pos) * 32;
                uint present = chunk < live_chunks ? uint(metal::min(32ul, live_chunks - chunk)) : 0u;
                uint kind = present == 0 ? 255u : ring_kinds[
                    source.hot_offset + (ulong(block) * p.positions + pos) * source.hot_stride];
                if (kind < 16u) {
                    for (uint v = 0; v < 16; v++)
                        work[r * 512 + v * 32 + c] = uniform_spectra[kind * 512 + v * 32 + c];
                } else if (kind != 255u) {
                    for (uint row = 0; row < present; row++) {
                        uint v = hot[source_base + (ulong(pos) * 32 + row) * source.hot_stride];
                        if (v != 255u) {
                            uint i = r * 512 + v * 32 + c;
                            work[i] = uint(csubp(prime, int(work[i] + tables[(v * 32 + row) * 32 + c])));
                        }
                    }
                    uint a[16];
                    for (uint v = 0; v < 16; v++) a[v] = work[r * 512 + v * 32 + c];
                    for (uint half_len = 8; half_len >= 1; half_len /= 2)
                        for (uint start = 0; start < 16; start += 2 * half_len)
                            for (uint j = 0; j < half_len; j++) {
                                uint x = a[start + j], y = a[start + j + half_len];
                                a[start + j] = uint(csubp(prime, int(x + y)));
                                uint diff = uint(caddp(prime, int(x) - int(y)));
                                a[start + j + half_len] = j == 0 ? diff :
                                    shoup_mul_canonical(diff, tw[j * (8 / half_len)], twq[j * (8 / half_len)], modulus);
                            }
                    for (uint v = 0; v < 16; v++) {
                        uint rev = ((v & 1) << 3) | ((v & 2) << 1) | ((v & 4) >> 1) | ((v & 8) >> 3);
                        work[r * 512 + rev * 32 + c] = a[v];
                    }
                }
            }
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        for (uint i = 0; i < 16; i++) {
            uint idx = tid + i * 256, coeff = idx % 512;
            uint lo = 0, hi = 0;
            for (uint r = 0; r < DOT_BATCH; r++) if (first + r < end) {
                uint a = key[first_key + KeyIndex(r) * key_stride + idx];
                uint b = work[r * 512 + coeff];
                uint next_lo = lo + a * b;
                hi += metal::mulhi(a, b) + uint(next_lo < lo);
                lo = next_lo;
            }
            // At most eight canonical products with p < 2^30. The sum plus
            // its Montgomery correction fits u64; the quotient is below 3p.
            // The correction's low word is -lo, so the carry is exactly lo!=0.
            uint correction = lo * (0u - uint(p.prime.pinv));
            uint v = hi + metal::mulhi(correction, modulus) + uint(lo != 0);
            // v can exceed i32::MAX before the two reductions.
            v = metal::select(v, v - modulus, v >= modulus);
            v = metal::select(v, v - modulus, v >= modulus);
            acc[i] = uint(csubp(prime, int(acc[i] + v)));
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 16; i++) {
        uint idx = tid + i * 256;
        work[(idx / 512) * 512 + (metal::reverse_bits(idx % 512) >> 23)] = acc[i];
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    for (uint len = 2; len <= 512; len *= 2) {
        uint half_len = len / 2;
        for (uint i = tid; i < 2048; i += 256) {
            uint limb = i / 256, b = i % 256, j = b % half_len, start = (b / half_len) * len;
            uint lo = limb * 512 + start + j, hi = lo + half_len;
            uint u = work[lo];
            uint v = shoup_mul_canonical(work[hi], invw[j * (512 / len)], invwq[j * (512 / len)], modulus);
            work[lo] = uint(csubp(prime, int(u + v)));
            work[hi] = uint(caddp(prime, int(u) - int(v)));
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    for (uint coeff = tid; coeff < 512; coeff += 256) {
        long carry = 0; uint4 lo(0u);
        for (uint limb = 0; limb < 8; limb++) {
            uint residue = shoup_mul_canonical(work[limb * 512 + coeff], untwist[coeff], untwistq[coeff], modulus);
            long centered = residue > modulus / 2 ? long(residue) - modulus : long(residue);
            long v = centered + carry;
            lo[limb / 2] |= uint(ushort(v)) << (16 * (limb % 2));
            carry = v >> 16;
        }
        F value{jolt::fp128_detail::fold2_canonicalize<C>(lo, 0)};
        value = value + F::from_i64(carry * long(C));
        ulong index = ulong(group) * 512 + coeff;
        out[index] = p.direct_output && p.first_chunk != 0 ? out[index] + value : value;
    }
}

template <uint C>
kernel void akita_onehot_limb_combine(
    device const jolt::Fp128<C>* partials [[buffer(0)]],
    device jolt::Fp128<C>* out [[buffer(1)]],
    constant OneHotLimbParams& p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    using F = jolt::Fp128<C>;
    ulong len = ulong(p.rings) * 512;
    if (i >= len) return;
    F sum = p.first_chunk == 0 ? F::zero() : out[i];
    for (uint part = 0; part < p.parts; part++) sum = sum + partials[ulong(part) * len + i];
    out[i] = sum;
}
} // namespace akita
