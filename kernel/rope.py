import itertools

import triton
import triton.language as tl

# for dim 768: BLOCK_SIZE = 32 num_warps = 2 num_stages = 2
@triton.jit
def rope_1d_kernel(Q, K, freq, pos, dim: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (dim // 2)

    freq_v = tl.load(freq + offsets, mask=mask, other=0.0)
    p_freq = freq_v * pos
    fcr, fci = tl.cos(p_freq), tl.sin(p_freq)

    q_even = tl.load(Q + 2 * offsets,     mask=mask, other=0.0)
    q_odd  = tl.load(Q + 2 * offsets + 1, mask=mask, other=0.0)
    k_even = tl.load(K + 2 * offsets,     mask=mask, other=0.0)
    k_odd  = tl.load(K + 2 * offsets + 1, mask=mask, other=0.0)

    q_new_even = q_even * fcr - q_odd * fci
    q_new_odd  = q_even * fci + q_odd * fcr
    k_new_even = k_even * fcr - k_odd * fci
    k_new_odd  = k_even * fci + k_odd * fcr

    mask_even = mask
    mask_odd  = mask

    tl.store(Q + 2 * offsets,     q_new_even, mask=mask_even)
    tl.store(Q + 2 * offsets + 1, q_new_odd,  mask=mask_odd)
    tl.store(K + 2 * offsets,     k_new_even, mask=mask_even)
    tl.store(K + 2 * offsets + 1, k_new_odd,  mask=mask_odd)