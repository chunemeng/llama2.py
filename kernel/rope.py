import itertools

import triton
import triton.language as tl


# for dim 768: BLOCK_SIZE = 32 num_warps = 2 num_stages = 2
@triton.jit
def rope_1d_kernel(Q, K, freq, pos, n_kv_heads: tl.constexpr, head_dim: tl.constexpr, kv_mul: tl.constexpr,
                   BLOCK_COL_SIZE: tl.constexpr, BLOCK_ROW_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    off_row = pid * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)
    off_col = tl.arange(0, BLOCK_COL_SIZE)
    mask_row = off_row < n_kv_heads
    mask_col = off_col < (head_dim // 2)

    freq_c = tl.load(freq + off_col, mask=mask_col, other=0.0)
    freq_v = freq_c[None, :]
    p_freq = freq_v * pos
    fcr, fci = tl.cos(p_freq), tl.sin(p_freq)
    mask = mask_row[:, None] & mask_col[None, :]

    q_offset_row = pid * BLOCK_ROW_SIZE * kv_mul + tl.arange(0, BLOCK_ROW_SIZE * kv_mul)
    mask_qrow = q_offset_row < n_kv_heads * kv_mul
    mask_q = mask_qrow[:, None] & mask_col[None, :]

    q_offset = head_dim * q_offset_row[:, None] + 2 * off_col[None, :]
    k_offset = head_dim * off_row[:, None] + 2 * off_col[None, :]
    q_even = tl.load(Q + q_offset, mask=mask_q, other=0.0)
    q_odd = tl.load(Q + q_offset + 1, mask=mask, other=0.0)
    k_even = tl.load(K + k_offset, mask=mask, other=0.0)
    k_odd = tl.load(K + k_offset + 1, mask=mask, other=0.0)

    q_new_even = q_even * fcr - q_odd * fci
    q_new_odd = q_even * fci + q_odd * fcr
    k_new_even = k_even * fcr - k_odd * fci
    k_new_odd = k_even * fci + k_odd * fcr

    tl.store(Q + q_offset, q_new_even, mask=mask_q)
    tl.store(Q + q_offset + 1, q_new_odd, mask=mask_q)
    tl.store(K + k_offset, k_new_even, mask=mask)
    tl.store(K + k_offset + 1, k_new_odd, mask=mask)


@triton.jit
def rope_1d_kernel_split_in_graph(Q, K, freq, pos, n_kv_heads: tl.constexpr, head_dim: tl.constexpr,
                                  kv_mul: tl.constexpr,
                                  BLOCK_COL_SIZE: tl.constexpr, BLOCK_ROW_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    off_row = pid * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)
    off_col = tl.arange(0, BLOCK_COL_SIZE)
    mask_row = off_row < n_kv_heads
    mask_col = off_col < head_dim // 2

    freq_c = tl.load(freq + off_col, mask=mask_col, other=0.0)
    freq_v = freq_c[None, :]
    p_freq = freq_v * pos
    fcr, fci = tl.cos(p_freq), tl.sin(p_freq)
    mask_k = mask_row[:, None] & mask_col[None, :]
    half = head_dim // 2

    q_offset_row = pid * BLOCK_ROW_SIZE * kv_mul + tl.arange(0, BLOCK_ROW_SIZE * kv_mul)
    mask_qrow = q_offset_row < n_kv_heads * kv_mul
    mask_q = mask_qrow[:, None] & mask_col[None, :]

    q_offset = head_dim * q_offset_row[:, None] + off_col[None, :]
    k_offset = head_dim * off_row[:, None] + off_col[None, :]
    # [batch * kv_mul, head_dim]
    ql = tl.load(Q + q_offset, mask=mask_q, other=0.0)
    qr = tl.load(Q + q_offset + half[None, :], mask=mask_q, other=0.0)
    # [batch, head_dim]
    kl = tl.load(K + k_offset, mask=mask_k, other=0.0)
    kr = tl.load(K + k_offset + half[None, :], mask=mask_k, other=0.0)

    qll = ql * fcr - qr * fci
    qrr = qr * fcr + ql * fci
    kll = kl * fcr - kr * fci
    krr = kr * fcr + kl * fci

    tl.store(Q + q_offset, qll, mask=mask_q)
    tl.store(Q + q_offset + half, qrr, mask=mask_q)
    tl.store(K + k_offset, kll, mask=mask_k)
    tl.store(K + k_offset + half, krr, mask=mask_k)


@triton.jit
def rope_1d_kernel_split(Q, K, freq, pos, n_kv_heads: tl.constexpr, head_dim: tl.constexpr, kv_mul: tl.constexpr,
                         BLOCK_COL_SIZE: tl.constexpr, BLOCK_ROW_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    off_row = pid * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)
    off_col = tl.arange(0, BLOCK_COL_SIZE)
    mask_row = off_row < n_kv_heads
    mask_col = off_col < head_dim // 2

    freq_c = tl.load(freq + off_col, mask=mask_col, other=0.0)
    freq_v = freq_c[None, :]
    p_freq = freq_v * pos
    fcr, fci = tl.cos(p_freq), tl.sin(p_freq)
    mask_k = mask_row[:, None] & mask_col[None, :]
    half = head_dim // 2

    q_offset_row = pid * BLOCK_ROW_SIZE * kv_mul + tl.arange(0, BLOCK_ROW_SIZE * kv_mul)
    mask_qrow = q_offset_row < n_kv_heads * kv_mul
    mask_q = mask_qrow[:, None] & mask_col[None, :]

    q_offset = head_dim * q_offset_row[:, None] + off_col[None, :]
    k_offset = head_dim * off_row[:, None] + off_col[None, :]
    # [batch * kv_mul, head_dim]
    ql = tl.load(Q + q_offset, mask=mask_q, other=0.0)
    qr = tl.load(Q + q_offset + half[None, :], mask=mask_q, other=0.0)
    # [batch, head_dim]
    kl = tl.load(K + k_offset, mask=mask_k, other=0.0)
    kr = tl.load(K + k_offset + half[None, :], mask=mask_k, other=0.0)

    qll = ql * fcr - qr * fci
    qrr = qr * fcr + ql * fci
    kll = kl * fcr - kr * fci
    krr = kr * fcr + kl * fci

    tl.store(Q + q_offset, qll, mask=mask_q)
    tl.store(Q + q_offset + half, qrr, mask=mask_q)
    tl.store(K + k_offset, kll, mask=mask_k)
    tl.store(K + k_offset + half, krr, mask=mask_k)


@triton.jit
def rope_1d_kernel_split_in_graph(Q, K, freq, pos_tensor, n_kv_heads: tl.constexpr, head_dim: tl.constexpr,
                                  kv_mul: tl.constexpr, l, seq_len,
                                  BLOCK_COL_SIZE: tl.constexpr, BLOCK_ROW_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pos = tl.load(pos_tensor)
    K = K + l * head_dim * n_kv_heads * seq_len + pos * head_dim * n_kv_heads
    off_row = pid * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)
    off_col = tl.arange(0, BLOCK_COL_SIZE)
    mask_row = off_row < n_kv_heads
    mask_col = off_col < head_dim // 2

    freq_c = tl.load(freq + off_col, mask=mask_col, other=0.0)
    freq_v = freq_c[None, :]
    p_freq = freq_v * pos
    fcr, fci = tl.cos(p_freq), tl.sin(p_freq)
    mask_k = mask_row[:, None] & mask_col[None, :]
    half = head_dim // 2

    q_offset_row = pid * BLOCK_ROW_SIZE * kv_mul + tl.arange(0, BLOCK_ROW_SIZE * kv_mul)
    mask_qrow = q_offset_row < n_kv_heads * kv_mul
    mask_q = mask_qrow[:, None] & mask_col[None, :]

    q_offset = head_dim * q_offset_row[:, None] + off_col[None, :]
    k_offset = head_dim * off_row[:, None] + off_col[None, :]
    # [batch * kv_mul, head_dim]
    ql = tl.load(Q + q_offset, mask=mask_q, other=0.0)
    qr = tl.load(Q + q_offset + half[None, :], mask=mask_q, other=0.0)
    # [batch, head_dim]
    kl = tl.load(K + k_offset, mask=mask_k, other=0.0)
    kr = tl.load(K + k_offset + half[None, :], mask=mask_k, other=0.0)

    qll = ql * fcr - qr * fci
    qrr = qr * fcr + ql * fci
    kll = kl * fcr - kr * fci
    krr = kr * fcr + kl * fci

    tl.store(Q + q_offset, qll, mask=mask_q)
    tl.store(Q + q_offset + half, qrr, mask=mask_q)
    tl.store(K + k_offset, kll, mask=mask_k)
    tl.store(K + k_offset + half, krr, mask=mask_k)


@triton.jit
def rope_1d_kernel_in_graph(Q, K, freq, pos_tensor, l, seq_len, n_kv_heads: tl.constexpr, head_dim: tl.constexpr,
                            kv_mul: tl.constexpr,
                            BLOCK_COL_SIZE: tl.constexpr, BLOCK_ROW_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pos = tl.load(pos_tensor)
    dim = head_dim * n_kv_heads
    K = K + l * dim * seq_len + pos * dim

    off_row = pid * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)
    off_col = tl.arange(0, BLOCK_COL_SIZE)
    mask_row = off_row < n_kv_heads
    mask_col = off_col < (head_dim // 2)

    freq_c = tl.load(freq + off_col, mask=mask_col, other=0.0)
    freq_v = freq_c[None, :]
    p_freq = freq_v * pos
    fcr, fci = tl.cos(p_freq), tl.sin(p_freq)
    mask = mask_row[:, None] & mask_col[None, :]

    q_offset = head_dim * off_row[:, None] * kv_mul + 2 * off_col[None, :]
    k_offset = head_dim * off_row[:, None] + 2 * off_col[None, :]
    q_even = tl.load(Q + q_offset, mask=mask, other=0.0)
    q_odd = tl.load(Q + q_offset + 1, mask=mask, other=0.0)
    k_even = tl.load(K + k_offset, mask=mask, other=0.0)
    k_odd = tl.load(K + k_offset + 1, mask=mask, other=0.0)

    q_new_even = q_even * fcr - q_odd * fci
    q_new_odd = q_even * fci + q_odd * fcr
    k_new_even = k_even * fcr - k_odd * fci
    k_new_odd = k_even * fci + k_odd * fcr

    tl.store(Q + q_offset, q_new_even, mask=mask)
    tl.store(Q + q_offset + 1, q_new_odd, mask=mask)
    tl.store(K + k_offset, k_new_even, mask=mask)
    tl.store(K + k_offset + 1, k_new_odd, mask=mask)
