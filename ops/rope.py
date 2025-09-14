import math

import torch
import triton

from kernel.rope import rope_1d_kernel_in_graph, rope_1d_kernel, rope_1d_kernel_split, rope_1d_kernel_split_in_graph
from ops.time_util import time_in_ms, store_time


def rope_base(q, k, pos, dim, head_size, kv_dim=None):
    tl = time_in_ms()
    for i in range(0, dim, 2):
        head_dim = i % head_size
        freq = 1.0 / (10000 ** (float(head_dim) / head_size))
        val = pos * freq
        fcr = math.cos(val)
        fci = math.sin(val)
        rotn = 2 if i < kv_dim else 1
        for v_idx in range(rotn):
            vec = q if v_idx == 0 else k
            v0, v1 = vec[i].clone(), vec[i + 1].clone()
            vec[i] = v0 * fcr - v1 * fci
            vec[i + 1] = v0 * fci + v1 * fcr
    to = time_in_ms()
    store_time('rope_base', to - tl)


def rope_triton(q, k, freq, pos):
    tr = time_in_ms()
    n_heads = q.shape[0]
    head_size = q.shape[1]
    n_kv_heads = k.shape[0]

    if n_kv_heads > n_heads:
        kv_mul = n_kv_heads // n_heads
        assert kv_mul * n_heads == n_kv_heads
        grid = lambda META: (n_heads,)
        rope_1d_kernel[grid](
            k, q, freq, pos, n_heads, head_size, kv_mul,
            BLOCK_COL_SIZE=triton.next_power_of_2(head_size // 2), BLOCK_ROW_SIZE=2)
    else:
        kv_mul = n_heads // n_kv_heads
        assert kv_mul * n_kv_heads == n_heads
        grid = lambda META: (n_kv_heads,)
        rope_1d_kernel[grid](
            q, k, freq, pos, n_kv_heads, head_size, kv_mul,
            BLOCK_COL_SIZE=triton.next_power_of_2(head_size // 2), BLOCK_ROW_SIZE=2)

    tr2 = time_in_ms()
    store_time('rope_triton', tr2 - tr)


def rope_triton_in_graph(q, k, freq, pos, l, seq_len):
    tr = time_in_ms()
    n_heads = q.shape[0]
    head_size = q.shape[1]
    n_kv_heads = k.shape[0]

    if n_kv_heads > n_heads:
        kv_mul = n_kv_heads // n_heads
        assert kv_mul * n_heads == n_kv_heads
        grid = lambda META: (n_heads,)
        rope_1d_kernel_in_graph[grid](
            k, q, freq, pos, seq_len, l, n_heads, head_size, kv_mul,
            BLOCK_COL_SIZE=triton.next_power_of_2(head_size // 2), BLOCK_ROW_SIZE=2)
    else:
        kv_mul = n_heads // n_kv_heads
        assert kv_mul * n_kv_heads == n_heads
        grid = lambda META: (n_kv_heads,)
        rope_1d_kernel_in_graph[grid](
            q, k, freq, pos, seq_len, l, n_heads, head_size, kv_mul,
            BLOCK_COL_SIZE=triton.next_power_of_2(head_size // 2), BLOCK_ROW_SIZE=2)
    tr2 = time_in_ms()
    store_time('rope_triton_in_graph', tr2 - tr)


def rope_opt(q, k, pos, freq):
    # q [n_heads, head_size]
    # k [n_heads, head_size]

    tt = time_in_ms()

    if q.is_cuda and k.is_cuda and freq.is_cuda:
        return rope_triton(q, k, freq, pos)

    # [head_dim]
    v = pos * freq

    val = v[None, :]  # [1, head_dim]

    fcr = torch.cos(val)

    fci = torch.sin(val)
    #
    q_even = q[:, 0::2]
    k_even = k[:, 0::2]
    q_odd = q[:, 1::2]
    k_odd = k[:, 1::2]

    q_new_even = q_even * fcr - q_odd * fci
    k_new_even = k_even * fcr - k_odd * fci
    k_new_odd = k_even * fci + k_odd * fcr
    q_new_odd = q_even * fci + q_odd * fcr

    tz = time_in_ms()
    q[:, 0::2] = q_new_even
    q[:, 1::2] = q_new_odd
    k[:, 0::2] = k_new_even
    k[:, 1::2] = k_new_odd
    to = time_in_ms()
    store_time('rope_opt', to - tt)


def rope_triton_split(q, k, freq, pos):
    tr = time_in_ms()
    n_heads = q.shape[0]
    head_size = q.shape[1]
    n_kv_heads = k.shape[0]

    if n_kv_heads > n_heads:
        kv_mul = n_kv_heads // n_heads
        assert kv_mul * n_heads == n_kv_heads
        grid = lambda META: (n_heads,)
        rope_1d_kernel_split[grid](
            k, q, freq, pos, n_heads, head_size, kv_mul,
            BLOCK_COL_SIZE=triton.next_power_of_2(head_size // 2), BLOCK_ROW_SIZE=1)
    else:
        kv_mul = n_heads // n_kv_heads
        assert kv_mul * n_kv_heads == n_heads
        grid = lambda META: (n_kv_heads,)
        rope_1d_kernel_split[grid](
            q, k, freq, pos, n_kv_heads, head_size, kv_mul,
            BLOCK_COL_SIZE=triton.next_power_of_2(head_size // 2), BLOCK_ROW_SIZE=1)

    tr2 = time_in_ms()
    store_time('rope_triton_split', tr2 - tr)


def rope_split_in_graph(q, k, freq, n_heads, head_size, n_kv_heads, pos, l, seq_len):
    tr = time_in_ms()

    kv_mul = n_heads // n_kv_heads
    assert kv_mul * n_kv_heads == n_heads
    grid = lambda META: (n_kv_heads,)
    rope_1d_kernel_split_in_graph[grid](
        q, k, freq, pos, n_kv_heads, head_size, kv_mul, l, seq_len,
        BLOCK_COL_SIZE=triton.next_power_of_2(head_size // 2), BLOCK_ROW_SIZE=1)

    tr2 = time_in_ms()
    store_time('rope_triton_split_in_graph', tr2 - tr)


def rope_split(q, k, pos, freq):
    # q [n_heads, head_size]
    # k [n_kv_heads, head_size]

    tt = time_in_ms()

    head_size = q.shape[1]

    if q.is_cuda and k.is_cuda and freq.is_cuda:
        return rope_triton_split(q, k, freq, pos)

    # [head_dim]
    f = freq.repeat(2)
    v = pos * f

    val = v[None, :]  # [1, head_dim]

    fcr = torch.cos(val)

    fci = torch.sin(val)

    half_head_size = head_size // 2
    q_left = q[:, :half_head_size]
    k_left = k[:, :half_head_size]
    q_right = q[:, half_head_size:]
    k_right = k[:, half_head_size:]

    qn = torch.cat([-q_right, q_left], dim=1)
    kn = torch.cat([-k_right, k_left], dim=1)

    q[:] = q * fcr + qn * fci
    k[:] = k * fcr + kn * fci

    to = time_in_ms()
    store_time('rope_split', to - tt)
