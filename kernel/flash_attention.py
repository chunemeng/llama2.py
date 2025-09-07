import math

import torch
import triton
import triton.language as tl

@triton.jit
def flash_attention_kernel_1d(Q, K, V, output, N, dim, scale, BLOCK_N: tl.constexpr,
                              HEAD_DIM: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    d = (pid_h * HEAD_DIM + tl.arange(0, HEAD_DIM))

    block_n = tl.arange(0, BLOCK_N)
    block_q = Q + d
    block_k = K + (block_n[:, None] * dim + d[None, :])
    block_v = V + (block_n[:, None] * dim + d[None, :])

    q_tile = tl.load(block_q, other=0.0)

    m = tl.zeros((BLOCK_N,), dtype=tl.float32) - float('inf')
    de = tl.zeros((BLOCK_N,), dtype=tl.float32)
    o = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)

    for n in range(0, N, BLOCK_N):
        mask = (block_n[:, None] + n < N)
        k_tile = tl.load(block_k, mask=mask, other=0.0)
        v_tile = tl.load(block_v, mask=mask, other=0.0)
        x = tl.sum(q_tile * k_tile, axis=1)

        mask = (n_ptr[:, None] < N) & (block_n[None, :] + n < N)
        x_max_masked = tl.where(mask, x, float('-inf'))

        x_max = tl.max(x_max_masked, axis=1)

        maxs = tl.maximum(m, x_max)

        diff_exp = tl.exp(m - maxs)

        m_exp = tl.exp(m - maxs)

        x_diff_exp = tl.exp(x - maxs[:, None])

        x_diff_exp_masked = tl.where(mask, x_diff_exp, 0.0)

        d_n = de * diff_exp + tl.sum(x_diff_exp_masked, axis=1)


        o_t = tl.sum(x_diff_exp_masked[:, :, None] * v_tile[None, :, :], axis=1)

        o = o_t + o * m_exp[:, None]
        m = maxs
        de = d_n
        block_k += BLOCK_N * dim
        block_v += BLOCK_N * dim

    o /= de[:, None] * scale
    out_ptr = output + d

    tl.store(out_ptr, o)

@triton.jit
def flash_attention_kernel_1d_corner(Q, K, V, output, N, h_dim, scale, BLOCK_N: tl.constexpr,
                              BLOCK_D: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    n_ptr = (pid_m * BLOCK_N + tl.arange(0, BLOCK_N))

    mask_d = tl.arange(0, BLOCK_D) < HEAD_DIM
    d = (pid_h * HEAD_DIM + tl.arange(0, BLOCK_D))

    block_n = tl.arange(0, BLOCK_N)
    block_q = Q + d
    block_k = K + (block_n[:, None] * h_dim + d[None, :])
    block_v = V + (block_n[:, None] * h_dim + d[None, :])

    q_tile = tl.load(block_q, mask=mask_d, other=0.0)

    m = tl.zeros((BLOCK_N,), dtype=tl.float32) - float('inf')
    de = tl.zeros((BLOCK_N,), dtype=tl.float32)
    o = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    for n in range(0, N, BLOCK_N):
        mask = (mask_d[None, :]) & (block_n[:, None] + n < N)
        k_tile = tl.load(block_k, mask=mask, other=0.0)
        v_tile = tl.load(block_v, mask=mask, other=0.0)
        x = tl.sum(q_tile * k_tile, axis=2)

        mask = (n_ptr[:, None] < N) & (block_n[None, :] + n < N)
        x_max_masked = tl.where(mask, x, float('-inf'))

        x_max = tl.max(x_max_masked, axis=1)

        maxs = tl.maximum(m, x_max)

        diff_exp = tl.exp(m - maxs)

        m_exp = tl.exp(m - maxs)

        x_diff_exp = tl.exp(x - maxs[:, None])

        x_diff_exp_masked = tl.where(mask, x_diff_exp, 0.0)

        d_n = de * diff_exp + tl.sum(x_diff_exp_masked, axis=1)

        if (BLOCK_N > 16) & (BLOCK_D > 16):
            o_t = tl.dot(x_diff_exp_masked, v_tile)
        else:
            o_t = tl.sum(x_diff_exp_masked[:, :, None] * v_tile[None, :, :], axis=1)

        o = o_t + o * m_exp[:, None]
        m = maxs
        de = d_n
        block_k += BLOCK_N * h_dim
        block_v += BLOCK_N * h_dim

    o /= de[:, None] * scale
    out_ptr = output + (n_ptr[:, None] * h_dim + d[None, :])

    tl.store(out_ptr, o, mask=(n_ptr[:, None] < N) & mask_d)


@triton.jit
def flash_attention_kernel(Q, K, V, output, N, d_model, h, BLOCK_N: tl.constexpr,
                           BLOCK_D: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    sqrt_d = HEAD_DIM ** 0.5
    n_ptr = (pid_m * BLOCK_N + tl.arange(0, BLOCK_N))

    mask_d = tl.arange(0, BLOCK_D)[None, :] < HEAD_DIM
    d = (pid_h * HEAD_DIM + tl.arange(0, BLOCK_D))

    block_n = tl.arange(0, BLOCK_N)
    block_q = Q + (n_ptr[:, None] * d_model + d[None, :])
    block_k = K + (block_n[:, None] * d_model + d[None, :])
    block_v = V + (block_n[:, None] * d_model + d[None, :])

    q_tile = tl.load(block_q, mask=(n_ptr[:, None] < N) & mask_d, other=0.0)

    m = tl.zeros((BLOCK_N,), dtype=tl.float32) - float('inf')
    de = tl.zeros((BLOCK_N,), dtype=tl.float32)
    o = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    for n in range(0, N, BLOCK_N):
        mask = mask_d & (block_n[:, None] + n < N)
        k_tile = tl.load(block_k, mask=mask, other=0.0)
        v_tile = tl.load(block_v, mask=mask, other=0.0)
        # x = tl.dot(q_tile, k_tile.T)
        if (BLOCK_N > 16) & (BLOCK_D > 16):
            x = tl.dot(q_tile, k_tile.T)
        else:
            x = tl.sum(q_tile[:, None, :] * k_tile[None, :, :], axis=2)

        mask = (n_ptr[:, None] < N) & (block_n[None, :] + n < N)
        x_max_masked = tl.where(mask, x, float('-inf'))

        x_max = tl.max(x_max_masked, axis=1)

        maxs = tl.maximum(m, x_max)

        diff_exp = tl.exp(m - maxs)

        m_exp = tl.exp(m - maxs)

        x_diff_exp = tl.exp(x - maxs[:, None])

        x_diff_exp_masked = tl.where(mask, x_diff_exp, 0.0)

        d_n = de * diff_exp + tl.sum(x_diff_exp_masked, axis=1)

        if (BLOCK_N > 16) & (BLOCK_D > 16):
            o_t = tl.dot(x_diff_exp_masked, v_tile)
        else:
            o_t = tl.sum(x_diff_exp_masked[:, :, None] * v_tile[None, :, :], axis=1)

        o = o_t + o * m_exp[:, None]
        m = maxs
        de = d_n
        block_k += BLOCK_N * d_model
        block_v += BLOCK_N * d_model

    o /= de[:, None] * sqrt_d
    out_ptr = output + (n_ptr[:, None] * d_model + d[None, :])

    tl.store(out_ptr, o, mask=(n_ptr[:, None] < N) & mask_d)


def next_power_of_2(x):
    return triton.next_power_of_2(x)

def is_power_of_2(x):
    return (x & (x - 1)) == 0 and x != 0


def flash_attention_triton(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, N: int,
                           d_model: int,
                           h: int):
    BLOCK_D = triton.next_power_of_2(d_model // h)
    grid = lambda META: (triton.cdiv(N, META['BLOCK_N']), h)
    flash_attention_kernel[grid](Q, K, V, output, N, d_model, h, BLOCK_N=4, BLOCK_D=BLOCK_D, HEAD_DIM=d_model // h)
    pass


def verify_flash_attention(num_tests=25):
    for test_id in range(1, num_tests + 1):
        N = torch.randint(1, 256, (1,)).item()
        d_model = triton.next_power_of_2(torch.randint(16, 256, (1,)).item())
        h = 2 ** torch.randint(1, int(math.log2(d_model)) + 1, (1,)).item()
        HEAD_DIM = d_model // h

        Q = torch.randn(N, d_model, device='cuda', dtype=torch.float32)
        K = torch.randn(N, d_model, device='cuda', dtype=torch.float32)
        V = torch.randn(N, d_model, device='cuda', dtype=torch.float32)
        Out_triton = torch.zeros((N, d_model), device='cuda', dtype=torch.float32)

        # Triton FlashAttention
        flash_attention_triton(Q, K, V, Out_triton, N, d_model, h=h)

        Qh = Q.view(N, h, HEAD_DIM)
        Kh = K.view(N, h, HEAD_DIM)
        Vh = V.view(N, h, HEAD_DIM)
        Out_heads = torch.zeros_like(Qh)
        for i in range(h):
            attn = torch.matmul(Qh[:, i, :], Kh[:, i, :].T)
            attn = torch.softmax(attn, dim=1)
            Out_heads[:, i, :] = torch.matmul(attn, Vh[:, i, :]) * (HEAD_DIM ** -0.5)
        Out_ref = Out_heads.view(N, d_model)

        # 验证结果
        if torch.allclose(Out_triton, Out_ref, rtol=1e-4, atol=1e-5):
            print(f"✅ Test {test_id}: 结果一致, N={N}, d_model={d_model}, h={h}")
        else:
            diff = (Out_triton - Out_ref).abs().max()
            print(f"❌ Test {test_id}: 最大误差={diff.item()}, N={N}, d_model={d_model}, h={h}")

# -----------------------------
# 运行验证
# -----------------------------
# verify_flash_attention()
