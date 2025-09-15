import triton
import triton.language as tl


@triton.jit
def matvec_kernel(X_ptr, W_ptr, Out_ptr, D, N, BLOCK_D: tl.constexpr = 128, BLOCK_N: tl.constexpr = 128,
                  num_stages: tl.constexpr = 3):
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.bfloat16)

    if pid * BLOCK_D + BLOCK_D > D:
        mask_d = row_offsets < D

        for col_start in tl.range(0, N, BLOCK_N, num_stages=num_stages, warp_specialize=True):
            col_idx = col_start + tl.arange(0, BLOCK_N)
            if col_start + BLOCK_N > N:
                mask_n = col_idx < N
                x_block = tl.load(X_ptr + col_idx, mask=mask_n, other=0.0)

                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :],
                    mask=mask_n[None, :],
                    other=0.0
                )

                acc += tl.sum(w_block * x_block, axis=1)
            else:
                x_block = tl.load(X_ptr + col_idx)
                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :], )

                acc += tl.sum(w_block * x_block, axis=1)
        tl.store(Out_ptr + row_offsets, acc, mask=mask_d)
        return
    else:
        for col_start in tl.range(0, N, BLOCK_N, num_stages=num_stages, warp_specialize=True):
            col_idx = col_start + tl.arange(0, BLOCK_N)
            if col_start + BLOCK_N > N:
                mask_n = col_idx < N
                x_block = tl.load(X_ptr + col_idx, mask=mask_n, other=0.0)  # 1×BLOCK_N

                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :],
                    mask=mask_n[None, :],
                    other=0.0
                )

                acc += tl.sum(w_block * x_block, axis=1)
            else:
                x_block = tl.load(X_ptr + col_idx)
                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :], )

                acc += tl.sum(w_block * x_block, axis=1)

        tl.store(Out_ptr + row_offsets, acc)
        return


@triton.jit
def matvec_kernel_bf16(X_ptr, W_ptr, Out_ptr, D, N, BLOCK_D: tl.constexpr = 128, BLOCK_N: tl.constexpr = 128,
                       num_stages: tl.constexpr = 3):
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.bfloat16)

    if pid * BLOCK_D + BLOCK_D > D:
        mask_d = row_offsets < D

        for col_start in tl.range(0, N, BLOCK_N, num_stages=num_stages, warp_specialize=True):
            col_idx = col_start + tl.arange(0, BLOCK_N)
            if col_start + BLOCK_N > N:
                mask_n = col_idx < N
                x_block = tl.load(X_ptr + col_idx, mask=mask_n, other=0.0)

                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :],
                    mask=mask_n[None, :],
                    other=0.0
                )

                acc += tl.sum(w_block * x_block, axis=1, dtype=tl.bfloat16)
            else:
                x_block = tl.load(X_ptr + col_idx)
                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :], )

                acc += tl.sum(w_block * x_block, axis=1, dtype=tl.bfloat16)
        tl.store(Out_ptr + row_offsets, acc, mask=mask_d)
        return
    else:
        for col_start in tl.range(0, N, BLOCK_N, num_stages=num_stages, warp_specialize=True):
            col_idx = col_start + tl.arange(0, BLOCK_N)
            if col_start + BLOCK_N > N:
                mask_n = col_idx < N
                x_block = tl.load(X_ptr + col_idx, mask=mask_n, other=0.0)  # 1×BLOCK_N

                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :],
                    mask=mask_n[None, :],
                    other=0.0
                )

                acc += tl.sum(w_block * x_block, axis=1, dtype=tl.bfloat16)
            else:
                x_block = tl.load(X_ptr + col_idx)
                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :], )

                acc += tl.sum(w_block * x_block, axis=1, dtype=tl.bfloat16)

        tl.store(Out_ptr + row_offsets, acc)
        return


@triton.jit
def matvec_kernel_f32(X_ptr, W_ptr, Out_ptr, D, N, BLOCK_D: tl.constexpr = 128, BLOCK_N: tl.constexpr = 128,
                      num_stages: tl.constexpr = 3):
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    if pid * BLOCK_D + BLOCK_D > D:
        mask_d = row_offsets < D

        for col_start in tl.range(0, N, BLOCK_N, num_stages=num_stages, warp_specialize=True):
            col_idx = col_start + tl.arange(0, BLOCK_N)
            if col_start + BLOCK_N > N:
                mask_n = col_idx < N
                x_block = tl.load(X_ptr + col_idx, mask=mask_n, other=0.0)

                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :],
                    mask=mask_n[None, :],
                    other=0.0
                )

                acc += tl.sum(w_block * x_block, axis=1, dtype=tl.float32)
            else:
                x_block = tl.load(X_ptr + col_idx)
                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :], )

                acc += tl.sum(w_block * x_block, axis=1, dtype=tl.float32)
        tl.store(Out_ptr + row_offsets, acc, mask=mask_d)
        return
    else:
        for col_start in tl.range(0, N, BLOCK_N, num_stages=num_stages, warp_specialize=True):
            col_idx = col_start + tl.arange(0, BLOCK_N)
            if col_start + BLOCK_N > N:
                mask_n = col_idx < N
                x_block = tl.load(X_ptr + col_idx, mask=mask_n, other=0.0)  # 1×BLOCK_N

                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :],
                    mask=mask_n[None, :],
                    other=0.0
                )

                acc += tl.sum(w_block * x_block, axis=1, dtype=tl.float32)
            else:
                x_block = tl.load(X_ptr + col_idx)
                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :], )

                acc += tl.sum(w_block * x_block, axis=1, dtype=tl.float32)

        tl.store(Out_ptr + row_offsets, acc)
        return


@triton.jit
def matvec_kernel_in_graph_w_p(X_ptr, w_ptr, out_ptr, l, pos_tensor, D, N, S, T, BLOCK_D: tl.constexpr = 128,
                               BLOCK_N: tl.constexpr = 128):
    pid = tl.program_id(0)

    pos = tl.load(pos_tensor)

    W_ptr = l * N * D + w_ptr
    Out_ptr = out_ptr + l * S * T + pos * T
    row_offsets = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    if pid * BLOCK_D + BLOCK_D > D:
        mask_d = row_offsets < D

        for col_start in tl.range(0, N, BLOCK_N, num_stages=3, warp_specialize=True):
            col_idx = col_start + tl.arange(0, BLOCK_N)
            if col_start + BLOCK_N > N:
                mask_n = col_idx < N
                x_block = tl.load(X_ptr + col_idx, mask=mask_n, other=0.0)

                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :],
                    mask=mask_n[None, :],
                    other=0.0
                )

                acc += tl.sum(w_block * x_block, axis=1)
            else:
                x_block = tl.load(X_ptr + col_idx)
                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :], )

                acc += tl.sum(w_block * x_block, axis=1)
        tl.store(Out_ptr + row_offsets, acc, mask=mask_d)
        return
    else:
        for col_start in tl.range(0, N, BLOCK_N, num_stages=3, warp_specialize=True):
            col_idx = col_start + tl.arange(0, BLOCK_N)
            if col_start + BLOCK_N > N:
                mask_n = col_idx < N
                x_block = tl.load(X_ptr + col_idx, mask=mask_n, other=0.0)  # 1×BLOCK_N

                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :],
                    mask=mask_n[None, :],
                    other=0.0
                )

                acc += tl.sum(w_block * x_block, axis=1)
            else:
                x_block = tl.load(X_ptr + col_idx)
                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :], )

                acc += tl.sum(w_block * x_block, axis=1)

        tl.store(Out_ptr + row_offsets, acc)
        return


@triton.jit
def matvec_kernel_in_graph_w_p_f32(X_ptr, w_ptr, out_ptr, l, pos_tensor, D, N, S, T, BLOCK_D: tl.constexpr = 128,
                                   BLOCK_N: tl.constexpr = 128):
    pid = tl.program_id(0)

    pos = tl.load(pos_tensor)

    W_ptr = l * N * D + w_ptr
    Out_ptr = out_ptr + l * S * T + pos * T
    row_offsets = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    if pid * BLOCK_D + BLOCK_D > D:
        mask_d = row_offsets < D

        for col_start in tl.range(0, N, BLOCK_N, num_stages=3, warp_specialize=True):
            col_idx = col_start + tl.arange(0, BLOCK_N)
            if col_start + BLOCK_N > N:
                mask_n = col_idx < N
                x_block = tl.load(X_ptr + col_idx, mask=mask_n, other=0.0)

                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :],
                    mask=mask_n[None, :],
                    other=0.0
                )

                acc += tl.sum(w_block * x_block, axis=1, dtype=tl.float32)
            else:
                x_block = tl.load(X_ptr + col_idx)
                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :], )

                acc += tl.sum(w_block * x_block, axis=1, dtype=tl.float32)
        tl.store(Out_ptr + row_offsets, acc, mask=mask_d)
        return
    else:
        for col_start in tl.range(0, N, BLOCK_N, num_stages=3, warp_specialize=True):
            col_idx = col_start + tl.arange(0, BLOCK_N)
            if col_start + BLOCK_N > N:
                mask_n = col_idx < N
                x_block = tl.load(X_ptr + col_idx, mask=mask_n, other=0.0)  # 1×BLOCK_N

                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :],
                    mask=mask_n[None, :],
                    other=0.0
                )

                acc += tl.sum(w_block * x_block, axis=1, dtype=tl.float32)
            else:
                x_block = tl.load(X_ptr + col_idx)
                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :], )

                acc += tl.sum(w_block * x_block, axis=1, dtype=tl.float32)

        tl.store(Out_ptr + row_offsets, acc)
        return


@triton.jit
def matvec_kernel_in_graph(X_ptr, W_ptr, out_ptr, pos_tensor, D, N, BLOCK_D: tl.constexpr = 128,
                           BLOCK_N: tl.constexpr = 128):
    pos = tl.load(pos_tensor)

    Out_ptr = out_ptr + pos * D
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    if pid * BLOCK_D + BLOCK_D > D:
        mask_d = row_offsets < D

        for col_start in tl.range(0, N, BLOCK_N, num_stages=3, warp_specialize=True):
            col_idx = col_start + tl.arange(0, BLOCK_N)
            if col_start + BLOCK_N > N:
                mask_n = col_idx < N
                x_block = tl.load(X_ptr + col_idx, mask=mask_n, other=0.0)

                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :],
                    mask=mask_n[None, :],
                    other=0.0
                )

                acc += tl.sum(w_block * x_block, axis=1)
            else:
                x_block = tl.load(X_ptr + col_idx)
                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :], )

                acc += tl.sum(w_block * x_block, axis=1)
        tl.store(Out_ptr + row_offsets, acc, mask=mask_d)
        return
    else:
        for col_start in tl.range(0, N, BLOCK_N, num_stages=3, warp_specialize=True):
            col_idx = col_start + tl.arange(0, BLOCK_N)
            if col_start + BLOCK_N > N:
                mask_n = col_idx < N
                x_block = tl.load(X_ptr + col_idx, mask=mask_n, other=0.0)  # 1×BLOCK_N

                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :],
                    mask=mask_n[None, :],
                    other=0.0
                )

                acc += tl.sum(w_block * x_block, axis=1)
            else:
                x_block = tl.load(X_ptr + col_idx)
                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :], )

                acc += tl.sum(w_block * x_block, axis=1)

        tl.store(Out_ptr + row_offsets, acc)
        return


@triton.jit
def set_layer_tensor(tensor, l: int):
    tensor[0] = l


@triton.jit
def matvec_kernel_in_graph_wo_p(X_ptr, w_ptr, Out_ptr, l, D, N, BLOCK_D: tl.constexpr = 128,
                                BLOCK_N: tl.constexpr = 128):
    W_ptr = l * N * D + w_ptr

    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    if pid * BLOCK_D + BLOCK_D > D:
        mask_d = row_offsets < D

        for col_start in tl.range(0, N, BLOCK_N, num_stages=3, warp_specialize=True):
            col_idx = col_start + tl.arange(0, BLOCK_N)
            if col_start + BLOCK_N > N:
                mask_n = col_idx < N
                x_block = tl.load(X_ptr + col_idx, mask=mask_n, other=0.0)

                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :],
                    mask=mask_n[None, :],
                    other=0.0
                )

                acc += tl.sum(w_block * x_block, axis=1, dtype=tl.bfloat16)
            else:
                x_block = tl.load(X_ptr + col_idx)
                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :], )

                acc += tl.sum(w_block * x_block, axis=1, dtype=tl.bfloat16)
        tl.store(Out_ptr + row_offsets, acc, mask=mask_d)
        return
    else:
        for col_start in tl.range(0, N, BLOCK_N, num_stages=3, warp_specialize=True):
            col_idx = col_start + tl.arange(0, BLOCK_N)
            if col_start + BLOCK_N > N:
                mask_n = col_idx < N
                x_block = tl.load(X_ptr + col_idx, mask=mask_n, other=0.0)  # 1×BLOCK_N

                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :],
                    mask=mask_n[None, :],
                    other=0.0
                )

                acc += tl.sum(w_block * x_block, axis=1, dtype=tl.bfloat16)
            else:
                x_block = tl.load(X_ptr + col_idx)
                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :], )

                acc += tl.sum(w_block * x_block, axis=1, dtype=tl.bfloat16)

        tl.store(Out_ptr + row_offsets, acc)
        return


@triton.jit
def matvec_kernel_in_graph_wo_p_f32(X_ptr, w_ptr, Out_ptr, l, D, N, BLOCK_D: tl.constexpr = 128,
                                    BLOCK_N: tl.constexpr = 128):
    W_ptr = l * N * D + w_ptr

    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    if pid * BLOCK_D + BLOCK_D > D:
        mask_d = row_offsets < D

        for col_start in tl.range(0, N, BLOCK_N, num_stages=3, warp_specialize=True):
            col_idx = col_start + tl.arange(0, BLOCK_N)
            if col_start + BLOCK_N > N:
                mask_n = col_idx < N
                x_block = tl.load(X_ptr + col_idx, mask=mask_n, other=0.0)

                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :],
                    mask=mask_n[None, :],
                    other=0.0
                )

                acc += tl.sum(w_block * x_block, axis=1, dtype=tl.float32)
            else:
                x_block = tl.load(X_ptr + col_idx)
                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :], )

                acc += tl.sum(w_block * x_block, axis=1, dtype=tl.float32)
        tl.store(Out_ptr + row_offsets, acc, mask=mask_d)
        return
    else:
        for col_start in tl.range(0, N, BLOCK_N, num_stages=3, warp_specialize=True):
            col_idx = col_start + tl.arange(0, BLOCK_N)
            if col_start + BLOCK_N > N:
                mask_n = col_idx < N
                x_block = tl.load(X_ptr + col_idx, mask=mask_n, other=0.0)  # 1×BLOCK_N

                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :],
                    mask=mask_n[None, :],
                    other=0.0
                )

                acc += tl.sum(w_block * x_block, axis=1, dtype=tl.float32)
            else:
                x_block = tl.load(X_ptr + col_idx)
                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :], )

                acc += tl.sum(w_block * x_block, axis=1, dtype=tl.float32)

        tl.store(Out_ptr + row_offsets, acc)
        return


@triton.jit
def matmul_residual_kernel_in_graph(X_ptr, W_ptr, Out_ptr, L, D, N, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
                                    num_stages: tl.constexpr = 3):
    W_ptr = L * N * D + W_ptr
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    if pid * BLOCK_D + BLOCK_D > D:
        mask_d = row_offsets < D

        for col_start in tl.range(0, N, BLOCK_N, num_stages=3, warp_specialize=True):
            col_idx = col_start + tl.arange(0, BLOCK_N)
            if col_start + BLOCK_N > N:
                mask_n = col_idx < N
                x_block = tl.load(X_ptr + col_idx, mask=mask_n, other=0.0)

                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :],
                    mask=mask_n[None, :],
                    other=0.0
                )

                acc += tl.sum(w_block * x_block, axis=1)
            else:
                x_block = tl.load(X_ptr + col_idx)
                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :], )

                acc += tl.sum(w_block * x_block, axis=1)
        o = tl.load(Out_ptr + row_offsets, mask=mask_d)
        tl.store(Out_ptr + row_offsets, acc + o, mask=mask_d)
        return
    else:
        for col_start in tl.range(0, N, BLOCK_N, num_stages=3, warp_specialize=True):
            col_idx = col_start + tl.arange(0, BLOCK_N)
            if col_start + BLOCK_N > N:
                mask_n = col_idx < N
                x_block = tl.load(X_ptr + col_idx, mask=mask_n, other=0.0)  # 1×BLOCK_N

                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :],
                    mask=mask_n[None, :],
                    other=0.0
                )

                acc += tl.sum(w_block * x_block, axis=1)
            else:
                x_block = tl.load(X_ptr + col_idx)
                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :], )

                acc += tl.sum(w_block * x_block, axis=1)

        o = tl.load(Out_ptr + row_offsets)
        tl.store(Out_ptr + row_offsets, acc + o)
        return


@triton.jit
def matmul_residual_kernel(X_ptr, W_ptr, Out_ptr, D, N, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
                           num_stages: tl.constexpr = 3):
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    if pid * BLOCK_D + BLOCK_D > D:
        mask_d = row_offsets < D

        for col_start in tl.range(0, N, BLOCK_N, num_stages=3, warp_specialize=True):
            col_idx = col_start + tl.arange(0, BLOCK_N)
            if col_start + BLOCK_N > N:
                mask_n = col_idx < N
                x_block = tl.load(X_ptr + col_idx, mask=mask_n, other=0.0)

                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :],
                    mask=mask_n[None, :],
                    other=0.0
                )

                acc += tl.sum(w_block * x_block, axis=1)
            else:
                x_block = tl.load(X_ptr + col_idx)
                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :], )

                acc += tl.sum(w_block * x_block, axis=1)
        o = tl.load(Out_ptr + row_offsets, mask=mask_d)
        tl.store(Out_ptr + row_offsets, acc + o, mask=mask_d)
        return
    else:
        for col_start in tl.range(0, N, BLOCK_N, num_stages=3, warp_specialize=True):
            col_idx = col_start + tl.arange(0, BLOCK_N)
            if col_start + BLOCK_N > N:
                mask_n = col_idx < N
                x_block = tl.load(X_ptr + col_idx, mask=mask_n, other=0.0)  # 1×BLOCK_N

                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :],
                    mask=mask_n[None, :],
                    other=0.0
                )

                acc += tl.sum(w_block * x_block, axis=1)
            else:
                x_block = tl.load(X_ptr + col_idx)
                w_block = tl.load(
                    W_ptr + row_offsets[:, None] * N + col_idx[None, :], )

                acc += tl.sum(w_block * x_block, axis=1)

        o = tl.load(Out_ptr + row_offsets)
        tl.store(Out_ptr + row_offsets, acc + o)
        return
