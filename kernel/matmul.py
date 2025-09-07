import triton
import triton.language as tl


@triton.jit
def matvec_kernel(X_ptr, W_ptr, Out_ptr, D, N, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr):
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
def matmul_residual_kernel(X_ptr, W_ptr, Out_ptr, D, N, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr, num_stages: tl.constexpr = 3):
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

