import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel_split_col(X, W, Y, stride_x, stride_y, M, N, eps: tl.constexpr, BLOCK_ROW_SIZE: tl.constexpr,
                             BLOCK_COL_SIZE: tl.constexpr):
    row = tl.program_id(0)
    block_row = tl.arange(0, BLOCK_ROW_SIZE)
    block_col = tl.arange(0, BLOCK_COL_SIZE)

    off_row = (row * BLOCK_ROW_SIZE + block_row)

    off_col = block_col

    mask_r = off_row[:, None] < M
    mask_c = off_col[None, :] < N

    x_ptr = off_row[:, None] * stride_x + off_col[None, :] * stride_y + X
    y_ptr = off_row[:, None] * stride_x + off_col[None, :] * stride_y + Y
    w_ptr = off_col[None, :] * stride_y + W

    mean = tl.zeros((BLOCK_ROW_SIZE,), dtype=tl.float32)
    for col in range(0, N, BLOCK_COL_SIZE):
        maks_c = off_col[None, :] + col < N
        x = tl.load(x_ptr, mask=mask_r & maks_c, other=0.0)
        mean += tl.sum(x * x, axis=1) / N
        x_ptr += BLOCK_COL_SIZE

    rms = tl.sqrt(mean + eps)[:, None]
    x_ptr = off_row[:, None] * stride_x + off_col[None, :] * stride_y + X
    for col in range(0, N, BLOCK_COL_SIZE):
        maks_c = off_col[None, :] + col < N
        w = tl.load(w_ptr, mask=mask_c, other=0.0)
        x = tl.load(x_ptr, mask=mask_r & maks_c, other=0.0)

        y = x / rms
        z = y * w
        tl.store(y_ptr, z, mask=mask_r & maks_c)
        x_ptr += BLOCK_COL_SIZE
        y_ptr += BLOCK_COL_SIZE
        w_ptr += BLOCK_COL_SIZE


@triton.jit
def rmsnorm_kernel_split_col_one_row(X, W, Y, N, eps: tl.constexpr,
                                     BLOCK_COL_SIZE: tl.constexpr, num_stages: tl.constexpr = 4,
                                     num_warps: tl.constexpr = 4):
    block_col = tl.arange(0, BLOCK_COL_SIZE)
    sum_sq = tl.zeros((1,), dtype=tl.float32)

    for col in tl.range(0, N, BLOCK_COL_SIZE, num_stages=num_stages):
        mask_c = block_col + col < N
        x = tl.load(X + col + block_col, mask=mask_c, other=0.0)
        sum_sq += tl.sum(x * x, axis=0, dtype=tl.float32)
    rms = tl.sqrt(sum_sq / N + eps)

    for col in tl.range(0, N, BLOCK_COL_SIZE, num_stages=num_stages):
        mask_c = block_col + col < N
        x = tl.load(X + col + block_col, mask=mask_c, other=0.0)
        w = tl.load(W + col + block_col, mask=mask_c, other=0.0)
        z = x / rms * w
        tl.store(Y + col + block_col, z, mask=mask_c)


@triton.jit
def rmsnorm_kernel_one_row_batch_in_graph_wp(X, W, Y, B, N, L, pos_tensor, seq_len, eps: tl.constexpr,
                                             BLOCK_COL_SIZE: tl.constexpr, BATCH_SIZE: tl.constexpr):
    pid_b = tl.program_id(0)
    pos = tl.load(pos_tensor)
    W = W + L * N
    X = X + L * seq_len * B * N + pos * B * N

    block_row = pid_b * BATCH_SIZE + tl.arange(0, BATCH_SIZE)
    block_col = tl.arange(0, BLOCK_COL_SIZE)
    sum_sq = tl.zeros((BATCH_SIZE,), dtype=tl.float32)

    mask_c = block_col < N
    mask_b = block_row < B
    mask = mask_b[:, None] & mask_c[None, :]
    bc = block_row[:, None] * N + block_col[None, :]
    x = tl.load(X + bc, mask=mask, other=0.0)
    w = tl.load(W + block_col, mask=mask_c, other=0.0)

    sum_sq += tl.sum(x * x, axis=1)
    rms = tl.sqrt(sum_sq / N + eps)
    z = x / rms[:, None] * w[None, :]

    tl.store(Y + bc, z, mask=mask)


@triton.jit
def rmsnorm_kernel_one_row_batch_in_graph(X, W, Y, B, N, L, eps: tl.constexpr,
                                          BLOCK_COL_SIZE: tl.constexpr, BATCH_SIZE: tl.constexpr):
    pid_b = tl.program_id(0)
    W = W + L * N
    block_row = pid_b * BATCH_SIZE + tl.arange(0, BATCH_SIZE)
    block_col = tl.arange(0, BLOCK_COL_SIZE)
    sum_sq = tl.zeros((BATCH_SIZE,), dtype=tl.float32)

    mask_c = block_col < N
    mask_b = block_row < B
    mask = mask_b[:, None] & mask_c[None, :]
    bc = block_row[:, None] * N + block_col[None, :]
    x = tl.load(X + bc, mask=mask, other=0.0)
    w = tl.load(W + block_col, mask=mask_c, other=0.0)

    sum_sq += tl.sum(x * x, axis=1)
    rms = tl.sqrt(sum_sq / N + eps)
    z = x / rms[:, None] * w[None, :]

    tl.store(Y + bc, z, mask=mask)

@triton.jit
def rmsnorm_kernel_one_row_batch(X, W, Y, B, N, eps: tl.constexpr,
                                 BLOCK_COL_SIZE: tl.constexpr, BATCH_SIZE: tl.constexpr):
    pid_b = tl.program_id(0)
    block_row = pid_b * BATCH_SIZE + tl.arange(0, BATCH_SIZE)
    block_col = tl.arange(0, BLOCK_COL_SIZE)
    sum_sq = tl.zeros((BATCH_SIZE,), dtype=tl.float32)

    mask_c = block_col < N
    mask_b = block_row < B
    mask = mask_b[:, None] & mask_c[None, :]
    bc = block_row[:, None] * N + block_col[None, :]
    x = tl.load(X + bc, mask=mask, other=0.0)
    w = tl.load(W + block_col, mask=mask_c, other=0.0)

    sum_sq += tl.sum(x * x, axis=1)
    rms = tl.sqrt(sum_sq / N + eps)
    z = x / rms[:, None] * w[None, :]

    tl.store(Y + bc, z, mask=mask)


@triton.jit
def rmsnorm_kernel_one_row(X, W, Y, N, eps: tl.constexpr,
                           BLOCK_COL_SIZE: tl.constexpr, num_warps: tl.constexpr = 4):
    block_col = tl.arange(0, BLOCK_COL_SIZE)
    sum_sq = tl.zeros((1,), dtype=tl.float32)

    mask_c = block_col < N
    x = tl.load(X + block_col, mask=mask_c, other=0.0)
    w = tl.load(W + block_col, mask=mask_c, other=0.0)

    sum_sq += tl.sum(x * x, axis=0)
    rms = tl.sqrt(sum_sq / N + eps)
    z = x / rms * w

    tl.store(Y + block_col, z, mask=mask_c)


@triton.jit
def rmsnorm_kernel_one_row_in_graph(X, W, Y, l, N, eps: tl.constexpr,
                                    BLOCK_COL_SIZE: tl.constexpr, num_warps: tl.constexpr = 4):
    W = W + l * N
    block_col = tl.arange(0, BLOCK_COL_SIZE)
    sum_sq = tl.zeros((1,), dtype=tl.float32)

    mask_c = block_col < N
    x = tl.load(X + block_col, mask=mask_c, other=0.0)
    w = tl.load(W + block_col, mask=mask_c, other=0.0)

    sum_sq += tl.sum(x * x, axis=0)
    rms = tl.sqrt(sum_sq / N + eps)
    z = x / rms * w

    tl.store(Y + block_col, z, mask=mask_c)


@triton.jit
def rmsnorm_kernel_one_row_split_col_in_graph(X, W, Y, l, N, eps: tl.constexpr,
                                              BLOCK_COL_SIZE: tl.constexpr, num_stages: tl.constexpr = 3):
    W = W + l * N
    block_col = tl.arange(0, BLOCK_COL_SIZE)
    sum_sq = tl.zeros((1,), dtype=tl.float32)

    for col in tl.range(0, N, BLOCK_COL_SIZE, num_stages=num_stages):
        mask_c = block_col + col < N
        x = tl.load(X + col + block_col, mask=mask_c, other=0.0)
        sum_sq += tl.sum(x * x, axis=0, dtype=tl.float32)
    rms = tl.sqrt(sum_sq / N + eps)

    for col in tl.range(0, N, BLOCK_COL_SIZE, num_stages=num_stages):
        mask_c = block_col + col < N
        x = tl.load(X + col + block_col, mask=mask_c, other=0.0)
        w = tl.load(W + col + block_col, mask=mask_c, other=0.0)
        z = x / rms * w
        tl.store(Y + col + block_col, z, mask=mask_c)


@triton.jit
def rmsnorm_kernel(X, W, Y, stride_x, stride_y, M, N, eps: tl.constexpr, BLOCK_ROW_SIZE: tl.constexpr,
                   COL_SIZE: tl.constexpr):
    row = tl.program_id(0)
    block_row = tl.arange(0, BLOCK_ROW_SIZE)
    block_col = tl.arange(0, COL_SIZE)

    off_row = (row * BLOCK_ROW_SIZE + block_row)

    off_col = block_col

    mask_r = off_row[:, None] < M
    mask_c = off_col[None, :] < N

    x_ptr = off_row[:, None] * stride_x + off_col[None, :] * stride_y + X
    y_ptr = off_row[:, None] * stride_x + off_col[None, :] * stride_y + Y
    w_ptr = off_col[None, :] * stride_y + W

    x = tl.load(x_ptr, mask=mask_r & mask_c, other=0.0)
    w = tl.load(w_ptr, mask=mask_c, other=0.0)

    rms = tl.sqrt(tl.sum(x * x, axis=1) / N + eps)
    y = x / rms[:, None]
    y = y * w
    tl.store(y_ptr, y, mask=mask_r & mask_c)
