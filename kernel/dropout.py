import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_ROW_SIZE": 8, "BLOCK_COL_SIZE": 8}, num_warps=4),
        triton.Config({"BLOCK_ROW_SIZE": 16, "BLOCK_COL_SIZE": 16}, num_warps=4),
        triton.Config({"BLOCK_ROW_SIZE": 32, "BLOCK_COL_SIZE": 32}, num_warps=8),
        triton.Config({"BLOCK_ROW_SIZE": 64, "BLOCK_COL_SIZE": 64}, num_warps=2),
        triton.Config({"BLOCK_ROW_SIZE": 64, "BLOCK_COL_SIZE": 64}, num_warps=4),
        triton.Config({"BLOCK_ROW_SIZE": 32, "BLOCK_COL_SIZE": 128}, num_warps=8),
        triton.Config({"BLOCK_ROW_SIZE": 64, "BLOCK_COL_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_ROW_SIZE": 64, "BLOCK_COL_SIZE": 128}, num_warps=4),
    ],
    key=["n_rows", "n_cols"]
)
@triton.jit
def dropout_kernel(
        x_ptr,
        y_ptr,
        p,
        n_rows,
        n_cols,
        BLOCK_ROW_SIZE: tl.constexpr,
        BLOCK_COL_SIZE: tl.constexpr
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)

    row_offsets = row_block * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)
    col_offsets = col_block * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)

    row_mask = row_offsets < n_rows
    col_mask = col_offsets < n_cols
    mask2d = row_mask[:, None] & col_mask[None, :]

    offset = row_offsets[:, None] * n_cols + col_offsets[None, :]
    x = tl.load(
        x_ptr + offset,
        mask=mask2d,
        other=0.0,
    )

    rng = tl.rand(seed=0, offset=offset)
    m = (rng < p)
    y = x * m

    tl.store(
        y_ptr + offset,
        y,
        mask=mask2d
    )
