import triton
import triton.language as tl


@triton.jit
def swiglu_1d_kernel(
        x_ptr,
        y_ptr,
        o_ptr,
        n_cols,
        BLOCK_COL_SIZE: tl.constexpr
):
    col_block = tl.program_id(0)

    col_offsets = col_block * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)

    col_mask = col_offsets < n_cols
    mask = col_mask

    offset = col_offsets
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    y = tl.load(y_ptr + offset, mask=mask, other=0.0)

    o = x * tl.sigmoid(x) * y

    tl.store(o_ptr + offset, o, mask=mask)
