import triton
import triton.language as tl

@triton.jit
def vec_add_kernel(X_ptr, Y_ptr, Out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(Y_ptr + offsets, mask=mask, other=0.0)
    z = x + y
    tl.store(Out_ptr + offsets, z, mask=mask)

