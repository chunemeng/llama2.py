import torch
import triton
import torch.nn.functional as F

from kernel.swiglu import swiglu_1d_kernel
from ops.time_util import store_time, time_in_ms


def swiglu_triton(x1, x2, output=None):
    t = time_in_ms()
    if output is None:
        output = torch.empty_like(x1)
    grid = lambda META: (triton.cdiv(x1.shape[0], META['BLOCK_COL_SIZE']),)
    swiglu_1d_kernel[grid](
        x1, x2, output,
        x1.shape[0],
        BLOCK_COL_SIZE=128
    )
    t2 = time_in_ms()
    store_time('swiglu_triton', t2 - t)
    return output


def swiglu(x1, x2, output=None):
    if x1.is_cuda and x2.is_cuda:
        return swiglu_triton(x1, x2, output)
    t = time_in_ms()
    res = F.silu(x1) * x2
    if output is not None:
        output[:] = res
        res = output
    t2 = time_in_ms()
    # assert torch.allclose(res, out, atol=1e-6), f"swiglu mismatch {torch.max(torch.abs(res - out))}"
    store_time('swiglu_cpu', t2 - t)
    return res
