import torch
import triton

from kernel.vec_add import vec_add_kernel
from ops.time_util import time_in_ms, store_time


def add_triton(x, y, out=None):
    t = time_in_ms()

    if out is None:
        out = torch.empty_like(x)

    grid = lambda META: (triton.cdiv(x.shape[0], META['BLOCK_SIZE']),)
    vec_add_kernel[grid](
        x, y, out,
        x.shape[0],
    )
    t2 = time_in_ms()
    store_time('add_triton', t2 - t)


def add(x, y, out=None):
    if x.is_cuda and y.is_cuda:
        return add_triton(x, y, out)
    t = time_in_ms()
    if out is None:
        out = torch.empty_like(x)
    out[:] = x + y
    t2 = time_in_ms()
    store_time('add_cpu', t2 - t)
    return out
