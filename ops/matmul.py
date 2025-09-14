import torch
import triton

from kernel.matmul import matvec_kernel, matvec_kernel_in_graph_wo_p, matvec_kernel_in_graph_w_p, \
    matmul_residual_kernel, matmul_residual_kernel_in_graph, matvec_kernel_bf16, matvec_kernel_f32
from ops.add import add
from ops.time_util import time_in_ms, global_time_dict, store_time


def matmul_triton(x, w, output=None):
    """
    w: [d, n]
    x: [n]
    return: xout [d]
    """
    t = time_in_ms()
    if output is None:
        output = torch.empty(w.shape[0], device=w.device, dtype=w.dtype)
    grid = lambda META: (triton.cdiv(w.shape[0], META['BLOCK_D']),)
    shape = (w.shape[0], w.shape[1])
    match shape:
        case (1024, 1024):
            matvec_kernel[grid](x, w, output,
                                w.shape[0], w.shape[1], BLOCK_D=64, BLOCK_N=128, num_stages=3)
        case (6144, 1024):
            matvec_kernel[grid](x, w, output,
                                w.shape[0], w.shape[1], BLOCK_D=64, BLOCK_N=128, num_stages=3)
        case _:
            matvec_kernel[grid](x, w, output,
                                w.shape[0], w.shape[1])

    # print(f'shape x: {w.shape}, best_config: {matvec_kernel.best_config}')
    t2 = time_in_ms()
    store_time('matmul_triton' + str(w.shape), t2 - t)
    return output


def matmul_triton_float32(x, w, output=None):
    """
    w: [d, n]
    x: [n]
    return: xout [d]
    """
    t = time_in_ms()
    if output is None:
        output = torch.empty(w.shape[0], device=w.device, dtype=w.dtype)
    grid = lambda META: (triton.cdiv(w.shape[0], META['BLOCK_D']),)

    matvec_kernel_f32[grid](x, w, output,
                          w.shape[0], w.shape[1], BLOCK_D=64, BLOCK_N=128, num_stages=3)

    # print(f'shape x: {w.shape}, best_config: {matvec_kernel.best_config}')
    t2 = time_in_ms()
    store_time('matmul_triton' + str(w.shape), t2 - t)
    return output


def matmul_triton_bfloat16(x, w, output=None):
    """
    w: [d, n]
    x: [n]
    return: xout [d]
    """
    t = time_in_ms()
    if output is None:
        output = torch.empty(w.shape[0], device=w.device, dtype=w.dtype)
    grid = lambda META: (triton.cdiv(w.shape[0], META['BLOCK_D']),)

    matvec_kernel_bf16[grid](x, w, output,
                             w.shape[0], w.shape[1], BLOCK_D=64, BLOCK_N=128, num_stages=3)

    # print(f'shape x: {w.shape}, best_config: {matvec_kernel.best_config}')
    t2 = time_in_ms()
    store_time('matmul_triton' + str(w.shape), t2 - t)
    return output


def matmul(x, w, output=None):
    """
    w: [d, n]
    x: [n]
    return: xout [d]
    """
    if x.is_cuda and w.is_cuda:
        return matmul_triton(x, w, output)
    t = time_in_ms()
    res = torch.mv(w, x, out=output)
    t2 = time_in_ms()
    store_time('matmul_cpu', t2 - t)
    return res


def matmul_in_graph(x, w, o, l, p=None):
    t = time_in_ms()
    grid = lambda META: (triton.cdiv(w.shape[1], META['BLOCK_D']),)

    if p is None:
        matvec_kernel_in_graph_wo_p[grid](x, w, o, l,
                                          w.shape[1], w.shape[2])
    else:
        matvec_kernel_in_graph_w_p[grid](x, w, o, l, p,
                                         w.shape[1], w.shape[2], o.shape[1], o.shape[2])
    t2 = time_in_ms()
    store_time('matmul_triton', t2 - t)


def matmul_residual_triton(x, w, output):
    t = time_in_ms()
    grid = lambda META: (triton.cdiv(w.shape[0], META['BLOCK_D']),)
    matmul_residual_kernel[grid](x, w, output,
                                 w.shape[0], w.shape[1], BLOCK_D=64, BLOCK_N=64, num_stages=3)
    t2 = time_in_ms()
    store_time('matmul_residual_triton', t2 - t)
    return output


def matmul_residual_in_graph(x, w, l, output):
    t = time_in_ms()
    grid = lambda META: (triton.cdiv(w.shape[1], META['BLOCK_D']),)
    matmul_residual_kernel_in_graph[grid](x, w, output, l,
                                          w.shape[1], w.shape[2], BLOCK_D=64, BLOCK_N=64, num_stages=3)
    t2 = time_in_ms()
    store_time('matmul_residual_in_graph', t2 - t)
    return output


def matmul_residual(x, w, output, tmp=None):
    """
    w: [d, n]
    x: [n]
    return: xout [d]
    """
    if x.is_cuda and w.is_cuda:
        return matmul_residual_triton(x, w, output)

    if tmp is None:
        tmp = torch.empty_like(x)
    t = time_in_ms()
    matmul(x, w, output=tmp)
    add(output, tmp, out=output)
    t2 = time_in_ms()
    # assert torch.allclose(oo, output, atol=1e-6), f"matmul_residual mismatch {torch.max(torch.abs(oo - output))}"
    store_time('matmul_residual_cpu', t2 - t)
    return output
