import torch
import triton
from triton.language.extra.cuda import num_warps

from kernel.rmsnorm import rmsnorm_kernel_split_col, rmsnorm_kernel_one_row, rmsnorm_kernel_one_row_in_graph, \
    rmsnorm_kernel_one_row_batch, rmsnorm_kernel_one_row_batch_in_graph, rmsnorm_kernel_one_row_batch_in_graph_wp, \
    rmsnorm_kernel_split_col_one_row
from ops.time_util import store_time, time_in_ms


def rmsnorm_triton_batch(x, weight, out, eps=1e-6):
    """
    x: [b, dim]
    weight: [dim]
    return: [b, dim]
    """
    t = time_in_ms()
    b = x.shape[0]
    d = x.shape[1]
    if out is None:
        out = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(b, META['BATCH_SIZE']),)
    if b == 16:
        rmsnorm_kernel_one_row_batch[grid](
            x, weight, out, b,
            d,
            eps=eps,
            BLOCK_COL_SIZE=triton.next_power_of_2(d),
            BATCH_SIZE=8,
            num_warps=8
        )
    else:
        rmsnorm_kernel_one_row_batch[grid](
            x, weight, out, b,
            d,
            eps=eps,
            BLOCK_COL_SIZE=triton.next_power_of_2(d),
            BATCH_SIZE=4,
            num_warps=4
        )

    # print(f"b: {b}, d: {d}, use_one_row: {rmsnorm_kernel_one_row_batch.best_config}")

    t2 = time_in_ms()
    store_time('rmsnorm_triton_batch', t2 - t)
    return out


def rmsnorm_config(d):
    match d:
        case 1024:
            return {
                'BLOCK_COL_SIZE': 256,
                'num_warps': 4,
                'num_stages': 3
            }
        case _:
            return {
                'BLOCK_COL_SIZE': triton.next_power_of_2(d),
                'num_warps': 2,
                'num_stages': 3
            }


def rmsnorm_triton(x, weight, out, eps=1e-6):
    """
    x: [dim]
    weight: [dim]
    return: [dim]
    """
    t = time_in_ms()
    d = x.shape[0]
    if out is None:
        out = torch.empty_like(x)
    grid = lambda META: (1,)
    use_one_row = d <= 1024
    assert use_one_row, "only support d <= 1024"
    rmsnorm_kernel_one_row[grid](
        x, weight, out,
        d,
        eps=eps,
        BLOCK_COL_SIZE=triton.next_power_of_2(d),
        num_warps=2
    )
    # cfg = rmsnorm_config(d)
    # grid = lambda META: (triton.cdiv(d, META['BLOCK_COL_SIZE']),)
    # rmsnorm_kernel_split_col_one_row[grid](
    #     x, weight, out,
    #     d,
    #     eps=eps,
    #     BLOCK_COL_SIZE=cfg['BLOCK_COL_SIZE'],
    #     num_warps=cfg['num_warps'],
    #     num_stages=cfg['num_stages']
    # )
    # print(f'shape {d}, use split col one row: {rmsnorm_kernel_split_col_one_row.best_config}')

    # rmsnorm_kernel_split_col_one_row[grid](
    #     x, weight, out,
    #     d,
    #     eps=eps,
    #     BLOCK_COL_SIZE=256,
    #     num_stages=4
    # )
    # print(rmsnorm_kernel_split_col_one_row.best_config)
    t2 = time_in_ms()
    store_time('rmsnorm_triton', t2 - t)
    return out


def rmsnorm_in_graph(x, weight, out, l, eps=1e-6):
    """
    x: [dim]
    weight: [dim]
    return: [dim]
    """
    t = time_in_ms()
    d = x.shape[0]
    if out is None:
        out = torch.empty_like(x)
    grid = lambda META: (1,)

    rmsnorm_kernel_one_row_in_graph[grid](
        x, weight, out, l,
        d,
        eps=eps,
        BLOCK_COL_SIZE=1024,
        num_warps=2
    )

    t2 = time_in_ms()
    store_time('rmsnorm_triton_in_graph', t2 - t)
    return out


def rmsnorm_batch(x, weight, out=None, eps=1e-6):
    # x: [B, dim]
    # w: [dim]
    # o: [B, dim]
    if x.is_cuda and weight.is_cuda:
        return rmsnorm_triton_batch(x, weight, out, eps)
    t = time_in_ms()
    if out is None:
        out = torch.empty_like(x)
    mean_square = x.pow(2).mean(dim=1, keepdim=True)  # [B, 1]
    norm_factor = torch.rsqrt(mean_square + eps)  # [B, 1]
    out[:] = x * norm_factor * weight[None, :]  # [B, dim]
    t2 = time_in_ms()
    store_time('rmsnorm_cpu_batch', t2 - t)
    return out


def rmsnorm_batch_in_graph(x, weight, l, head_num, head_size, out=None, pos=None, seq_len=None, eps=1e-6):
    # x: [B, dim]
    # w: [dim]
    # o: [B, dim]
    t = time_in_ms()
    if out is None:
        out = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(head_num, META['BATCH_SIZE']),)
    use_one_row = head_size <= 1024
    if pos is None:
        rmsnorm_kernel_one_row_batch_in_graph[grid](
            x, weight, out, head_num,
            head_size, l,
            eps=eps,
            BLOCK_COL_SIZE=triton.next_power_of_2(head_size),
            BATCH_SIZE=4
        )
    else:
        rmsnorm_kernel_one_row_batch_in_graph_wp[grid](
            x, weight, out, head_num,
            head_size, l, pos, seq_len,
            eps=eps,
            BLOCK_COL_SIZE=triton.next_power_of_2(head_size),
            BATCH_SIZE=4
        )

    t2 = time_in_ms()
    store_time('rmsnorm_triton_batch_in_graph', t2 - t)
    return out


def rmsnorm(x, weight, out=None, eps=1e-6):
    if x.is_cuda and weight.is_cuda:
        return rmsnorm_triton(x, weight, out, eps)
    t = time_in_ms()
    res = weight * x / torch.sqrt(torch.mean(x ** 2) + eps)
    # assert torch.allclose(res, oo, atol=1e-6), f"rmsnorm mismatch {torch.max(torch.abs(res - oo))}"
    if out is not None:
        out[:] = res
    t2 = time_in_ms()
    store_time('rmsnorm_cpu', t2 - t)
    return res
