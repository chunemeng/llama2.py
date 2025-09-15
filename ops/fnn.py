import torch

from ops.add import add
from ops.cast import cast
from ops.matmul import matmul_triton_float32, matmul_triton_bfloat16, matmul, matmul_in_graph_f32, matmul_in_graph
from ops.swiglu import swiglu

import torch.nn.functional as F

from ops.time_util import time_in_ms, store_time


def fnn_module(state, l, transformer_weights, config):
    if config.fnn_use_torch:
        torch_fnn(state, l, transformer_weights)
    elif config.merge_matmul_check:
        fnn_merge(state, l, transformer_weights, config)
    else:
        fnn(state, l, transformer_weights)

def fnn_module_in_graph(state, l, transformer_weights, config):
    if config.fnn_use_torch:
        torch_fnn(state, l, transformer_weights)
    elif config.merge_matmul_check:
        fnn_merge(state, l, transformer_weights, config)
    else:
        fnn(state, l, transformer_weights)


def torch_fnn(state, l, transformer_weights):
    # speed: mv > matmul > linear
    t0 = time_in_ms()
    torch.mv(transformer_weights.w13[l], state.xb, out=state.hbm)
    half = state.hbm.shape[0] // 2
    F.silu(state.hbm[:half], inplace=True)
    state.x += torch.mv(transformer_weights.w2[l], state.hbm[:half].mul_(state.hbm[half:]), out=state.xb)
    t1 = time_in_ms()
    store_time('fnn_torch', t1 - t0)


def fnn(state, l, transformer_weights):
    t0 = time_in_ms()
    matmul(state.xb, transformer_weights.w1[l], output=state.hb)
    matmul(state.xb, transformer_weights.w3[l], output=state.hb2)
    hb = state.hb if state.hb.dtype == torch.float32 else cast(state.hb, torch.float32)
    hb2 = state.hb2

    swiglu(hb, hb2, output=hb)
    matmul_triton_bfloat16(hb, transformer_weights.w2[l], output=state.xb)
    add(state.x, state.xb, out=state.x)
    t1 = time_in_ms()
    store_time('fnn', t1 - t0)

def fnn_merge(state, l, transformer_weights, config):
    t0 = time_in_ms()
    matmul_triton_float32(state.xb, transformer_weights.w13[l], output=state.hbm32)
    hb = state.hbm32[:config.hidden_dim]
    hb2 = state.hbm32[config.hidden_dim:]

    swiglu(hb, hb2, output=hb)
    matmul_triton_bfloat16(hb, transformer_weights.w2[l], output=state.xb)
    add(state.x, state.xb, out=state.x)
    t1 = time_in_ms()
    store_time('fnn_merge', t1 - t0)

def fnn_merge_in_graph(state, l, transformer_weights, config):
    matmul_in_graph_f32(state.xb, transformer_weights.w13, l=l, o=state.hbm32)
    hb = state.hbm32[:config.hidden_dim]
    hb2 = state.hbm32[config.hidden_dim:]
    hbo = state.hbm[:config.hidden_dim]

    swiglu(hb, hb2, output=hbo)
    matmul_in_graph(hbo, transformer_weights.w2, l=l, o=state.xb)
    add(state.x, state.xb, out=state.x)