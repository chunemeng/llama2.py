import torch

from kernel.flash_attention import next_power_of_2, is_power_of_2, flash_attention_kernel_1d, \
    flash_attention_kernel_1d_corner
from ops.softmax import softmax
from ops.time_util import time_in_ms, store_time


def batch_mha(q, state, l, pos, n_heads, head_size, transformer_weights):
    tl = time_in_ms()
    q_heads = q.view(n_heads, head_size)
    K_heads = state.key_cache[l, :pos + 1].reshape(pos + 1, n_heads, head_size).permute(1, 0,
                                                                                        2)  # [H, S, D]
    V_heads = state.value_cache[l, :pos + 1].reshape(pos + 1, n_heads, head_size).permute(1, 0,
                                                                                          2)  # [H, S, D]
    attd = torch.bmm(K_heads, q_heads.unsqueeze(-1)).squeeze(-1) / transformer_weights.scale

    attd = softmax(attd, dim=1)
    out_heads = torch.bmm(attd.unsqueeze(1), V_heads).squeeze(1)  # [H, D]
    state.atten_out[:] = out_heads.reshape(-1)
    to = time_in_ms()
    store_time('attention_bmm', to - tl)


def flash_attention(q, state, l, pos, n_heads, head_size, kv_mul, transformer_weights):
    t1 = time_in_ms()
    BLOCK_N = min(next_power_of_2(pos + 1), 64)
    q_heads = q.view(n_heads, head_size)  # [H, D]
    K_heads = state.key_cache[l]  # [L, D]
    V_heads = state.value_cache[l]  # [L, D]
    dim = head_size * n_heads

    assert kv_mul > 0

    if is_power_of_2(head_size):
        grid = lambda META: (n_heads,)

        flash_attention_kernel_1d[grid](q_heads, K_heads,
                                        V_heads,
                                        state.atten_out, pos + 1, dim, kv_mul, transformer_weights.scale,
                                        BLOCK_N=BLOCK_N,
                                        HEAD_DIM=head_size)

    else:
        assert False
        grid = lambda META: (n_heads,)
        flash_attention_kernel_1d_corner[grid](q_heads, K_heads,
                                               V_heads,
                                               state.atten_out, pos + 1, dim, head_size, transformer_weights.scale,
                                               BLOCK_N=BLOCK_N, HEAD_DIM=next_power_of_2(head_size)
                                               )
    te = time_in_ms()
    store_time('attention_flash', te - t1)


def attention(q, state, l, pos, n_heads, head_size, kv_mul, transformer_weights):
    for h in range(n_heads):
        lq = q[h * head_size:(h + 1) * head_size]
        t_z = time_in_ms()
        K = state.key_cache[l, :pos + 1, (h // kv_mul) * head_size:(h // kv_mul + 1) * head_size]
        att = (K @ lq) / transformer_weights.scale
        t_z2 = time_in_ms()
        store_time('attention_dot', t_z2 - t_z)
        att = softmax(att, dim=0)
        V = state.value_cache[l, :pos + 1, (h // kv_mul) * head_size:(h // kv_mul + 1) * head_size]
        a_v = att @ V
        state.atten_out[h * head_size:(h + 1) * head_size] = a_v
