import mmap
import os
import string
import struct
import time

import torch
import torch.nn.functional as F
import math

import triton

from torch import nn

from kernel.flash_attention import flash_attention_kernel, next_power_of_2, flash_attention_kernel_1d, is_power_of_2
from kernel.matmul import matvec_kernel, matmul_residual_kernel
from kernel.rmsnorm import rmsnorm_kernel_split_col, rmsnorm_kernel_split_col_one_row, rmsnorm_kernel_one_row
from kernel.rope import rope_1d_kernel
from kernel.swiglu import swiglu_1d_kernel
from kernel.vec_add import vec_add_kernel


# ----------------------------------------------------------------------------
# Transformer config and weights

class Config:
    def __init__(self, dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, device='cuda'):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.device = device

    @classmethod
    def size(cls):
        return 7 * 4

    @classmethod
    def from_bytes(cls, config_bytes):
        if len(config_bytes) != cls.size():
            raise ValueError(f"Invalid config size: expected {cls.size()} bytes, got {len(config_bytes)} bytes")
        unpacked = struct.unpack('iiiiiii', config_bytes)
        return cls(*unpacked[:], device='cuda')


class TransformerWeights:
    def __init__(self, config: Config, device='cuda'):
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size = \
            config.dim, config.hidden_dim, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size
        head_size = dim // n_heads
        kv_dim = dim * n_kv_heads // n_heads

        self.dev = device
        dev = device
        # preallocate as empty tensors (will be memory-mapped later)
        self.token_embedding_table = None  # torch.empty(vocab_size, dim, device=dev)
        self.rms_att_weight = torch.empty(n_layers, dim, device=dev)
        self.rms_ffn_weight = torch.empty(n_layers, dim, device=dev)
        self.rms_final_weight = torch.empty(dim, device=dev)

        self.wq = torch.empty(n_layers, dim, dim, device=dev)
        self.wk = torch.empty(n_layers, dim, kv_dim, device=dev)
        self.wv = torch.empty(n_layers, dim, kv_dim, device=dev)
        self.wo = torch.empty(n_layers, dim, dim, device=dev)

        merge_matmul_check = os.getenv('MERGE_MATMUL') == '1'
        if merge_matmul_check:
            self.w13 = None  # for cat(w1, w3) check only
        else:
            self.w1 = None  # torch.empty(n_layers, hidden_dim, dim, device=dev)
            self.w2 = None  # torch.empty(n_layers, dim, hidden_dim, device=dev)
            self.w3 = None  # torch.empty(n_layers, hidden_dim, dim, device=dev)

        self.wcls = torch.empty(vocab_size, dim, device=dev)
        self.freq = None
        self.scale = 1.0 / math.sqrt(head_size)

    def memory_map_weights(self, config: Config, data: torch.Tensor, shared_weights: int, device='cuda'):
        """
        Map a contiguous float tensor `data` to the weight tensors according to C logic.
        `data` should be a 1D float tensor (from mmap or torch.frombuffer)
        """
        dim = config.dim
        hidden_dim = config.hidden_dim
        n_layers = config.n_layers
        n_heads = config.n_heads
        n_kv_heads = config.n_kv_heads
        vocab_size = config.vocab_size
        seq_len = config.seq_len
        head_size = dim // n_heads

        ptr = 0  # current offset in the 1D data tensor

        def slice_tensor(shape):
            nonlocal ptr
            n_elems = 1
            for s in shape:
                n_elems *= s
            t = data[ptr:ptr + n_elems]
            ptr += n_elems
            if device == 'cuda':
                return t.view(*shape).to(device)
            else:
                return t.view(*shape)

        # map weights
        self.token_embedding_table = slice_tensor((vocab_size, dim))
        self.rms_att_weight = slice_tensor((n_layers, dim))
        self.wq = slice_tensor((n_layers, dim, head_size * n_heads))
        self.wk = slice_tensor((n_layers, dim, n_kv_heads * head_size))
        self.wv = slice_tensor((n_layers, dim, n_kv_heads * head_size))
        self.wo = slice_tensor((n_layers, n_heads * head_size, dim))
        self.rms_ffn_weight = slice_tensor((n_layers, dim))
        w1 = slice_tensor((n_layers, hidden_dim, dim))
        self.w2 = slice_tensor((n_layers, dim, hidden_dim))
        w3 = slice_tensor((n_layers, hidden_dim, dim))
        merge_matmul_check = os.getenv('MERGE_MATMUL') == '1'

        if not merge_matmul_check:
            self.w1 = w1
            self.w3 = w3
        else:
            self.w13 = torch.cat([w1, w3], dim=1)  # for checking only
        self.rms_final_weight = slice_tensor((dim,))

        # skip RoPE frequencies (if present in the C mmap)
        ptr += seq_len * head_size / 2  # real
        ptr += seq_len * head_size / 2  # imag

        # optional classifier weight
        if shared_weights:
            self.wcls = self.token_embedding_table
        else:
            self.wcls = slice_tensor((vocab_size, dim))

        head_dim = torch.arange(0, dim, 2, device=self.dev) % head_size  # [0, 1, ..., head_size//2]
        self.freq = 1.0 / (10000 ** (head_dim.float() / head_size))


# ----------------------------------------------------------------------------
# RunState
class RunState:
    def __init__(self, config: Config, device='cuda'):
        dim = config.dim
        hidden_dim = config.hidden_dim
        n_layers = config.n_layers
        n_heads = config.n_heads
        n_kv_heads = config.n_kv_heads
        seq_len = config.seq_len
        vocab_size = config.vocab_size
        kv_dim = (dim * n_kv_heads) // n_heads

        dev = device

        # allocate buffers, initialized to zero (like calloc)
        self.x = torch.zeros(dim, device=dev)
        self.xb = torch.zeros(dim, device=dev)
        self.xb2 = torch.zeros(dim, device=dev)
        merge_matmul_check = os.getenv('MERGE_MATMUL') == '1'
        if merge_matmul_check:
            self.hbm = torch.zeros(hidden_dim * 2, device=dev)
        else:
            self.hb = torch.zeros(hidden_dim, device=dev)
            self.hb2 = torch.zeros(hidden_dim, device=dev)
        qkv_device = 'cuda'
        self.key_cache = torch.zeros(n_layers, seq_len, kv_dim, device=qkv_device)
        self.value_cache = torch.zeros(n_layers, seq_len, kv_dim, device=qkv_device)
        self.logits = torch.zeros(vocab_size, device=dev)


time_dict = {}
num_dict = {}


def store_time(name, delta):
    disable = False
    if disable == False:
        time_dict[name] = time_dict.get(name, 0) + delta
        num_dict[name] = num_dict.get(name, 0) + 1


# ----------------------------------------------------------------------------
# neural net operations

def rmsnorm_triton(x, weight, out, eps=1e-5):
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
    if not use_one_row:
        rmsnorm_kernel_split_col[grid](
            x, weight, out,
            d,
            eps=eps,
            BLOCK_COL_SIZE=512,
            num_stages=4
        )
    else:
        rmsnorm_kernel_one_row[grid](
            x, weight, out,
            d,
            eps=eps,
            BLOCK_COL_SIZE=1024,
            num_warps=2
        )

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


def rmsnorm(x, weight, out=None, eps=1e-5):
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


def softmax(x, dim):
    t = time_in_ms()
    res = torch.softmax(x, dim=dim)
    t2 = time_in_ms()
    store_time('softmax_cpu', t2 - t)
    return res


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
    matvec_kernel[grid](x, w, output,
                        w.shape[0], w.shape[1], BLOCK_D=128, BLOCK_N=128)
    t2 = time_in_ms()
    store_time('matmul_triton', t2 - t)
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


def add_triton(x, y, out=None):
    t = time_in_ms()

    if out is None:
        out = torch.empty_like(x)

    grid = lambda META: (triton.cdiv(x.shape[0], META['BLOCK_SIZE']),)
    vec_add_kernel[grid](
        x, y, out,
        x.shape[0],
        BLOCK_SIZE=128
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


def matmul_residual_triton(x, w, output):
    t = time_in_ms()
    grid = lambda META: (triton.cdiv(w.shape[0], META['BLOCK_D']),)
    matmul_residual_kernel[grid](x, w, output,
                                 w.shape[0], w.shape[1], BLOCK_D=64, BLOCK_N=64, num_stages=3)
    t2 = time_in_ms()
    store_time('matmul_residual_triton', t2 - t)
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


def rope_base(q, k, pos, dim, head_size, kv_dim=None):
    tl = time_in_ms()
    for i in range(0, dim, 2):
        head_dim = i % head_size
        freq = 1.0 / (10000 ** (float(head_dim) / head_size))
        val = pos * freq
        fcr = math.cos(val)
        fci = math.sin(val)
        rotn = 2 if i < kv_dim else 1
        for v_idx in range(rotn):
            vec = q if v_idx == 0 else k
            v0, v1 = vec[i].clone(), vec[i + 1].clone()
            vec[i] = v0 * fcr - v1 * fci
            vec[i + 1] = v0 * fci + v1 * fcr
    to = time_in_ms()
    store_time('rope_base', to - tl)


def rope_triton(q, k, freq, pos, dim):
    tr = time_in_ms()
    grid = lambda META: (triton.cdiv(dim // 2, META['BLOCK_SIZE']),)
    rope_1d_kernel[grid](
        q, k, freq, pos,
        dim, BLOCK_SIZE=128)
    tr2 = time_in_ms()
    store_time('rope_triton', tr2 - tr)


def rope_opt(q, k, pos, freq, dim, kv_dim=None):
    tt = time_in_ms()
    assert dim == kv_dim
    # optimized RoPE implementation using slicing

    if q.is_cuda and k.is_cuda and freq.is_cuda:
        return rope_triton(q, k, freq, pos, dim)

    val = pos * freq

    fcr = torch.cos(val)

    fci = torch.sin(val)
    #
    q_even = q[0::2]
    k_even = k[0::2]
    q_odd = q[1::2]
    k_odd = k[1::2]
    #
    # even_idx = torch.arange(0, head_size, 2)
    # fcr = torch.cos(val).unsqueeze(0)  # [1, head_size//2]
    # fci = torch.sin(val).unsqueeze(0)

    # q[0::2], q[::2] = q[0::2] * fcr - q[1::2] * fci, \
    #                   q[0::2] * fci + q[1::2] * fcr

    tc = time_in_ms()
    q_new_even = q_even * fcr - q_odd * fci
    k_new_even = k_even * fcr - k_odd * fci
    k_new_odd = k_even * fci + k_odd * fcr
    q_new_odd = q_even * fci + q_odd * fcr

    tz = time_in_ms()
    q[0::2] = q_new_even
    q[1::2] = q_new_odd
    k[0::2] = k_new_even
    k[1::2] = k_new_odd
    to = time_in_ms()
    # time_dict['rope_opt_in'] = time_dict.get('rope_opt_in', 0) + (tc - tt)
    # time_dict['rope_opt_calc'] = time_dict.get('rope_opt_calc', 0) + (tz - tc)
    # time_dict['rope_opt_output'] = time_dict.get('rope_opt_output', 0) + (to - tz)
    # assert torch.allclose(q, qq, atol=1e-4), f"RoPE q mismatch {torch.max(torch.abs(q - qq))}"
    # assert torch.allclose(k, qk, atol=1e-4), f"RoPE k mismatch {torch.max(torch.abs(k - qk))}"
    store_time('rope_opt', to - tt)


# ----------------------------------------------------------------------------

class Transformer:
    def __init__(self):
        self.config = None
        self.weights = None
        self.state = None

        self.fd = -1
        self.file_size = 0
        self.mmap_obj = None
        self.data = None

    def read_checkpoint(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

        with open(checkpoint_path, "rb") as f:
            config_bytes = f.read(Config.size())  # config_class.size() = sizeof(Config)
            config = Config.from_bytes(config_bytes)
            self.config = config
            shared_weights = 1 if config.vocab_size > 0 else 0
            config.vocab_size = abs(config.vocab_size)

            # figure out file size
            f.seek(0, os.SEEK_END)
            file_size = f.tell()

            self.fd = os.open(self.checkpoint_path, os.O_RDONLY)
            self.mmap_obj = mmap.mmap(self.fd, file_size, access=mmap.ACCESS_READ)

            offset = Config.size()
            # number of floats
            n_floats = (file_size - offset) // 4
            data = torch.frombuffer(self.mmap_obj[offset:], dtype=torch.float32)

            # 4. map weights
            weights = TransformerWeights(config)
            weights.memory_map_weights(config, data, shared_weights, device=config.device)
            self.weights = weights

    def build_transformer(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.read_checkpoint(checkpoint_path)

        self.state = RunState(self.config, device=self.config.device)

    def close(self):
        if self.mmap_obj is not None:
            self.mmap_obj.close()
            self.mmap_obj = None
        if self.fd != -1:
            os.close(self.fd)
            self.fd = -1
        self.data = None

    def forward(self, token, pos):
        tt = time_in_ms()
        config = self.config
        transformer_weights = self.weights
        state = self.state

        dim = config.dim
        hidden_dim = config.hidden_dim
        n_layers = config.n_layers
        n_heads = config.n_heads
        n_kv_heads = config.n_kv_heads
        seq_len = config.seq_len
        head_size = dim // n_heads
        kv_dim = dim * n_kv_heads // n_heads
        kv_mul = n_heads // n_kv_heads

        # embedding
        state.x[:] = transformer_weights.token_embedding_table[token]

        tc = 0
        for l in range(n_layers):
            # attention RMSNorm
            rmsnorm(state.x, transformer_weights.rms_att_weight[l], state.xb)

            # q, k, v
            q = matmul(state.xb, transformer_weights.wq[l])
            k = matmul(state.xb, transformer_weights.wk[l], state.key_cache[l, pos])
            v = matmul(state.xb, transformer_weights.wv[l], state.value_cache[l, pos])

            # RoPE
            rope_opt(q, k, pos, transformer_weights.freq, dim, kv_dim)
            # rope_base(q, k, pos, dim, head_size, kv_dim)
            # assert torch.allclose(q, qq, atol=1e-4), f"RoPE q mismatch {torch.max(torch.abs(q - qq))}"
            # assert torch.allclose(k, kq, atol=1e-4), f"RoPE k mismatch {torch.max(torch.abs(k - kq))}"

            scale = float(math.sqrt(head_size))

            # multihead attention
            if True:
                BLOCK_D = next_power_of_2(head_size)
                BLOCK_N = 64
                q_heads = q.view(n_heads, head_size)
                K_heads = state.key_cache[l, :pos + 1].reshape(pos + 1, head_size)  # [L, D]
                V_heads = state.value_cache[l, :pos + 1].reshape(pos + 1, head_size)  # [L, D]
                grid = lambda META: (triton.cdiv(pos + 1, META['BLOCK_N']), head_size)
                if BLOCK_D == head_size:
                    flash_attention_kernel_1d[grid](q_heads, K_heads,
                                                    V_heads,
                                                    state.xb, pos + 1, dim, transformer_weights.scale,
                                                    BLOCK_N=BLOCK_N, HEAD_DIM=head_size)
                else:
                    flash_attention_kernel[grid](
                        q.view(1, n_heads, head_size),
                        state.key_cache[l, :pos + 1].reshape(1, pos + 1, head_size),
                        state.value_cache[l, :pos + 1].reshape(1, pos + 1, head_size),
                        state.xb.view(1, n_heads, head_size),
                        pos + 1, dim, n_heads,
                        BLOCK_N=BLOCK_N,
                        BLOCK_D=BLOCK_D,
                        HEAD_DIM=head_size
                    )



            else:
                if kv_mul == 1:
                    tl = time_in_ms()
                    q_heads = q.view(n_heads, head_size)
                    K_heads = state.key_cache[l, :pos + 1].reshape(pos + 1, n_heads, head_size).permute(1, 0,
                                                                                                        2)  # [H, S, D]
                    V_heads = state.value_cache[l, :pos + 1].reshape(pos + 1, n_heads, head_size).permute(1, 0,
                                                                                                          2)  # [H, S, D]
                    attd = torch.bmm(K_heads, q_heads.unsqueeze(-1)).squeeze(-1) / scale

                    attd = softmax(attd, dim=1)
                    out_heads = torch.bmm(attd.unsqueeze(1), V_heads).squeeze(1)  # [H, D]
                    state.xb[:] = out_heads.reshape(-1)
                    to = time_in_ms()
                    store_time('attention_bmm', to - tl)
                else:
                    for h in range(n_heads):
                        lq = q[h * head_size:(h + 1) * head_size]
                        t_z = time_in_ms()
                        K = state.key_cache[l, :pos + 1, (h // kv_mul) * head_size:(h // kv_mul + 1) * head_size]
                        att = (K @ lq) / scale
                        t_z2 = time_in_ms()
                        store_time('attention_dot', t_z2 - t_z)
                        att = softmax(att, dim=0)
                        V = state.value_cache[l, :pos + 1, (h // kv_mul) * head_size:(h // kv_mul + 1) * head_size]
                        a_v = att @ V
                        state.xb[h * head_size:(h + 1) * head_size] = a_v

            # attention output
            matmul_residual(state.xb, transformer_weights.wo[l], state.x, tmp=state.xb2)

            # matmul(state.xb, transformer_weights.wo[l], output=state.xb2)
            # add(state.x, state.xb2, out=state.x)

            # ffn
            rmsnorm(state.x, transformer_weights.rms_ffn_weight[l], out=state.xb)
            merge_matmul_check = os.getenv('MERGE_MATMUL') == '1'
            if merge_matmul_check:
                matmul(state.xb, transformer_weights.w13[l], output=state.hbm)
                hb = state.hbm.narrow(0, 0, hidden_dim)
                hb2 = state.hbm.narrow(0, hidden_dim, hidden_dim)
            else:
                matmul(state.xb, transformer_weights.w1[l], output=state.hb)
                matmul(state.xb, transformer_weights.w3[l], output=state.hb2)
                hb = state.hb
                hb2 = state.hb2

            swiglu(hb, hb2, output=hb)
            matmul(hb, transformer_weights.w2[l], output=state.xb)
            add(state.x, state.xb, out=state.x)

        # final RMSNorm
        rmsnorm(state.x, transformer_weights.rms_final_weight, out=state.x)
        # logits
        matmul(state.x, transformer_weights.wcls, output=state.logits)

        tt2 = time_in_ms()
        store_time('forward_total', tt2 - tt)
        return state.logits


class TokenIndex:
    def __init__(self, s: str, idx: int):
        self.str = s
        self.id = idx


class Tokenizer:
    def __init__(self):
        self.vocab = []  # List[str]
        self.vocab_scores = None  # torch.FloatTensor
        self.sorted_vocab = None  # List[TokenIndex]
        self.vocab_size = 0
        self.max_token_length = 0
        self.byte_pieces = [bytes([i]) for i in range(256)]

    def build_tokenizer(self, tokenizer_path: str, vocab_size: int):
        self.vocab_size = vocab_size
        self.vocab_scores = torch.zeros(vocab_size, dtype=torch.float32)
        self.vocab = [''] * vocab_size
        self.sorted_vocab = None

        with open(tokenizer_path, 'rb') as f:
            self.max_token_length = struct.unpack('i', f.read(4))[0]
            for i in range(vocab_size):
                self.vocab_scores[i] = struct.unpack('f', f.read(4))[0]
                length = struct.unpack('i', f.read(4))[0]
                self.vocab[i] = f.read(length).decode('utf-8')

    def decode(self, prev_token: int, token: int) -> str:
        t = time_in_ms()
        piece = self.vocab[token]
        if prev_token == 1 and piece.startswith(' '):
            piece = piece[1:]

        # handle raw byte tokens like '<0x01>'
        if piece.startswith('<0x') and piece.endswith('>'):
            byte_val = int(piece[3:-1], 16)
            piece = self.byte_pieces[byte_val].decode('latin1')

        t2 = time_in_ms()
        store_time('token_decode', t2 - t)
        return piece

    def safe_print(self, piece: str):
        if piece is None or piece == '':
            return
        if len(piece) == 1:
            if piece not in string.printable:
                return
        print(piece, end='')

    def _build_sorted_vocab(self):
        if self.sorted_vocab is None:
            self.sorted_vocab = [TokenIndex(s, idx) for idx, s in enumerate(self.vocab)]
            self.sorted_vocab.sort(key=lambda x: x.str)

    def str_lookup(self, s: str) -> int:
        self._build_sorted_vocab()
        # binary search
        lo, hi = 0, self.vocab_size - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            cmp = (self.sorted_vocab[mid].str > s) - (self.sorted_vocab[mid].str < s)
            if cmp == 0:
                return self.sorted_vocab[mid].id
            elif cmp < 0:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1

    def encode(self, text: str, bos: bool = True, eos: bool = True):
        if text is None:
            raise ValueError("Cannot encode None text")

        self._build_sorted_vocab()
        tokens = []

        # add BOS token
        if bos:
            tokens.append(1)

        # dummy prefix token
        if text:
            dummy_prefix = self.str_lookup(" ")
            if dummy_prefix != -1:
                tokens.append(dummy_prefix)

        # UTF-8 byte processing
        str_buffer = bytearray()
        i = 0
        while i < len(text):
            c = text[i]
            b = c.encode('utf-8')
            str_buffer.extend(b)

            # try to match the whole buffer
            s = str_buffer.decode('utf-8', errors='ignore')
            idx = self.str_lookup(s)
            if idx != -1:
                tokens.append(idx)
                str_buffer.clear()
            else:
                # byte fallback for each UTF-8 byte
                for byte in b:
                    tokens.append(byte + 3)
                str_buffer.clear()
            i += 1

        # greedy merge according to vocab_scores
        while True:
            best_score = -1e10
            best_id = -1
            best_idx = -1
            for i in range(len(tokens) - 1):
                s = self.vocab[tokens[i]] + self.vocab[tokens[i + 1]]
                idx = self.str_lookup(s)
                if idx != -1 and self.vocab_scores[idx] > best_score:
                    best_score = self.vocab_scores[idx].item()
                    best_id = idx
                    best_idx = i
            if best_idx == -1:
                break
            tokens[best_idx] = best_id
            tokens.pop(best_idx + 1)

        # add EOS token
        if eos:
            tokens.append(2)

        return tokens


class ProbIndex:
    def __init__(self, prob: float, index: int):
        self.prob = prob
        self.index = index


class Sampler:
    def __init__(self, vocab_size: int, temperature: float = 1.0, topp: float = 1.0, rng_seed: int = 1337):
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.topp = topp
        self.rng_state = rng_seed
        # buffer used for top-p sampling
        self.probindex = [ProbIndex(0.0, 0) for _ in range(vocab_size)]

    # ----------------- Xorshift64* RNG -----------------
    def random_u32(self):
        s = self.rng_state
        s ^= (s >> 12) & 0xFFFFFFFFFFFFFFFF
        s ^= (s << 25) & 0xFFFFFFFFFFFFFFFF
        s ^= (s >> 27) & 0xFFFFFFFFFFFFFFFF
        self.rng_state = s
        return ((s * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF) >> 32

    def random_f32(self):
        return (self.random_u32() >> 8) / 16777216.0

    # ----------------- Sampling Methods -----------------
    @staticmethod
    def sample_argmax(probabilities: torch.Tensor) -> int:
        return int(torch.argmax(probabilities).item())

    @staticmethod
    def sample_mult(probabilities: torch.Tensor, coin: float) -> int:
        cdf = torch.cumsum(probabilities, dim=0)
        idx = torch.searchsorted(cdf, torch.tensor([coin], dtype=probabilities.dtype))
        return int(idx.item())

    def sample_topp(self, probabilities: torch.Tensor, coin: float) -> int:
        # 1. 过滤候选
        cutoff = (1.0 - self.topp) / (self.vocab_size - 1)
        mask = probabilities >= cutoff
        probs = probabilities[mask]
        indices = torch.arange(self.vocab_size, device=probabilities.device)[mask]

        # 2. 排序（降序）
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        sorted_indices = indices[sorted_idx]

        # 3. 累积概率
        cum_probs = torch.cumsum(sorted_probs, dim=0)

        # 4. 找到截断位置
        last_idx = torch.searchsorted(cum_probs, self.topp, right=True)
        last_idx = min(last_idx.item(), len(sorted_probs) - 1)  # 防止越界

        # 5. 样本采样
        r = coin * cum_probs[last_idx]
        pick_idx = torch.searchsorted(cum_probs[:last_idx + 1], r)
        return sorted_indices[pick_idx].item()

    # ----------------- Main Sample Function -----------------
    def sample(self, logits: torch.Tensor) -> int:
        t = time_in_ms()
        logits = logits.clone()
        if self.temperature == 0.0:
            t_o = time_in_ms()
            res = self.sample_argmax(logits)
            store_time('sample_argmax', t_o - t)
            return res

        # apply temperature
        logits /= self.temperature

        # softmax
        probs = torch.softmax(logits, dim=0)

        # random coin
        coin = self.random_f32()

        if self.topp <= 0.0 or self.topp >= 1.0:
            t_o = time_in_ms()
            res = self.sample_mult(probs, coin)
            store_time('sample_mult', t_o - t)
            return res
        else:
            t_o = time_in_ms()
            res = self.sample_topp(probs, coin)
            store_time('sample_topp', t_o - t)
            return res


# --------------------- Utilities ---------------------
def time_in_ms():
    """Return current time in milliseconds."""
    return int(time.time() * 1000)


def read_stdin(prompt: str) -> str:
    """Read a line from stdin."""
    return input(prompt)


def safe_print(piece: str):
    """Print safely, skip non-printable characters."""
    if piece is None or piece == "":
        return
    for c in piece:
        if c.isprintable() or c.isspace():
            print(c, end="")


# --------------------- Generation Loop ---------------------
def generate(transformer, tokenizer, sampler, prompt: str, steps: int):
    prompt = prompt or ""
    # encode prompt
    prompt_tokens = tokenizer.encode(prompt, bos=True, eos=False)
    if len(prompt_tokens) < 1:
        raise RuntimeError("Expected at least 1 prompt token")

    pos = 0
    token = prompt_tokens[0]
    next_token = None
    start_time = None

    while pos < steps:
        logits = transformer.forward(token, pos)  # logits tensor [vocab_size]

        # determine next token
        if pos < len(prompt_tokens) - 1:
            next_token = prompt_tokens[pos + 1]
        else:
            next_token = sampler.sample(logits)

        # print output
        piece = tokenizer.decode(token, next_token)
        safe_print(piece)
        token = next_token
        pos += 1

        # start timer after first iteration
        if start_time is None:
            start_time = time_in_ms()

        # stop if BOS token encountered
        if next_token == 1:
            break

    print()
    # flush to stdout immediately
    import sys
    sys.stdout.flush()

    # report tokens/sec
    if pos > 1:
        end_time = time_in_ms()
        tok_per_sec = (pos - 1) / ((end_time - start_time) / 1000.0)
        print(f"achieved tok/s: {tok_per_sec:.2f}")


# --------------------- Chat Loop ---------------------
def chat(transformer, tokenizer, sampler, cli_user_prompt=None, cli_system_prompt=None, steps=256):
    system_prompt = ""
    user_prompt = ""
    rendered_prompt = ""
    prompt_tokens = []
    user_turn = True
    pos = 0
    next_token = None
    user_idx = 0

    while pos < steps:
        if user_turn:
            if pos == 0:
                if cli_system_prompt is not None:
                    system_prompt = cli_system_prompt
                else:
                    system_prompt = read_stdin("Enter system prompt (optional): ")

            if pos == 0 and cli_user_prompt is not None:
                user_prompt = cli_user_prompt
            else:
                user_prompt = read_stdin("User: ")

            if pos == 0 and system_prompt.strip():
                rendered_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
            else:
                rendered_prompt = f"[INST] {user_prompt} [/INST]"

            prompt_tokens = tokenizer.encode(rendered_prompt, bos=True, eos=False)
            user_idx = 0
            user_turn = False
            print("Assistant: ", end="", flush=True)

        # select token to feed into transformer
        if user_idx < len(prompt_tokens):
            token = prompt_tokens[user_idx]
            user_idx += 1
        else:
            token = next_token

        # EOS token ends Assistant turn
        if token == 2:
            user_turn = True

        logits = transformer.forward(token, pos)
        next_token = sampler.sample(logits)
        pos += 1

        # print output
        if user_idx >= len(prompt_tokens) and next_token != 2:
            piece = tokenizer.decode(token, next_token)
            safe_print(piece)
        if next_token == 2:
            print()


def warmup(steps=10):
    x = torch.randn(768, device='cuda')
    w = torch.randn(768, device='cuda')
    o = torch.randn(768, device='cuda')
    for _ in range(steps):
        y = rope_opt(x, x, 0, w, 768, 768)


# --------------------- CLI ---------------------
def main_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Transformer inference")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("-n", type=int, default=256)
    parser.add_argument("-i", type=str, default=None)
    parser.add_argument("-t", type=float, default=1.0)
    parser.add_argument("-p", type=float, default=0.9)
    parser.add_argument("-s", type=int, default=None)
    parser.add_argument("-z", type=str, default="tokenizer.bin")
    parser.add_argument("-m", type=str, default="generate", choices=["generate", "chat"])
    parser.add_argument("-y", type=str, default=None)
    args = parser.parse_args()
    # warmup()

    rng_seed = args.s if args.s is not None else int(time.time())
    steps = args.n
    temperature = max(args.t, 0.0)
    topp = min(max(args.p, 0.0), 1.0)

    # build transformer
    transformer = Transformer()  # assume Python Transformer class wraps checkpoint
    transformer.build_transformer(args.checkpoint)
    vocab_size = transformer.config.vocab_size
    if steps <= 0 or steps > transformer.config.seq_len:
        steps = transformer.config.seq_len

    # build tokenizer
    tokenizer = Tokenizer()
    tokenizer.build_tokenizer(args.z, vocab_size)

    # build sampler
    sampler = Sampler(vocab_size=vocab_size, temperature=temperature, topp=topp, rng_seed=rng_seed)

    # run mode
    if args.m == "generate":
        generate(transformer, tokenizer, sampler, args.i, steps)
    elif args.m == "chat":
        chat(transformer, tokenizer, sampler, args.i, args.y, steps)


if __name__ == "__main__":
    main_cli()
    for k, v in zip(time_dict.keys(), time_dict.values()):
        print(f"{k}: {v:.2f} ms ({num_dict[k]} calls) avg: {v / num_dict[k]:.2f} ms")
