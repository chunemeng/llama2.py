import time

import torch

from kernel.flash_attention import flash_attention_kernel, next_power_of_2, flash_attention_kernel_1d, is_power_of_2, \
    flash_attention_kernel_1d_in_graph, flash_attention_kernel_1d_corner_in_graph, flash_attention_kernel_1d_corner
from ops.add import add
from ops.cast import cast
from ops.matmul import matmul_in_graph, matmul, matmul_residual_in_graph, matmul_residual, matmul_triton_float32, \
    matmul_triton_bfloat16
from ops.rmsnorm import rmsnorm_in_graph, rmsnorm, rmsnorm_batch, rmsnorm_batch_in_graph, rmsnorm_batch_merge
from ops.rope import rope_triton_in_graph, rope_opt, rope_split, rope_split_in_graph
from ops.softmax import softmax
from ops.swiglu import swiglu
from ops.time_util import time_in_ms, global_timing_report, store_time, clear_timing
from ops.attention import batch_mha, flash_attention, attention
from src.infer import generate, chat
from src.sampler import Sampler
from src.tokenizer import Tokenizer, ModelTokenizer
from src.transformer import Transformer, Config

# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# neural net operations


tune_dict = {(768, 768): (64, 128)}


def autotune_matmul(D, N):
    return tune_dict.get((D, N), (128, 128))


class Qwen3Transformer(Transformer):
    def __init__(self):
        super().__init__()
        # self.merge_matmul_check = True

    def read_hf_config(self, config_path: str = None):
        super().read_hf_config(config_path)
        self.config.has_qk_norm = True

    def read_checkpoint(self, checkpoint_path: str, config_type=torch.bfloat16, to_type=None):
        to_type = config_type
        super().read_checkpoint(checkpoint_path, config_type, to_type)

    def build_transformer(self, checkpoint_path: str, config_path: str = None):
        super().build_transformer(checkpoint_path, config_path)
        self.forward(0, 0)
        self.capture_graph(144, 0)
        clear_timing()

    def capture_graph(self, token, pos):
        config = self.config
        transformer_weights = self.weights
        state = self.state

        # embedding
        if config.token_table_cpu and config.device == 'cuda':
            state.x[:] = transformer_weights.token_embedding_table[token].to('cuda')
        else:
            state.x[:] = transformer_weights.token_embedding_table[token]

        state.pos_tensor[0] = pos

        n_layers = config.n_layers
        dim = config.dim
        head_size = config.head_dim
        kv_dim = dim * config.n_kv_heads // config.n_heads
        n_heads = config.n_heads
        n_kv_heads = config.n_kv_heads
        kv_mul = n_heads // config.n_kv_heads
        assert self.merge_matmul_check

        capture_stream = torch.cuda.Stream()

        # 确保默认 stream 上的操作完成
        torch.cuda.synchronize()

        with torch.cuda.stream(capture_stream):
            g = torch.cuda.CUDAGraph()
            self.cuda_graph = g
            g.capture_begin()
            for l in range(n_layers):
                # attention RMSNorm
                rmsnorm_in_graph(state.x, transformer_weights.rms_att_weight, out=state.xb, l=l)

                # q, k, v
                matmul_in_graph(state.xb, transformer_weights.wq, o=state.q, l=l)
                matmul_in_graph(state.xb, transformer_weights.wk, o=state.key_cache, l=l, p=state.pos_tensor)
                matmul_in_graph(state.xb, transformer_weights.wv, o=state.value_cache, l=l, p=state.pos_tensor)
                rmsnorm_batch_in_graph(state.q, transformer_weights.rms_q_weight,
                                       l=l, head_num=n_heads, head_size=head_size, out=state.q)
                rmsnorm_batch_in_graph(state.key_cache, transformer_weights.rms_k_weight, l=l, head_num=n_kv_heads,
                                       head_size=head_size,
                                       out=state.key_cache, pos=state.pos_tensor, seq_len=config.seq_len)

                # RoPE
                rope_split_in_graph(state.q, state.key_cache, transformer_weights.freq, n_heads, head_size,
                                    n_kv_heads, state.pos_tensor, l, config.seq_len)

                # multihead attention
                flash_attention_kernel_1d_in_graph[(config.n_heads,)](
                    state.q, state.key_cache, state.value_cache,
                    state.xb,
                    state.pos_tensor,
                    l,
                    dim, config.seq_len, kv_mul, transformer_weights.scale,
                    BLOCK_N=128, HEAD_DIM=head_size
                )

                # attention output
                matmul_residual_in_graph(state.xb, transformer_weights.wo, output=state.x, l=l)
                # ffn
                rmsnorm_in_graph(state.x, transformer_weights.rms_ffn_weight, l=l, out=state.xb)
                matmul_in_graph(state.xb, transformer_weights.w13, l=l, o=state.hbm)
                state.hbm32.copy_(state.hbm.type(torch.float32))
                hb = state.hbm32[:config.hidden_dim]
                hb2 = state.hbm32[config.hidden_dim:]
                hbo = state.hbm[:config.hidden_dim]

                swiglu(hb, hb2, output=hbo)
                matmul_in_graph(hbo, transformer_weights.w2, l=l, o=state.xb)
                add(state.x, state.xb, out=state.x)

            # final RMSNorm
            rmsnorm(state.x, transformer_weights.rms_final_weight, out=state.x)
            # logits
            matmul(state.x, transformer_weights.wcls, output=state.logits)
            g.capture_end()

        self._graph_constructed = True
        return state.logits

    def forward_in_graph(self, token, pos):
        tt = time_in_ms()
        transformer_weights = self.weights
        state = self.state
        config = self.config

        # embedding
        if config.token_table_cpu and config.device == 'cuda':
            state.x[:] = transformer_weights.token_embedding_table[token].to('cuda')
        else:
            state.x[:] = transformer_weights.token_embedding_table[token]

        state.pos_tensor[0] = pos

        self.cuda_graph.replay()

        te = time_in_ms()

        store_time('forward_in_graph', te - tt)
        return state.logits

    def prefill(self, tokens):
        tt = time_in_ms()
        config = self.config
        transformer_weights = self.weights
        state = self.state

        hidden_dim = config.hidden_dim
        n_layers = config.n_layers
        n_heads = config.n_heads
        n_kv_heads = config.n_kv_heads
        seq_len = config.seq_len
        head_size = config.head_dim
        dim = head_size * n_heads

        kv_dim = head_size * n_kv_heads
        kv_mul = n_heads // n_kv_heads

        # embedding
        state.x.copy_(transformer_weights.token_embedding_table[token].to(state.x.device))

        tc = 0
        for l in range(n_layers):
            # attention RMSNorm
            rmsnorm(state.x, transformer_weights.rms_att_weight[l], state.xb)

            # q, k, v
            q = matmul(state.xb, transformer_weights.wq[l])
            k = matmul(state.xb, transformer_weights.wk[l], state.key_cache[l, pos])
            v = matmul(state.xb, transformer_weights.wv[l], state.value_cache[l, pos])
            rmsnorm_batch_merge(q.view((n_heads, head_size)), transformer_weights.rms_q_weight[l],
                                q.view(n_heads, head_size),
                                k.view((n_kv_heads, head_size)), transformer_weights.rms_k_weight[l],
                                k.view(n_kv_heads, head_size))

            # RoPE
            # rope_opt(q, k, pos, transformer_weights.freq, dim, kv_dim)
            rope_split(q.view(n_heads, head_size), k.view(n_kv_heads, head_size), pos, transformer_weights.freq)

            # multihead attention
            if config.use_fused_attention:
                flash_attention(q, state, l, pos, n_heads, head_size, kv_mul,
                                transformer_weights)
            else:
                if kv_mul == 1:
                    batch_mha(q, state, l, pos, n_heads, head_size,
                              transformer_weights)
                else:
                    attention(q, state, l, pos, n_heads, head_size, kv_mul,
                              transformer_weights)

            # attention output
            matmul_residual(state.atten_out, transformer_weights.wo[l], output=state.x, tmp=state.xb2)

            # ffn
            rmsnorm(state.x, transformer_weights.rms_ffn_weight[l], out=state.xb)
            if self.merge_matmul_check:
                matmul_triton_float32(state.xb, transformer_weights.w13[l], output=state.hbm32)
                hb = state.hbm32[:config.hidden_dim]
                hb2 = state.hbm32[config.hidden_dim:]
            else:
                matmul(state.xb, transformer_weights.w1[l], output=state.hb)
                matmul(state.xb, transformer_weights.w3[l], output=state.hb2)
                hb = state.hb if state.hb.dtype == torch.float32 else cast(state.hb, torch.float32)
                hb2 = state.hb2

            swiglu(hb, hb2, output=hb)
            matmul_triton_bfloat16(hb, transformer_weights.w2[l], output=state.xb)
            add(state.x, state.xb, out=state.x)

        # final RMSNorm
        rmsnorm(state.x, transformer_weights.rms_final_weight, out=state.x)
        # logits
        matmul(state.x, transformer_weights.wcls, output=state.logits)

        tt2 = time_in_ms()
        store_time('forward_total', tt2 - tt)
        return state.logits

    def forward(self, token, pos):
        # if self._graph_constructed:
        #     return self.forward_in_graph(token, pos)
        tt = time_in_ms()
        config = self.config
        transformer_weights = self.weights
        state = self.state

        hidden_dim = config.hidden_dim
        n_layers = config.n_layers
        n_heads = config.n_heads
        n_kv_heads = config.n_kv_heads
        seq_len = config.seq_len
        head_size = config.head_dim
        dim = head_size * n_heads

        kv_dim = head_size * n_kv_heads
        kv_mul = n_heads // n_kv_heads

        # embedding
        state.x.copy_(transformer_weights.token_embedding_table[token].to(state.x.device))

        tc = 0
        for l in range(n_layers):
            # attention RMSNorm
            rmsnorm(state.x, transformer_weights.rms_att_weight[l], state.xb)

            # q, k, v
            q = matmul(state.xb, transformer_weights.wq[l])
            k = matmul(state.xb, transformer_weights.wk[l], state.key_cache[l, pos])
            v = matmul(state.xb, transformer_weights.wv[l], state.value_cache[l, pos])
            rmsnorm_batch_merge(q.view((n_heads, head_size)), transformer_weights.rms_q_weight[l],
                                q.view(n_heads, head_size),
                                k.view((n_kv_heads, head_size)), transformer_weights.rms_k_weight[l],
                                k.view(n_kv_heads, head_size))

            # RoPE
            # rope_opt(q, k, pos, transformer_weights.freq, dim, kv_dim)
            rope_split(q.view(n_heads, head_size), k.view(n_kv_heads, head_size), pos, transformer_weights.freq)

            # multihead attention
            if config.use_fused_attention:
                flash_attention(q, state, l, pos, n_heads, head_size, kv_mul,
                                transformer_weights)
            else:
                if kv_mul == 1:
                    batch_mha(q, state, l, pos, n_heads, head_size,
                              transformer_weights)
                else:
                    attention(q, state, l, pos, n_heads, head_size, kv_mul,
                              transformer_weights)

            # attention output
            matmul_residual(state.atten_out, transformer_weights.wo[l], output=state.x, tmp=state.xb2)

            # ffn
            rmsnorm(state.x, transformer_weights.rms_ffn_weight[l], out=state.xb)
            if self.merge_matmul_check:
                matmul_triton_float32(state.xb, transformer_weights.w13[l], output=state.hbm32)
                hb = state.hbm32[:config.hidden_dim]
                hb2 = state.hbm32[config.hidden_dim:]
            else:
                matmul(state.xb, transformer_weights.w1[l], output=state.hb)
                matmul(state.xb, transformer_weights.w3[l], output=state.hb2)
                hb = state.hb if state.hb.dtype == torch.float32 else cast(state.hb, torch.float32)
                hb2 = state.hb2

            swiglu(hb, hb2, output=hb)
            matmul_triton_bfloat16(hb, transformer_weights.w2[l], output=state.xb)
            add(state.x, state.xb, out=state.x)

        # final RMSNorm
        rmsnorm(state.x, transformer_weights.rms_final_weight, out=state.x)
        # logits
        matmul(state.x, transformer_weights.wcls, output=state.logits)

        tt2 = time_in_ms()
        store_time('forward_total', tt2 - tt)
        return state.logits


def warmup(steps=10):
    x = torch.randn(768, device='cuda')
    w = torch.randn(768, device='cuda')
    o = torch.randn(768, device='cuda')


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
    parser.add_argument("-c", "--config_path", type=str, default=None)
    args = parser.parse_args()
    # warmup()

    rng_seed = args.s if args.s is not None else int(time.time())
    steps = args.n
    temperature = max(args.t, 0.0)
    topp = min(max(args.p, 0.0), 1.0)

    # build transformer
    transformer = Qwen3Transformer()  # assume Python Transformer class wraps checkpoint
    transformer.build_transformer(args.checkpoint, args.config_path)
    vocab_size = transformer.config.vocab_size
    if steps <= 0 or steps > transformer.config.seq_len:
        steps = transformer.config.seq_len

    # build sampler
    sampler = Sampler(vocab_size=vocab_size, temperature=temperature, topp=topp, rng_seed=rng_seed)

    tokenizer = ModelTokenizer("./qwen3-600M")

    # run mode
    if args.m == "generate":
        generate(transformer, tokenizer, sampler, args.i, steps)
    elif args.m == "chat":
        chat(transformer, tokenizer, sampler, args.i, args.y, steps)


if __name__ == "__main__":
    main_cli()
    global_timing_report()
