# Transformer config and weights
import math
import mmap
import os
import struct

import torch

dtype_map = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


class Config:
    def __init__(self):
        self.dim = 0
        self.hidden_dim = 0
        self.n_layers = 0
        self.n_heads = 0
        self.n_kv_heads = 0
        self.vocab_size = 0
        self.seq_len = 0
        self.torch_dtype = torch.float32
        self.q_type = None
        self.use_fused_attention = True
        self.device = 'cuda'
        self.merge_matmul_check = os.getenv('MERGE_MATMUL') == '1'
        self.has_qk_norm = False
        self.token_table_cpu = False
        self.rms_norm_eps = 1e-6
        self.head_dim = 0
        self.rope_range = 0
        self.max_batch_size = 10
        self.fnn_use_torch = False
        self.use_matmul_triton = True
        self.rope_theta = 10000

    def from_params(self, dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len,
                    config_type='float32', q_type=None, device='cuda'):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.torch_dtype = dtype_map.get(config_type, config_type)
        if q_type is None:
            self.q_type = self.torch_dtype
        else:
            self.q_type = dtype_map.get(q_type, q_type)
        self.use_fused_attention = True
        self.device = device
        self.head_dim = dim // n_heads
        self.rope_range = self.head_dim

    def from_json(self, json_dict):
        self.dim = json_dict['hidden_size']
        self.hidden_dim = json_dict['intermediate_size']
        self.head_dim = json_dict['head_dim']
        self.n_layers = json_dict['num_hidden_layers']
        self.n_heads = json_dict['num_attention_heads']
        self.n_kv_heads = json_dict['num_key_value_heads']
        self.vocab_size = json_dict['vocab_size']
        self.seq_len = 256
        self.torch_dtype = dtype_map.get(json_dict.get('torch_dtype', 'float32'), torch.float32)
        self.q_type = self.torch_dtype
        self.use_fused_attention = json_dict.get('use_fused_attention', True)
        self.device = json_dict.get('device', 'cuda')
        self.rms_norm_eps = json_dict.get('rms_norm_eps', 1e-6)
        self.rope_theta = json_dict.get('rope_theta', 10000)
        self.rope_range = self.head_dim

    @classmethod
    def size(cls):
        return 7 * 4

    @classmethod
    def from_bytes(cls, config_bytes):
        if len(config_bytes) != cls.size():
            raise ValueError(f"Invalid config size: expected {cls.size()} bytes, got {len(config_bytes)} bytes")
        unpacked = struct.unpack('iiiiiii', config_bytes)
        cfg = cls()
        cfg.from_params(*unpacked[:], device='cuda')
        return cfg


class TransformerWeights:
    def __init__(self, config: Config, device='cuda'):
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size = \
            config.dim, config.hidden_dim, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size
        head_size = config.head_dim
        self.dev = device
        # preallocate as empty tensors (will be memory-mapped later)
        self.token_embedding_table = None  # torch.empty(vocab_size, dim, device=dev)
        self.rms_att_weight = None  # (n_layers, dim)
        self.rms_ffn_weight = None  # (n_layers, dim)
        self.rms_final_weight = None  # (dim,)

        self.wq = None  # (n_layers,head_size * n_heads, dim)
        self.wk = None  # (n_layers, dim, kv_dim)
        self.wv = None  # (n_layers, kv_dim, dim)
        self.wo = None  # (n_layers, dim, q_dim)

        if config.has_qk_norm:
            self.rms_q_weight = None
            self.rms_k_weight = None

        if config.merge_matmul_check:
            self.w13 = None  # for cat(w1, w3) check only
        else:
            self.w1 = None  # torch.empty(n_layers, hidden_dim, dim, device=dev)
            self.w2 = None  # torch.empty(n_layers, dim, hidden_dim, device=dev)
            self.w3 = None  # torch.empty(n_layers, hidden_dim, dim, device=dev)

        self.wcls = None  # (vocab_size, dim) or shared with token_embedding_table
        self.freq = None
        self.scale = math.sqrt(head_size)

    def to_dtype(self, config, dtype: torch.dtype):
        attr_list = ['token_embedding_table', 'rms_att_weight', 'rms_ffn_weight', 'rms_final_weight',
                     'wq', 'wk', 'wv', 'wo', 'w13', 'wcls']
        if config.merge_matmul_check:
            attr_list.append('w13')
        else:
            attr_list.extend(['w1', 'w2', 'w3'])

        for attr in attr_list:
            tensor = getattr(self, attr, None)
            if tensor is not None:
                setattr(self, attr, tensor.to(dtype=dtype))

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
        head_size = config.head_dim
        type = config.torch_dtype

        ptr = 0  # current offset in the 1D data tensor

        def slice_tensor(shape, device=device):
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
        self.token_embedding_table = slice_tensor((vocab_size, dim), device='cpu' if config.token_table_cpu else device)
        self.rms_att_weight = slice_tensor((n_layers, dim))
        self.wq = slice_tensor((n_layers, head_size * n_heads, dim))
        self.wk = slice_tensor((n_layers, dim, n_kv_heads * head_size))
        self.wv = slice_tensor((n_layers, dim, n_kv_heads * head_size))
        self.wo = slice_tensor((n_layers, dim, n_heads * head_size))
        if config.has_qk_norm:
            self.rms_q_weight = slice_tensor((n_layers, head_size))
            self.rms_k_weight = slice_tensor((n_layers, head_size))

        self.rms_ffn_weight = slice_tensor((n_layers, dim))
        w1 = slice_tensor((n_layers, hidden_dim, dim))
        self.w2 = slice_tensor((n_layers, dim, hidden_dim))
        w3 = slice_tensor((n_layers, hidden_dim, dim))

        if not config.merge_matmul_check:
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

        head_dim = torch.arange(0, config.rope_range, 2, device=self.dev)
        self.freq = 1.0 / (config.rope_theta ** (head_dim.float() / head_size))

        if config.q_type != config.torch_dtype:
            self.to_dtype(config, config.q_type)


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
        kv_dim = n_kv_heads * config.head_dim
        type = config.q_type
        head_size = config.head_dim

        dev = device

        self.pos_tensor = torch.tensor([0], dtype=torch.int32, device='cuda')

        # allocate buffers, initialized to zero (like calloc)
        self.batch_x = torch.zeros(config.max_batch_size, dim, device=dev, dtype=type)
        self.x = torch.zeros(dim, device=dev, dtype=type)
        self.xb = torch.zeros(dim, device=dev, dtype=type)
        self.xb2 = torch.zeros(dim, device=dev, dtype=type)
        self.atten_out = torch.zeros(head_size * n_heads, device=dev, dtype=type)
        if config.merge_matmul_check:
            self.hbm = torch.zeros(hidden_dim * 2, device=dev, dtype=type)
            self.hbm32 = torch.zeros(hidden_dim * 2, device=dev, dtype=torch.float32)
        else:
            self.hb = torch.zeros(hidden_dim, device=dev, dtype=type)
            self.hb2 = torch.zeros(hidden_dim, device=dev, dtype=type)
        qkv_device = 'cuda'
        self.q = torch.zeros(n_heads, dim // n_heads, device=qkv_device, dtype=type)
        self.key_cache = torch.zeros(n_layers, seq_len, kv_dim, device=qkv_device, dtype=type)
        self.value_cache = torch.zeros(n_layers, seq_len, kv_dim, device=qkv_device, dtype=type)
        self.logits = torch.zeros(vocab_size, device=dev, dtype=type)


class Transformer:
    def __init__(self):
        self.config = None
        self.weights = None
        self.state = None

        self.fd = -1
        self.file_size = 0
        self.mmap_obj = None
        self.data = None
        self.cuda_graph = None
        self._graph_constructed = False
        self.merge_matmul_check = os.getenv('MERGE_MATMUL') == '1'

    def read_checkpoint(self, checkpoint_path: str, config_type=torch.float32, to_type=None):
        self.checkpoint_path = checkpoint_path

        with open(checkpoint_path, "rb") as f:
            offset = 0
            if self.config is None:
                config_bytes = f.read(Config.size())  # config_class.size() = sizeof(Config)
                config = Config.from_bytes(config_bytes)
                config.torch_dtype = config_type
                if to_type is not None:
                    config.q_type = to_type
                self.config = config
                offset = Config.size()
            config = self.config
            shared_weights = 1 if config.vocab_size > 0 else 0
            config.vocab_size = abs(config.vocab_size)

            # figure out file size
            f.seek(0, os.SEEK_END)
            file_size = f.tell()

            self.fd = os.open(self.checkpoint_path, os.O_RDONLY)
            self.mmap_obj = mmap.mmap(self.fd, file_size, access=mmap.ACCESS_READ)

            # number of floats
            n_floats = (file_size - offset) // config_type.itemsize
            data = torch.frombuffer(self.mmap_obj[offset:], dtype=config_type)

            # 4. map weights
            weights = TransformerWeights(config)
            weights.memory_map_weights(config, data, shared_weights, device=config.device)
            self.weights = weights

            if self.config.device == 'cuda':
                self.mmap_obj.close()
                self.mmap_obj = None

    def read_hf_config(self, config_path: str = None):
        if config_path is None:
            return
        import json
        with open(config_path, 'r') as f:
            hf_config = json.load(f)
            self.config = Config()
            self.config.from_json(hf_config)

    def build_transformer(self, checkpoint_path: str, config_path: str = None):
        self.checkpoint_path = checkpoint_path

        self.read_hf_config(config_path)
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
