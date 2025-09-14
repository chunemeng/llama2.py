import torch

from ops.time_util import time_in_ms, store_time


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
