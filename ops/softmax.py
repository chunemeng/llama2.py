import torch

from ops.time_util import time_in_ms, store_time


def softmax(x, dim):
    t = time_in_ms()
    res = torch.softmax(x, dim=dim)
    t2 = time_in_ms()
    store_time('softmax_cpu', t2 - t)
    return res
