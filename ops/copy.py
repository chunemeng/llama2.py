from ops.time_util import time_in_ms, store_time


def copy(dst, src):
    """Copy tensor from src to dst."""
    t0 = time_in_ms()
    if dst.device != src.device:
        dst.copy_(src.to(dst.device))
    else:
        dst.copy_(src)
    t1 = time_in_ms()
    store_time('copy', t1 - t0)
    return src
