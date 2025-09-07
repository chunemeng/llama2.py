import triton
import triton.language as tl


@triton.jit
def ffn_kernel_wo_rms(
        state_x_ptr,  # [D]
        wo_ptr,  # [D, D]
        w1_ptr, w3_ptr, w2_ptr,  # FFN weights
        D: tl.constexpr,  # 向量维度
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D

    # ---------------- Residual 1: xb2 = state.x @ wo ----------------
    x_val = tl.load(state_x_ptr + offsets, mask=mask, other=0.0)  # [BLOCK_SIZE]
    wo_val = tl.load(wo_ptr + offsets[:, None] * D + offsets[None, :], mask=mask[:, None] & mask[None, :], other=0.0)
    xb2 = tl.sum(x_val * wo_val, axis=1)
    x_val += xb2  # residual 1
    tl.store(state_x_ptr + offsets, x_val, mask=mask)

    # ---------------- RMSNorm ----------------
    rms = tl.sqrt(tl.sum(x_val * x_val, axis=0) / D + 1e-5)
    x_norm = x_val / rms
    rms_w = tl.load(rms_weight_ptr + offsets, mask=mask, other=1.0)
    xb = x_norm * rms_w

    # ---------------- Merge w1 & w3 matmul ----------------
    w1_val = tl.load(w1_ptr + offsets[:, None] * D + offsets[None, :], mask=mask[:, None] & mask[None, :], other=0.0)
    w3_val = tl.load(w3_ptr + offsets[:, None] * D + offsets[None, :], mask=mask[:, None] & mask[None, :], other=0.0)
    # 合并列
    hb = tl.sum(xb * w1_val, axis=1)
    hb2 = tl.sum(xb * w3_val, axis=1)

    # ---------------- Swiglu + w2 matmul ----------------
    # Swiglu: hb * sigmoid(hb2)
    hb_swiglu = hb * tl.sigmoid(hb2)

    w2_val = tl.load(w2_ptr + offsets[:, None] * D + offsets[None, :], mask=mask[:, None] & mask[None, :], other=0.0)
    xb_out = tl.sum(hb_swiglu * w2_val, axis=1)

    # ---------------- Residual 2 ----------------
    x_val += xb_out
    tl.store(state_x_ptr + offsets, x_val, mask=mask)
