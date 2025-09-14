from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3Model
import torch

from kernel.rmsnorm import rmsnorm_kernel_one_row_split_col_in_graph
from ops.rmsnorm import rmsnorm
from ops.time_util import time_in_ms

# 替换为你要用的 Qwen-3 模型名，例如 Qwen-3-7B
model_name = "./qwen3-600M"

# 1️⃣ 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# 2️⃣ 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,  # 推荐半精度加速
    device_map="auto"  # 自动分配到 GPU/CPU
)

model.eval()  # 推理模式


def verify_rms(num_tests=2, max_M=4096, max_N=4096):
    for test_id in range(1, num_tests + 1):
        # Randomly generate matrix size
        M = 768
        # N = torch.randint(1, max_N + 1, (1,)).item()
        # W = torch.randn(N, device='cuda', dtype=torch.float32)
        #
        # # Generate random input tensor
        # input_tensor = torch.randn(M, N, device='cuda', dtype=torch.float32)
        # output_triton = torch.zeros(M, N, device='cuda', dtype=torch.float32)
        # output_triton1 = torch.zeros(M, N, device='cuda', dtype=torch.float32)

        # output_torch = forward(x=input_tensor, weight=W, eps=1e-6)
        #
        # col_size = triton.next_power_of_2(N)
        # grid = lambda META: (triton.cdiv(M, META['BLOCK_ROW_SIZE']),)

        input_tensor = torch.randn(M, device='cuda', dtype=torch.float16)
        weight = torch.randn(M, device='cuda', dtype=torch.float16)
        out = torch.zeros_like(input_tensor)
        rmsnorm(input_tensor, weight, eps=1e-6, out=out)
        rmsnorm_kernel_one_row_split_col_in_graph[(1,)](
            input_tensor, weight, out, 0,
            M,
            eps=1e-6,
            BLOCK_COL_SIZE=512,
            num_stages=2
        )

        # # Validate results
        # if torch.allclose(output_triton, output_torch, rtol=1e-5, atol=1e-6):
        #     print(f"✅ Test {test_id}: M={M}, N={N} 结果一致")
        # else:
        #     diff = (output_triton - output_torch).abs().max()
        #     print(f"❌ Test {test_id}: M={M}, N={N} 最大误差={diff.item()}")
        #
        #
        # if torch.allclose(output_triton1, output_torch, rtol=1e-5, atol=1e-6):
        #     print(f"✅ splitk Test {test_id}: M={M}, N={N} 结果一致")
        # else:
        #     diff = (output_triton1 - output_torch).abs().max()
        #     print(f"❌ splitk Test {test_id}: M={M}, N={N} 最大误差={diff.item()}")


if __name__ == "__main__":
    # verify_rms()
    to = time_in_ms()
    ids = tokenizer.encode("One day, Lily met a Shoggoth", return_tensors="pt").to(model.device)
    # 生成文本
    generated_ids = model.generate(ids,
                                   max_new_tokens=256,
                                   temperature=None,
                                   top_p=0.9,
                                   eos_token_id=tokenizer.eos_token_id,
                                   pad_token_id=tokenizer.eos_token_id,
                                   )

    te = time_in_ms()
    print(
        f"tokens: {generated_ids.shape[1]}, time: {te - to} ms, speed: {generated_ids.shape[1] / (te - to * 1.0) * 1000:.2f} tok/s")

    # 解码生成结果

    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print(output_text)
