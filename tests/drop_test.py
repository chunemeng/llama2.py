import math

import torch
import triton
import bench_util
import kernel.dropout as dropout


def _norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def forward(x, weight, eps=1e-6):
    output = _norm(x.float()).type_as(x)
    return output * weight


def verify_rms(num_tests=10, max_M=4096, max_N=4096):
    for test_id in range(1, num_tests + 1):
        M = torch.randint(1, max_M + 1, (1,)).item()
        N = torch.randint(1, max_N + 1, (1,)).item()
        drop_prob = torch.rand(1).item() * 0.5

        input_tensor = torch.randn(M, N, device='cuda', dtype=torch.float32)
        output_triton = torch.zeros(M, N, device='cuda', dtype=torch.float32)

        output_torch = torch.dropout(input_tensor, p=drop_prob, train=True)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_ROW_SIZE']), triton.cdiv(N, META['BLOCK_COL_SIZE']))
        dropout.dropout_kernel[grid](
            input_tensor, output_triton,
            1 - drop_prob,
            M, N,
            BLOCK_ROW_SIZE=32,
            BLOCK_COL_SIZE=32
        )

        if output_triton.max() != output_triton.max():
            print(f"❌ Test {test_id}: M={M}, N={N} not equal")
            continue

        count_torch = output_torch.nonzero().size(0)

        count_triton = output_triton.nonzero().size(0)

            # 计算比例
        ratio_torch = count_torch / (M * N)
        ratio_triton = count_triton / (M * N)
        ratio_diff = abs(ratio_torch - ratio_triton)

        # 理论标准差
        sigma = math.sqrt(drop_prob * (1 - drop_prob) / (M * N))

        print(f"M={M}, N={N}, drop_prob={drop_prob:.4f}")
        print(f"PyTorch drop count={count_torch}, ratio={ratio_torch:.4f}")
        print(f"Triton  drop count={count_triton}, ratio={ratio_triton:.4f}")
        print(f"Ratio diff={ratio_diff:.6f}, theoretical sigma={sigma:.6f}")

        # 判断是否合理
        if ratio_diff <= 6 * sigma:
            print("✅ Triton dropout close to PyTorch (between 6 sigma)")
        else:
            print("⚠️ Triton dropout differs from PyTorch (beyond 6 sigma)")


line_vals = ['torch', 'triton']


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # 用作图表x轴的参数名。
        x_vals=[128 * i for i in range(2, 150, 20)],  # `x_name`的不同可能值。
        x_log=True,  # x轴是对数的。
        line_arg='provider',  # 其值对应图表中不同线条的参数名。
        line_vals=line_vals,  # `line_arg`的可能值。
        line_names=line_vals,  # 线条的标签名。
        styles=bench_util.build_styles(),  # 线条样式。
        ylabel='GB/s',  # y轴的标签名。
        plot_name='dropout',  # 图表的名称。也用作保存图表的文件名。
        args={},  # 不在`x_names`和`y_name`中的函数参数值。
    ))
def benchmark(size, provider):
    print(provider, size)

    M = size
    K = size
    N = size
    a = torch.randn((M, N), device='cuda', dtype=torch.float16)
    b = torch.randn((M, N), device='cuda', dtype=torch.float16)
    c = torch.zeros(M, device='cuda', dtype=torch.float16)
    p = 0.9

    def torch_rms(intput,out):
        torch.cuda.synchronize()
        out = torch.dropout(intput, p=1 - p, train=True)
        torch.cuda.synchronize()

    def triton_rms(intput, out):
        torch.cuda.synchronize()
        grid = lambda META: (triton.cdiv(M, META['BLOCK_ROW_SIZE']), triton.cdiv(N, META['BLOCK_COL_SIZE']))
        dropout.dropout_kernel[grid](
            intput, out,
            p,
            M, N,
        )
        torch.cuda.synchronize()

    quantiles = [0.5, 0.2, 0.9]
    ms = 1.0
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch_rms(a,c),
                                     quantiles=quantiles)
    if provider == 'triton':
        ms = triton.testing.do_bench(
            lambda: triton_rms(a, c), quantiles=quantiles)

    # torch always better than triton here
    gbps_list = lambda ms: [4 * M * N / (_ * 1e-3) * 1e9 for _ in ms]
    return gbps_list(ms)

if __name__ == "__main__":
    # verify_rms(num_tests=20, max_M=2048, max_N=2048)
    benchmark.run(show_plots=True, save_path='./output', print_data=True)
    # verify_rms(num_tests=1000, max_M=4096, max_N=4096)