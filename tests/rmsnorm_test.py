import torch
import triton
import bench_util
import kernel.rmsnorm as rmsnorm


def _norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def forward(x, weight, eps=1e-6):
    output = _norm(x.float()).type_as(x)
    return output * weight


def verify_rms(num_tests=10, max_M=4096, max_N=4096):
    for test_id in range(1, num_tests + 1):
        # Randomly generate matrix size
        M = torch.randint(1, max_M + 1, (1,)).item()
        N = torch.randint(1, max_N + 1, (1,)).item()
        W = torch.randn(N, device='cuda', dtype=torch.float32)

        # Generate random input tensor
        input_tensor = torch.randn(M, N, device='cuda', dtype=torch.float32)
        output_triton = torch.zeros(M, N, device='cuda', dtype=torch.float32)
        output_triton1 = torch.zeros(M, N, device='cuda', dtype=torch.float32)

        output_torch = forward(x=input_tensor, weight=W, eps=1e-6)

        col_size = triton.next_power_of_2(N)
        grid = lambda META: (triton.cdiv(M, META['BLOCK_ROW_SIZE']),)

        rmsnorm.rmsnorm_kernel[grid](
            input_tensor, W, output_triton,
            input_tensor.stride(0), input_tensor.stride(1),
            M, N,
            eps=1e-6,
            BLOCK_ROW_SIZE=32,
            COL_SIZE=col_size
        )
        rmsnorm.rmsnorm_kernel_split_col[grid](
            input_tensor, W, output_triton1,
            input_tensor.stride(0), input_tensor.stride(1),
            M, N,
            eps=1e-6,
            BLOCK_ROW_SIZE=32,
            BLOCK_COL_SIZE=32
        )



        # Validate results
        if torch.allclose(output_triton, output_torch, rtol=1e-5, atol=1e-6):
            print(f"✅ Test {test_id}: M={M}, N={N} 结果一致")
        else:
            diff = (output_triton - output_torch).abs().max()
            print(f"❌ Test {test_id}: M={M}, N={N} 最大误差={diff.item()}")


        if torch.allclose(output_triton1, output_torch, rtol=1e-5, atol=1e-6):
            print(f"✅ splitk Test {test_id}: M={M}, N={N} 结果一致")
        else:
            diff = (output_triton1 - output_torch).abs().max()
            print(f"❌ splitk Test {test_id}: M={M}, N={N} 最大误差={diff.item()}")


# , 'softmax1', 'softmax2', 'softmax_2way'
line_vals = ['torch', 'triton', 'triton_splitk']


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
        plot_name='rmsnorm',  # 图表的名称。也用作保存图表的文件名。
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

    def torch_rms(intput, w, out):
        torch.cuda.synchronize()
        out = forward(intput, w, eps=1e-6)
        torch.cuda.synchronize()

    def triton_rms(intput, w, out):
        torch.cuda.synchronize()
        col_size = triton.next_power_of_2(N)
        grid = lambda META: (triton.cdiv(M, META['BLOCK_ROW_SIZE']),)
        rmsnorm.rmsnorm_kernel[grid](
            intput, w, out,
            intput.stride(0), intput.stride(1),
            M, N,
            eps=1e-6,
            BLOCK_ROW_SIZE=32,
            COL_SIZE=col_size
        )
        torch.cuda.synchronize()

    def triton_splitk_rms(intput, w, out):
        torch.cuda.synchronize()
        grid = lambda META: (triton.cdiv(M, META['BLOCK_ROW_SIZE']),)
        rmsnorm.rmsnorm_kernel_split_col[grid](
            intput, w, out,
            intput.stride(0), intput.stride(1),
            M, N,
            eps=1e-6,
            BLOCK_ROW_SIZE=32,
            BLOCK_COL_SIZE=32
        )
        torch.cuda.synchronize()

    quantiles = [0.5, 0.2, 0.9]
    ms = 1.0
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch_rms(a, b, c),
                                     quantiles=quantiles)
    if provider == 'triton':
        if size > 1408:
            return [0.0 for _ in quantiles]
        ms = triton.testing.do_bench(
            lambda: triton_rms(a, b, c), quantiles=quantiles)
    if provider == 'triton_splitk':
        ms = triton.testing.do_bench(
            lambda: triton_splitk_rms(a, b, c), quantiles=quantiles)
    # if provider == 'softmax1':
    #     ms = triton.testing.do_bench(
    #         lambda: softmax1(a, b), quantiles=quantiles)
    # if provider == 'softmax2':
    #     ms = triton.testing.do_bench(
    #         lambda: softmax2(a, b), quantiles=quantiles)
    # if provider == 'softmax3way':
    #     ms = triton.testing.do_bench(
    #         lambda: softmax(a, b), quantiles=quantiles)
    # if provider == 'softmax_2way':
    #     ms = triton.testing.do_bench(
    #         lambda: softmax3(a, b), quantiles=quantiles)

    gbps_list = lambda ms: [4 * M * N / (_ * 1e-3) * 1e9 for _ in ms]
    return gbps_list(ms)

if __name__ == "__main__":
    # verify_rms(num_tests=20, max_M=2048, max_N=2048)
    benchmark.run(show_plots=True, save_path='./output', print_data=True)