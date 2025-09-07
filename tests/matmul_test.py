import torch
import triton

from kernel.matmul import matvec_kernel
# import bench_util


def matvec_torch(x, w):
    return w @ x

# -------------------- Verification function -------------------- #
def verify_matvec(num_tests=2, max_D=512, max_N=512):
    for test_id in range(1, num_tests + 1):
        D = 768
        N = 768

        x = torch.randn(N, device='cuda', dtype=torch.float32)
        w = torch.randn(D, N, device='cuda', dtype=torch.float32)
        out_triton = torch.zeros(D, device='cuda', dtype=torch.float32)
        out_torch = matvec_torch(x, w)

        grid = lambda META: (triton.cdiv(D, META['BLOCK_D']),)
        matvec_kernel[grid](x, w, out_triton, D, N)

        if torch.allclose(out_triton, out_torch, rtol=1e-5, atol=1e-6):
            print(f"✅ Test {test_id}: D={D}, N={N} 结果一致")
        else:
            diff = (out_triton - out_torch).abs().max()
            print(f"❌ Test {test_id}: D={D}, N={N} 最大误差={diff.item()}")

# -------------------- Run verification -------------------- #


line_vals = ['torch', 'triton']
#
#
# @triton.testing.perf_report(
#     triton.testing.Benchmark(
#         x_names=['size'],  # 用作图表x轴的参数名。
#         x_vals=[128 * i for i in range(2, 40, 2)],  # `x_name`的不同可能值。
#         x_log=True,  # x轴是对数的。
#         line_arg='provider',  # 其值对应图表中不同线条的参数名。
#         line_vals=line_vals,  # `line_arg`的可能值。
#         line_names=line_vals,  # 线条的标签名。
#         styles=bench_util.build_styles(),  # 线条样式。
#         ylabel='GB/s',  # y轴的标签名。
#         plot_name='matvec',  # 图表的名称。也用作保存图表的文件名。
#         args={},  # 不在`x_names`和`y_name`中的函数参数值。
#     ))
# def benchmark(size, provider):
#     print(provider, size)
#
#     D = size
#     N = size
#     a = torch.randn(D, N, device='cuda', dtype=torch.float16)
#     b = torch.randn(N, device='cuda', dtype=torch.float16)
#     c = torch.zeros(D, device='cuda', dtype=torch.float16)
#
#     def torch_rms(intput, w, out):
#         torch.cuda.synchronize()
#         out = matvec_torch(intput, w)
#         torch.cuda.synchronize()
#
#     def triton_rms(intput, w, out):
#         torch.cuda.synchronize()
#         grid = lambda META: (triton.cdiv(D, META['BLOCK_D']),)
#         matvec_kernel[grid](intput, w, out, D, N)
#         torch.cuda.synchronize()
#
#     quantiles = [0.5, 0.2, 0.9]
#     ms = 1.0
#     if provider == 'torch':
#         ms = triton.testing.do_bench(lambda: torch_rms(a, b, c),
#                                      quantiles=quantiles)
#     if provider == 'triton':
#         ms = triton.testing.do_bench(
#             lambda: triton_rms(a, b, c), quantiles=quantiles)
#     # if provider == 'softmax1':
#     #     ms = triton.testing.do_bench(
#     #         lambda: softmax1(a, b), quantiles=quantiles)
#     # if provider == 'softmax2':
#     #     ms = triton.testing.do_bench(
#     #         lambda: softmax2(a, b), quantiles=quantiles)
#     # if provider == 'softmax3way':
#     #     ms = triton.testing.do_bench(
#     #         lambda: softmax(a, b), quantiles=quantiles)
#     # if provider == 'softmax_2way':
#     #     ms = triton.testing.do_bench(
#     #         lambda: softmax3(a, b), quantiles=quantiles)
#
#     gbps_list = lambda ms: [4 * D * N / (_ * 1e-3) * 1e9 for _ in ms]
#     return gbps_list(ms)
# verify_matvec()
if __name__ == "__main__":
    verify_matvec()
    # benchmark.run(show_plots=True, save_path='./output', print_data=True)