
import sys
import argparse
#for triton
import torch
import triton
import triton.language as tl;
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

#for add_kernel
@triton.jit
def add_kernel(
    x_ptr, 
    y_ptr, 
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask = mask, other = None)
    y = tl.load(y_ptr + offsets, mask = mask, other = None)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
    
def add(
    x: torch.Tensor, 
    y: torch.Tensor
) -> torch.Tensor:
    assert x.shape == y.shape, "Input tensors must have the same shape"
    output = torch.empty_like(x)
    N = output.numel()

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, N, BLOCK_SIZE= 1024)
    return output

def test_add_kernel(size, atol=1e-3, rtol=1e-3, device=DEVICE):
    torch.manual_seed(0)
    x = torch.randn(size, device=device)
    y = torch.randn(size, device=device)
    output = add(x, y)
    expected = x + y
    torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)
    print("Test passed!")


#for test and benchmark
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'], 
        x_vals=[2**i for i in range(12, 28, 1)], 
        x_log=True,
        line_arg='provider', 
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='vector-addition-triton-vs-torch',
        args={}
    )
)

def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)

    quantiles = [0.5, 0.05, 0.95]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: x + y, 
            quantiles=quantiles
        )
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add(x, y), 
            quantiles=quantiles
        )
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(min_ms), gbps(max_ms)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--size', type=int, default=2**20, help='Size of vectors')

    if '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    if args.benchmark:
        benchmark.run(save_path='.', print_data=False)
    elif args.size:
        test_add_kernel(args.size)