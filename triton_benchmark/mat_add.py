import triton
import triton.language as tl
import torch
DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")


## to-do
# triton_kernel
# python wrapper
# benchmark

def cdiv(a,b):
    return (a + b - 1) // b

@triton.jit
def _mat_add_kernel(a_ptr, b_ptr, c_ptr,
                    n_element:tl.constexpr,
                    BLOCK_SIZE:tl.constexpr):

        PID = tl.program_id(axis=0)

        a_start_ptr = a_ptr + (PID * BLOCK_SIZE)
        b_start_ptr = b_ptr + (PID * BLOCK_SIZE)

        col_offset= tl.arange(0, BLOCK_SIZE)
        a_row = a_start_ptr + col_offset
        b_row = b_start_ptr + col_offset

        masked = col_offset < n_element

        a_data = tl.load(a_row, mask = masked, other = float('-inf'))
        b_data = tl.load(b_row, mask = masked, other = float('-inf'))

        output = a_data + b_data
    
        c_start_ptr = c_ptr + (PID * BLOCK_SIZE)
        c_row = c_start_ptr + col_offset
        tl.store(c_row, output, mask=masked)

def matrix_add(a,b):
    assert a.device==b.device == DEVICE
    assert a.numel()==b.numel()

    num_element = a.numel()
    BLOCK_SIZE= 1024
    grid_size = cdiv(num_element,BLOCK_SIZE)
    grid = (grid_size,)
    num_warps = 4
    if BLOCK_SIZE>2048:
        num_warps = 8
    if BLOCK_SIZE >4096:
        num_warps = 16

    c_out = torch.empty_like(a)

    _mat_add_kernel[grid](a,b,c_out,
                    num_element,
                    BLOCK_SIZE)

    return c_out


def test_mat_add_kernel(size:tuple, device = DEVICE, atol=1e-3, rtol=1e-2):
    a = torch.randn(size, dtype=torch.float16, device = device) 
    b = torch.randn(size, dtype=torch.float16, device = device)

    assert a.ndim==b.ndim==2, f"only accept two dimension"

    torch.allclose(matrix_add(a,b), torch.add(a,b), atol=atol, rtol=rtol)
    print("PASSED!")


## benchmarking
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names = ['size'],
        x_vals =
            [2**i for i in range(10,30,1)],
        x_log = True,
        line_arg = 'provider',
        line_vals = ['triton', 'torch'],
        line_names = ['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel = "GB/s",
        plot_name = 'vector_add_performance',
        args = {}
    )
)

def benchmark(size, provider):
  # Create 2D tensors (1 row, 'size' columns) for vector-like operations
  x = torch.randn(1, size, device =DEVICE, dtype = torch.float32) # Removed torch.tensor() wrapper
  y = torch.randn(1, size, device =DEVICE, dtype= torch.float32) # Removed torch.tensor() wrapper

  quantiles = [0.5, 0.05, 0.95]

  if provider == 'torch':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.add(x,y), quantiles= quantiles)

  if provider == "triton":
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: matrix_add(x,y), quantiles = quantiles)

  gbps = lambda ms: 12* size/ms * 1e-6
  return gbps(ms), gbps(max_ms), gbps(min_ms)



if __name__=="__main__":
   test_mat_add_kernel((512,512))
  # import sys
 #  if len(sys.argv)> 1 and sys.argv[1]=='--benchmark':
   benchmark.run(print_data=True, show_plots=True, save_path = './triton_benchmark')