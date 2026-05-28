from sympy.printing.pretty.pretty_symbology import B
import torch
import triton
import triton.language as tl
import torch.nn as nn
import torch.nn.functional as F
DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")

## things to do
# naive softmax
# assert correction
# benchmarking against torch and naive
# implenting triton kernel and wrapper


def naive_softmax(x:torch.tensor, device = DEVICE)->torch.tensor:
    assert x.ndim ==2, f'only accept 2-d tensor'
    r,c = x.shape
    maxed_x = x - x.max(dim=1, keepdim=True)[0]
    num= torch.exp(maxed_x)
    den = torch.sum(num, dim=1,keepdim=True)
    
    return num/den

def ceildiv(a,b):
    return (a+b-1)//b

@triton.jit
def _softmax_kernel(
    a_ptr, a_row_stride,
    out_ptr, out_row_stride,
    n_col,
    BLOCK_SIZE: tl.constexpr
):
    PID= tl.program_id(0)

    a_ptr_start_row = a_ptr +(PID*a_row_stride)
    col_offset = tl.arange(0,BLOCK_SIZE)
    a_pointer = col_offset + a_ptr_start_row

    masked = col_offset<n_col

    data = tl.load(a_pointer, mask=masked, other= float('-inf'))

    maxed_x = data - tl.max(data,axis=0)
    num = tl.exp(maxed_x)
    den= tl.sum(num, axis=0)
    output = num/den

    out_pointer = out_ptr+ (PID*out_row_stride)
    out_data = out_pointer+col_offset

    tl.store(out_data, output, mask=masked)



def triton_softmax(x:torch.tensor)->torch.tensor:
    assert x.ndim==2, f"only accept two dimensional tensor"

    r,c= x.shape

    BLOCK_SIZE = 1024#triton.next_power_of_2(c)
    grid = (r,)

    out = torch.zeros_like(x)

    _softmax_kernel[grid](
        x, x.stride(0),
        out, out.stride(0), c,BLOCK_SIZE
    )
    
    return out

def softmax_test(size:tuple, device=DEVICE, atol=1e-3, rtol=1e-2)->bool:
    assert isinstance(size, tuple)
    a = torch.randn(size, dtype=torch.float32, device=DEVICE)
    assert torch.allclose(torch.softmax(a, dim=1), triton_softmax(a),atol=atol, rtol=rtol)
 
    print('PASSED')


## benchmarking
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names = ['N'],
        x_vals =
            [2**i for i in range(4,16)],
        x_log = True,
        line_arg = 'provider',
        line_vals = ['triton', 'torch','naive'],
        line_names = ['Triton', 'Torch', "Naive"],
        styles=[('blue', '-'), ('green', '-'), ('yellow', '-')],
        ylabel = "GB/s",
        plot_name = 'softmax_performance',
        args = {'M':2048}
    )
)

def benchmark(M,N, provider):
  # Create 2D tensors (1 row, 'size' columns) for vector-like operations
  x = torch.randn([M,N], device =DEVICE, dtype = torch.float32)
  
  quantiles = [0.5, 0.05, 0.95]

  if provider == 'torch':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=1), quantiles= quantiles)

  if provider == "triton":
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_softmax(x), quantiles = quantiles)
  
  if provider == 'naive':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x), quantiles=quantiles)

  gbps = lambda ms: 12* M*N/ms * 1e-6
  #gbps = lambda ms: 2*x.numel()*x.element_size()*1e-9
  return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__=="__main__":
    #softmax_test((5,5), device=DEVICE)
    benchmark.run(print_data=True, show_plots=True, save_path = './triton_benchmark')
