import triton
import torch
import torch.nn as nn
import triton.language as tl
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

def naive_matmul(x,y):
    x_r, x_c = x.shape
    y_r, y_c = y.shape

    assert x_c ==y_r, f"incompatible matrixes"

    c = torch.zeros(x_r, y_c, dtype=torch.float32, device = x.device)

    for i in range(x_r):
        for j in range(y_c):
            accum = 0
            for k in range(x_c):
                accum+= x[i,k]*y[k,j]
            
            c[i,j]= accum

    return c

def ceildiv(a,b):
    return (a+b-1)//b
autotune_config=[
    triton.Config({'BLOCK_SIZE_M':128, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':64, 'GROUP_SIZE':8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':32, 'GROUP_SIZE':8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M':128, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':32, 'GROUP_SIZE':8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M':128, 'BLOCK_SIZE_N':64, 'BLOCK_SIZE_K':32, 'GROUP_SIZE':8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':32, 'GROUP_SIZE':8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M':128, 'BLOCK_SIZE_N':32, 'BLOCK_SIZE_K':32, 'GROUP_SIZE':8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':32, 'BLOCK_SIZE_K':32, 'GROUP_SIZE':8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M':32, 'BLOCK_SIZE_N':64, 'BLOCK_SIZE_K':32, 'GROUP_SIZE':8}, num_stages=5, num_warps=2),
]
@triton.autotune(configs = autotune_config, key = ['M', 'N', "k"])
@triton.jit
def _matmul_kernel(
    x_ptr, y_ptr, z_ptr,
    M,N,K, x_row_stride, x_col_stride,
    y_row_stride, y_col_stride,
    z_row_stride, z_col_stride,
    BLOCK_SIZE_M:tl.constexpr,
    BLOCK_SIZE_N:tl.constexpr,
    BLOCK_SIZE_K:tl.constexpr,
    GROUP_SIZE:tl.constexpr
):
    PID = tl.program_id(0)
    num_pid_along_m =tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_along_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_in_group = GROUP_SIZE * num_pid_along_n
    group_id = PID//pid_in_group

    first_group_pid_m = group_id * GROUP_SIZE
    group_size_adj = min(num_pid_along_m - first_group_pid_m, GROUP_SIZE)
    pid_m = first_group_pid_m + ((PID%pid_in_group) % group_size_adj)
    pid_n = (PID % pid_in_group)//group_size_adj

    offset_m = pid_m *BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M)
    offset_n = pid_n *BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N)
    offset_k = tl.arange(0,BLOCK_SIZE_K)

    x_offset = offset_m[:, None]*x_row_stride + offset_k[None, :]*x_col_stride
    y_offset = offset_k[:,None]*y_row_stride + offset_n[None, :]* y_col_stride

    accumulators = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)



    
    #pid_n = tl.program_id(1)

    #row_starter_m = pid_m * BLOCK_SIZE_M
    #col_offset_m = row_starter_m + tl.arange(0, BLOCK_SIZE_M)

    #row_starter_n  = pid_n * BLOCK_SIZE_N
    #col_offset_n = row_starter_n + tl.arange(0, BLOCK_SIZE_N)

    #offset_k = tl.arange(0,BLOCK_SIZE_K)

    #accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        col_offset_k = k *BLOCK_SIZE_K + offset_k

        #x_ptrs = x_ptr+x_offset
        #y_ptrs = y_ptr + y_offset
        
        mask = offset_k<K - k*BLOCK_SIZE_K
        #mask_a = (offset_m[:,None]<M) & (offset_k[None, :]< K)
        #mask_y = (offset_k[:,None]< K) & (offset_n[None,:]<N)

        x = tl.load(x_ptr+x_offset, mask=mask[None,:], other = 0.0)
        y = tl.load(y_ptr+ y_offset, mask=mask[:, None], other = 0.0)

        accumulators=tl.dot(x,y, acc=accumulators)

        x_offset += BLOCK_SIZE_K*x_col_stride
        y_offset += BLOCK_SIZE_K * y_row_stride

    z_ptrs = z_ptr+offset_m[:,None]*z_row_stride + offset_n[None,:]*z_col_stride
    mask_z = (offset_m[:,None]<M) & (offset_n[None,:]<N)

    tl.store(z_ptrs, accumulators.to(tl.float16), mask=mask_z)


def matmul(x:torch.tensor, y:torch.tensor):
    assert x.ndim==y.ndim==2
    assert x.shape[1]==y.shape[0]

    M,K = x.shape
    _,N = y.shape

    ##BLOCK_SIZE_M = 1024
    #BLOCK_SIZE_N = 1024
    #BLOCK_SIZE_K= 512
    
    grid = lambda meta: (ceildiv(M, meta['BLOCK_SIZE_M']) * ceildiv(M, meta['BLOCK_SIZE_N']),)

    c_out = torch.zeros(M,N, dtype=torch.float32, device=DEVICE)

    _matmul_kernel[grid](
        x,y,c_out, M,N,K, x.stride(0),x.stride(1),
        y.stride(0), y.stride(1), c_out.stride(0),
        c_out.stride(1),
    )

    return c_out


def check_matmul(M,N, device =DEVICE, atol=1e-3, rtol=1e-2)->bool:
    x = torch.randn(M, N, dtype=torch.float32, device =DEVICE)
    y = torch.randn(N,M, dtype= torch.float32, device =DEVICE)
    assert x.shape[1]==y.shape[0]

    torch.allclose(torch.matmul(x,y), matmul(x,y), atol=atol, rtol=rtol)
    print("PASSED!")


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
    y = torch.randn([N,M], device = DEVICE, dtype=torch.float32)
  
    quantiles = [0.5, 0.05, 0.95]

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(x,y), quantiles= quantiles)

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(x,y), quantiles = quantiles)
  
    if provider == 'naive':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_matmul(x,y), quantiles=quantiles)

    gbps = lambda ms: (12* M*N/ms * 1e-6)
  #gbps = lambda ms: 2*x.numel()*x.element_size()*1e-9
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__=="__main__":
    #check_matmul(10,30)
    benchmark.run(print_data=True, show_plots=True, save_path = './triton_benchmark/matmul')