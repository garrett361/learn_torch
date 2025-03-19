# AOT ID: ['0_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


cpp_fused_relu_0 = async_compile.cpp_pybinding(['float*'], '''
#include "/tmp/torchinductor_goon/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(float* in_out_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(16L); x0+=static_cast<int64_t>(16L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(16L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    tmp1.store(in_out_ptr0 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (1, 32), (32, 1))
    assert_size_stride(primals_2, (16, 32), (32, 1))
    assert_size_stride(primals_3, (32, 16), (16, 1))
    buf0 = empty_strided_cpu((1, 16), (16, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
    extern_kernels.mm(primals_1, reinterpret_tensor(primals_2, (32, 16), (1, 32), 0), out=buf0)
    del primals_2
    buf1 = buf0; del buf0  # reuse
    cpp_fused_relu_0(buf1)
    buf2 = empty_strided_cpu((1, 32), (32, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
    extern_kernels.mm(buf1, reinterpret_tensor(primals_3, (16, 32), (1, 16), 0), out=buf2)
    # Topologically Sorted Source Nodes: [outputs_1], Original ATen: [_c10d_functional.all_reduce]
    torch.ops._c10d_functional.all_reduce_.default(buf2, 'sum', '1')
    # Topologically Sorted Source Nodes: [outputs_1], Original ATen: [_c10d_functional.wait_tensor]
    torch.ops._c10d_functional.wait_tensor.default(buf2)
    return (buf2, primals_1, buf1, primals_3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 32), (32, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((16, 32), (32, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((32, 16), (16, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
