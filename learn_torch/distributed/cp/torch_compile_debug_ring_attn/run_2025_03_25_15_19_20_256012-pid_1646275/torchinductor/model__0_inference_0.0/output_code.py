# AOT ID: ['0_inference']
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


cpp_fused_clone_0 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include "/tmp/torchinductor_goon/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(4L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(2L); x1+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(32L); x2+=static_cast<int64_t>(1L))
                {
                    for(int64_t x3=static_cast<int64_t>(0L); x3<static_cast<int64_t>(64L); x3+=static_cast<int64_t>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x3 + (128L*x2) + (16384L*x0)), static_cast<int64_t>(16));
                        tmp0.store(out_ptr0 + static_cast<int64_t>(x3 + (64L*x2) + (2048L*x1) + (4096L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_exp_masked_fill_max_sub_1 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*', 'float*', 'float*'], '''
#include "/tmp/torchinductor_goon/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(32L); x1+=static_cast<int64_t>(1L))
            {
                {
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(32L); x2+=static_cast<int64_t>(16L))
                    {
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x2 + (32L*x1) + (1024L*x0)), static_cast<int64_t>(16));
                        auto tmp0 = x2 + ((-1L)*x1);
                        auto tmp1 = c10::convert<int64_t>(tmp0);
                        auto tmp2 = at::vec::VectorizedN<int64_t,2>::arange(tmp1, 1);
                        auto tmp3 = static_cast<int64_t>(1);
                        auto tmp4 = at::vec::VectorizedN<int64_t,2>(tmp3);
                        auto tmp5 = at::vec::VecMask<int64_t,2>(tmp2 >= tmp4);
                        auto tmp6 = static_cast<bool>(true);
                        auto tmp7 = at::vec::VecMask<float,1>::from(tmp6);
                        auto tmp8 = tmp5 & tmp7;
                        auto tmp10 = static_cast<float>(0.125);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp13 = -std::numeric_limits<float>::infinity();
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = decltype(tmp14)::blendv(tmp12, tmp14, tmp8.template cast<float,1>());
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp15);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<int64_t>(x1 + (32L*x0))] = static_cast<float>(tmp_acc0);
                }
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(32L); x2+=static_cast<int64_t>(16L))
                {
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x2 + (32L*x1) + (1024L*x0)), static_cast<int64_t>(16));
                    auto tmp16 = out_ptr0[static_cast<int64_t>(x1 + (32L*x0))];
                    auto tmp0 = x2 + ((-1L)*x1);
                    auto tmp1 = c10::convert<int64_t>(tmp0);
                    auto tmp2 = at::vec::VectorizedN<int64_t,2>::arange(tmp1, 1);
                    auto tmp3 = static_cast<int64_t>(1);
                    auto tmp4 = at::vec::VectorizedN<int64_t,2>(tmp3);
                    auto tmp5 = at::vec::VecMask<int64_t,2>(tmp2 >= tmp4);
                    auto tmp6 = static_cast<bool>(true);
                    auto tmp7 = at::vec::VecMask<float,1>::from(tmp6);
                    auto tmp8 = tmp5 & tmp7;
                    auto tmp10 = static_cast<float>(0.125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = -std::numeric_limits<float>::infinity();
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = decltype(tmp14)::blendv(tmp12, tmp14, tmp8.template cast<float,1>());
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 - tmp17;
                    auto tmp19 = tmp18.exp();
                    tmp19.store(out_ptr1 + static_cast<int64_t>(x2 + (32L*x1) + (1024L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(4L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(32L); x1+=static_cast<int64_t>(1L))
            {
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(64L); x2+=static_cast<int64_t>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x2 + (128L*x1) + (16384L*x0)), static_cast<int64_t>(16));
                    tmp0.store(out_ptr2 + static_cast<int64_t>(x2 + (64L*x1) + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_2 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include "/tmp/torchinductor_goon/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(4L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(32L); x1+=static_cast<int64_t>(1L))
            {
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(64L); x2+=static_cast<int64_t>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x2 + (128L*x1) + (16384L*x0)), static_cast<int64_t>(16));
                    tmp0.store(out_ptr0 + static_cast<int64_t>(x2 + (64L*x1) + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_3 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include "/tmp/torchinductor_goon/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(4L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(2L); x1+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(32L); x2+=static_cast<int64_t>(1L))
                {
                    for(int64_t x3=static_cast<int64_t>(0L); x3<static_cast<int64_t>(64L); x3+=static_cast<int64_t>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x3 + (128L*x2) + (16384L*x0)), static_cast<int64_t>(16));
                        tmp0.store(out_ptr0 + static_cast<int64_t>(x3 + (64L*x2) + (2048L*x1) + (4096L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_div_sum_4 = async_compile.cpp_pybinding(['float*', 'const float*', 'float*'], '''
#include "/tmp/torchinductor_goon/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(256L); x0+=static_cast<int64_t>(1L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(32L); x1+=static_cast<int64_t>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + (32L*x0)), static_cast<int64_t>(16));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
            }
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(16L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + (64L*x0)), static_cast<int64_t>(16));
                auto tmp1 = out_ptr0[static_cast<int64_t>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<int64_t>(x1 + (64L*x0)));
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2, 4, 32, 64), (32768, 8192, 64, 1))
    assert_size_stride(arg1_1, (2, 2, 32, 64), (32768, 16384, 128, 1))
    assert_size_stride(arg2_1, (2, 2, 32, 64), (32768, 16384, 128, 1))
    buf0 = empty_strided_cpu((2, 2, 2, 32, 64), (8192, 4096, 2048, 64, 1), torch.float32)
    cpp_fused_clone_0(arg1_1, buf0)
    buf1 = empty_strided_cpu((8, 32, 32), (1024, 32, 1), torch.float32)
    # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(arg0_1, (8, 32, 64), (8192, 64, 1), 0), reinterpret_tensor(buf0, (8, 64, 32), (2048, 1, 64), 0), out=buf1)
    del arg0_1
    buf2 = empty_strided_cpu((2, 4, 32, 1), (128, 32, 1, 256), torch.float32)
    buf24 = empty_strided_cpu((2, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
    buf4 = empty_strided_cpu((2, 2, 32, 64), (4096, 2048, 64, 1), torch.float32)
    cpp_fused_clone_div_exp_masked_fill_max_sub_1(buf1, arg1_1, buf2, buf24, buf4)
    del arg1_1
    del buf1
    # Topologically Sorted Source Nodes: [contiguous, tensor], Original ATen: [aten.clone, _c10d_functional.all_to_all_single]
    buf5 = torch.ops._c10d_functional.all_to_all_single.default(buf4, [0, 0, 0, 2], [0, 2, 0, 0], '0')
    assert_size_stride(buf5, (2, 2, 32, 64), (4096, 2048, 64, 1))
    # Topologically Sorted Source Nodes: [k_1], Original ATen: [_c10d_functional.wait_tensor]
    torch.ops._c10d_functional.wait_tensor.default(buf5)
    del buf4
    # Topologically Sorted Source Nodes: [tensor_2], Original ATen: [_c10d_functional.all_to_all_single]
    buf8 = torch.ops._c10d_functional.all_to_all_single.default(buf5, [0, 0, 0, 2], [0, 2, 0, 0], '0')
    assert_size_stride(buf8, (2, 2, 32, 64), (4096, 2048, 64, 1))
    # Topologically Sorted Source Nodes: [k_2], Original ATen: [_c10d_functional.wait_tensor]
    torch.ops._c10d_functional.wait_tensor.default(buf8)
    del buf5
    # Topologically Sorted Source Nodes: [tensor_4], Original ATen: [_c10d_functional.all_to_all_single]
    buf11 = torch.ops._c10d_functional.all_to_all_single.default(buf8, [0, 0, 0, 2], [0, 2, 0, 0], '0')
    assert_size_stride(buf11, (2, 2, 32, 64), (4096, 2048, 64, 1))
    # Topologically Sorted Source Nodes: [k_3], Original ATen: [_c10d_functional.wait_tensor]
    torch.ops._c10d_functional.wait_tensor.default(buf11)
    del buf11
    buf14 = buf8; del buf8  # reuse
    cpp_fused_clone_2(arg2_1, buf14)
    # Topologically Sorted Source Nodes: [contiguous_1, tensor_1], Original ATen: [aten.clone, _c10d_functional.all_to_all_single]
    buf15 = torch.ops._c10d_functional.all_to_all_single.default(buf14, [0, 0, 0, 2], [0, 2, 0, 0], '0')
    assert_size_stride(buf15, (2, 2, 32, 64), (4096, 2048, 64, 1))
    # Topologically Sorted Source Nodes: [v_1], Original ATen: [_c10d_functional.wait_tensor]
    torch.ops._c10d_functional.wait_tensor.default(buf15)
    del buf14
    # Topologically Sorted Source Nodes: [tensor_3], Original ATen: [_c10d_functional.all_to_all_single]
    buf18 = torch.ops._c10d_functional.all_to_all_single.default(buf15, [0, 0, 0, 2], [0, 2, 0, 0], '0')
    assert_size_stride(buf18, (2, 2, 32, 64), (4096, 2048, 64, 1))
    # Topologically Sorted Source Nodes: [v_2], Original ATen: [_c10d_functional.wait_tensor]
    torch.ops._c10d_functional.wait_tensor.default(buf18)
    del buf15
    # Topologically Sorted Source Nodes: [tensor_5], Original ATen: [_c10d_functional.all_to_all_single]
    buf21 = torch.ops._c10d_functional.all_to_all_single.default(buf18, [0, 0, 0, 2], [0, 2, 0, 0], '0')
    assert_size_stride(buf21, (2, 2, 32, 64), (4096, 2048, 64, 1))
    # Topologically Sorted Source Nodes: [v_3], Original ATen: [_c10d_functional.wait_tensor]
    torch.ops._c10d_functional.wait_tensor.default(buf21)
    del buf18
    del buf21
    buf25 = buf0; del buf0  # reuse
    cpp_fused_clone_3(arg2_1, buf25)
    del arg2_1
    buf26 = empty_strided_cpu((8, 32, 64), (2048, 64, 1), torch.float32)
    # Topologically Sorted Source Nodes: [numerator], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf24, (8, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf25, (8, 32, 64), (2048, 64, 1), 0), out=buf26)
    del buf25
    buf27 = buf2; del buf2  # reuse
    buf28 = reinterpret_tensor(buf26, (2, 4, 32, 64), (8192, 2048, 64, 1), 0); del buf26  # reuse
    cpp_fused_div_sum_4(buf28, buf24, buf27)
    return (buf28, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 4, 32, 64), (32768, 8192, 64, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((2, 2, 32, 64), (32768, 16384, 128, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((2, 2, 32, 64), (32768, 16384, 128, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
