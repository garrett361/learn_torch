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
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(2L); x1+=static_cast<int64_t>(1L))
            {
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(2048L); x2+=static_cast<int64_t>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x2 + (2048L*x0)), static_cast<int64_t>(16));
                    tmp0.store(out_ptr0 + static_cast<int64_t>(x2 + (2048L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_div_exp_max_sub_sum_3 = async_compile.cpp_pybinding(['const float*', 'float*', 'float*', 'float*'], '''
#include "/tmp/torchinductor_goon/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(256L); x0+=static_cast<int64_t>(1L))
        {
            {
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(32L); x1+=static_cast<int64_t>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + (32L*x0)), static_cast<int64_t>(16));
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
            }
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(32L); x1+=static_cast<int64_t>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + (32L*x0)), static_cast<int64_t>(16));
                    auto tmp4 = out_ptr0[static_cast<int64_t>(x0)];
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 - tmp5;
                    auto tmp7 = tmp6.exp();
                    tmp7.store(out_ptr1 + static_cast<int64_t>(x1 + (32L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr2[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused_clone_4 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
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
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(2048L); x2+=static_cast<int64_t>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x2 + (2048L*x0)), static_cast<int64_t>(16));
                    tmp0.store(out_ptr0 + static_cast<int64_t>(x2 + (2048L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_div_exp_max_sub_sum_5 = async_compile.cpp_pybinding(['const float*', 'float*', 'float*', 'float*'], '''
#include "/tmp/torchinductor_goon/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(256L); x0+=static_cast<int64_t>(1L))
        {
            {
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(32L); x1+=static_cast<int64_t>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + (32L*x0)), static_cast<int64_t>(16));
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
            }
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(32L); x1+=static_cast<int64_t>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + (32L*x0)), static_cast<int64_t>(16));
                    auto tmp4 = out_ptr0[static_cast<int64_t>(x0)];
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 - tmp5;
                    auto tmp7 = tmp6.exp();
                    tmp7.store(out_ptr1 + static_cast<int64_t>(x1 + (32L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr2[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused_clone_6 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
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
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(2048L); x2+=static_cast<int64_t>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x2 + (2048L*x0)), static_cast<int64_t>(16));
                    tmp0.store(out_ptr0 + static_cast<int64_t>(x2 + (2048L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_exp_max_sub_sum_7 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/tmp/torchinductor_goon/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(256L); x0+=static_cast<int64_t>(1L))
        {
            {
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(32L); x1+=static_cast<int64_t>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + (32L*x0)), static_cast<int64_t>(16));
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
            }
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(32L); x1+=static_cast<int64_t>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + (32L*x0)), static_cast<int64_t>(16));
                    auto tmp4 = out_ptr0[static_cast<int64_t>(x0)];
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 - tmp5;
                    auto tmp7 = tmp6.exp();
                    tmp7.store(out_ptr1 + static_cast<int64_t>(x1 + (32L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr2[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
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
                    tmp0.store(out_ptr3 + static_cast<int64_t>(x2 + (64L*x1) + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_8 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
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
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(2048L); x2+=static_cast<int64_t>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x2 + (2048L*x0)), static_cast<int64_t>(16));
                    tmp0.store(out_ptr0 + static_cast<int64_t>(x2 + (2048L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_9 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
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
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(2048L); x2+=static_cast<int64_t>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x2 + (2048L*x0)), static_cast<int64_t>(16));
                    tmp0.store(out_ptr0 + static_cast<int64_t>(x2 + (2048L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_10 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
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
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(2048L); x2+=static_cast<int64_t>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x2 + (2048L*x0)), static_cast<int64_t>(16));
                    tmp0.store(out_ptr0 + static_cast<int64_t>(x2 + (2048L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_11 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
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


cpp_fused_add_div_exp_maximum_mul_sub_sum_12 = async_compile.cpp_pybinding(['float*', 'float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/tmp/torchinductor_goon/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
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
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(256L); x0+=static_cast<int64_t>(16L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            auto tmp15 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
            auto tmp2 = at::vec::maximum(tmp0, tmp1);
            auto tmp3 = tmp0 - tmp2;
            auto tmp4 = tmp3.exp();
            auto tmp6 = at::vec::maximum(tmp2, tmp5);
            auto tmp7 = tmp2 - tmp6;
            auto tmp8 = tmp7.exp();
            auto tmp10 = at::vec::maximum(tmp6, tmp9);
            auto tmp11 = tmp9 - tmp10;
            auto tmp12 = tmp11.exp();
            auto tmp13 = tmp6 - tmp10;
            auto tmp14 = tmp13.exp();
            auto tmp16 = tmp12 * tmp15;
            auto tmp17 = tmp5 - tmp6;
            auto tmp18 = tmp17.exp();
            auto tmp20 = tmp18 * tmp19;
            auto tmp21 = tmp1 - tmp2;
            auto tmp22 = tmp21.exp();
            auto tmp24 = tmp22 * tmp23;
            auto tmp26 = tmp4 * tmp25;
            auto tmp27 = tmp24 + tmp26;
            auto tmp28 = tmp8 * tmp27;
            auto tmp29 = tmp20 + tmp28;
            auto tmp30 = tmp14 * tmp29;
            auto tmp31 = tmp16 + tmp30;
            tmp4.store(out_ptr1 + static_cast<int64_t>(x0));
            tmp8.store(out_ptr2 + static_cast<int64_t>(x0));
            tmp12.store(out_ptr3 + static_cast<int64_t>(x0));
            tmp14.store(out_ptr4 + static_cast<int64_t>(x0));
            tmp31.store(in_out_ptr0 + static_cast<int64_t>(x0));
            tmp18.store(out_ptr5 + static_cast<int64_t>(x0));
            tmp22.store(out_ptr6 + static_cast<int64_t>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(256L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(16L))
            {
                auto tmp0 = out_ptr3[static_cast<int64_t>(x0)];
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<int64_t>(x1 + (64L*x0)), static_cast<int64_t>(16));
                auto tmp4 = out_ptr4[static_cast<int64_t>(x0)];
                auto tmp5 = out_ptr5[static_cast<int64_t>(x0)];
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<int64_t>(x1 + (64L*x0)), static_cast<int64_t>(16));
                auto tmp9 = out_ptr2[static_cast<int64_t>(x0)];
                auto tmp10 = out_ptr6[static_cast<int64_t>(x0)];
                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<int64_t>(x1 + (64L*x0)), static_cast<int64_t>(16));
                auto tmp14 = out_ptr1[static_cast<int64_t>(x0)];
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<int64_t>(x1 + (64L*x0)), static_cast<int64_t>(16));
                auto tmp25 = in_out_ptr0[static_cast<int64_t>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp0);
                auto tmp3 = tmp2 * tmp1;
                auto tmp7 = at::vec::Vectorized<float>(tmp5);
                auto tmp8 = tmp7 * tmp6;
                auto tmp12 = at::vec::Vectorized<float>(tmp10);
                auto tmp13 = tmp12 * tmp11;
                auto tmp16 = at::vec::Vectorized<float>(tmp14);
                auto tmp17 = tmp16 * tmp15;
                auto tmp18 = tmp13 + tmp17;
                auto tmp19 = at::vec::Vectorized<float>(tmp9);
                auto tmp20 = tmp19 * tmp18;
                auto tmp21 = tmp8 + tmp20;
                auto tmp22 = at::vec::Vectorized<float>(tmp4);
                auto tmp23 = tmp22 * tmp21;
                auto tmp24 = tmp3 + tmp23;
                auto tmp26 = at::vec::Vectorized<float>(tmp25);
                auto tmp27 = tmp24 / tmp26;
                tmp27.store(in_out_ptr1 + static_cast<int64_t>(x1 + (64L*x0)));
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
    buf2 = empty_strided_cpu((2, 4, 32, 1), (128, 32, 1, 256), torch.float32)
    buf48 = empty_strided_cpu((2, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
    buf4 = empty_strided_cpu((2, 2, 32, 64), (4096, 2048, 64, 1), torch.float32)
    cpp_fused_clone_div_exp_masked_fill_max_sub_1(buf1, arg1_1, buf2, buf48, buf4)
    del arg1_1
    # Topologically Sorted Source Nodes: [contiguous, tensor], Original ATen: [aten.clone, _c10d_functional.all_to_all_single]
    buf5 = torch.ops._c10d_functional.all_to_all_single.default(buf4, [0, 0, 2, 0], [2, 0, 0, 0], '0')
    assert_size_stride(buf5, (2, 2, 32, 64), (4096, 2048, 64, 1))
    # Topologically Sorted Source Nodes: [k_1], Original ATen: [_c10d_functional.wait_tensor]
    torch.ops._c10d_functional.wait_tensor.default(buf5)
    buf8 = buf0; del buf0  # reuse
    cpp_fused_clone_2(buf5, buf8)
    buf9 = reinterpret_tensor(buf4, (8, 32, 32), (1024, 32, 1), 0); del buf4  # reuse
    # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(arg0_1, (8, 32, 64), (8192, 64, 1), 0), reinterpret_tensor(buf8, (8, 64, 32), (2048, 1, 64), 0), out=buf9)
    buf10 = empty_strided_cpu((2, 4, 32, 1), (128, 32, 1, 256), torch.float32)
    buf44 = reinterpret_tensor(buf1, (2, 4, 32, 32), (4096, 1024, 32, 1), 0); del buf1  # reuse
    buf57 = empty_strided_cpu((2, 4, 32, 1), (128, 32, 1, 256), torch.float32)
    cpp_fused_div_exp_max_sub_sum_3(buf9, buf10, buf44, buf57)
    # Topologically Sorted Source Nodes: [tensor_2], Original ATen: [_c10d_functional.all_to_all_single]
    buf12 = torch.ops._c10d_functional.all_to_all_single.default(buf5, [0, 0, 2, 0], [2, 0, 0, 0], '0')
    assert_size_stride(buf12, (2, 2, 32, 64), (4096, 2048, 64, 1))
    # Topologically Sorted Source Nodes: [k_3], Original ATen: [_c10d_functional.wait_tensor]
    torch.ops._c10d_functional.wait_tensor.default(buf12)
    buf15 = buf8; del buf8  # reuse
    cpp_fused_clone_4(buf12, buf15)
    buf16 = reinterpret_tensor(buf5, (8, 32, 32), (1024, 32, 1), 0); del buf5  # reuse
    # Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(arg0_1, (8, 32, 64), (8192, 64, 1), 0), reinterpret_tensor(buf15, (8, 64, 32), (2048, 1, 64), 0), out=buf16)
    buf17 = empty_strided_cpu((2, 4, 32, 1), (128, 32, 1, 256), torch.float32)
    buf40 = reinterpret_tensor(buf9, (2, 4, 32, 32), (4096, 1024, 32, 1), 0); del buf9  # reuse
    buf56 = empty_strided_cpu((2, 4, 32, 1), (128, 32, 1, 256), torch.float32)
    cpp_fused_div_exp_max_sub_sum_5(buf16, buf17, buf40, buf56)
    # Topologically Sorted Source Nodes: [tensor_4], Original ATen: [_c10d_functional.all_to_all_single]
    buf19 = torch.ops._c10d_functional.all_to_all_single.default(buf12, [0, 0, 2, 0], [2, 0, 0, 0], '0')
    assert_size_stride(buf19, (2, 2, 32, 64), (4096, 2048, 64, 1))
    # Topologically Sorted Source Nodes: [k_5], Original ATen: [_c10d_functional.wait_tensor]
    torch.ops._c10d_functional.wait_tensor.default(buf19)
    buf22 = buf15; del buf15  # reuse
    cpp_fused_clone_6(buf19, buf22)
    buf23 = reinterpret_tensor(buf19, (8, 32, 32), (1024, 32, 1), 0); del buf19  # reuse
    # Topologically Sorted Source Nodes: [matmul_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(arg0_1, (8, 32, 64), (8192, 64, 1), 0), reinterpret_tensor(buf22, (8, 64, 32), (2048, 1, 64), 0), out=buf23)
    del arg0_1
    buf24 = empty_strided_cpu((2, 4, 32, 1), (128, 32, 1, 256), torch.float32)
    buf26 = reinterpret_tensor(buf12, (2, 4, 32, 32), (4096, 1024, 32, 1), 0); del buf12  # reuse
    buf55 = empty_strided_cpu((2, 4, 32, 1), (128, 32, 1, 256), torch.float32)
    buf27 = reinterpret_tensor(buf16, (2, 2, 32, 64), (4096, 2048, 64, 1), 0); del buf16  # reuse
    cpp_fused_clone_div_exp_max_sub_sum_7(buf23, arg2_1, buf24, buf26, buf55, buf27)
    del buf23
    # Topologically Sorted Source Nodes: [contiguous_1, tensor_1], Original ATen: [aten.clone, _c10d_functional.all_to_all_single]
    buf28 = torch.ops._c10d_functional.all_to_all_single.default(buf27, [0, 0, 2, 0], [2, 0, 0, 0], '0')
    assert_size_stride(buf28, (2, 2, 32, 64), (4096, 2048, 64, 1))
    # Topologically Sorted Source Nodes: [v_1], Original ATen: [_c10d_functional.wait_tensor]
    torch.ops._c10d_functional.wait_tensor.default(buf28)
    del buf27
    # Topologically Sorted Source Nodes: [tensor_3], Original ATen: [_c10d_functional.all_to_all_single]
    buf31 = torch.ops._c10d_functional.all_to_all_single.default(buf28, [0, 0, 2, 0], [2, 0, 0, 0], '0')
    assert_size_stride(buf31, (2, 2, 32, 64), (4096, 2048, 64, 1))
    # Topologically Sorted Source Nodes: [v_3], Original ATen: [_c10d_functional.wait_tensor]
    torch.ops._c10d_functional.wait_tensor.default(buf31)
    # Topologically Sorted Source Nodes: [tensor_5], Original ATen: [_c10d_functional.all_to_all_single]
    buf34 = torch.ops._c10d_functional.all_to_all_single.default(buf31, [0, 0, 2, 0], [2, 0, 0, 0], '0')
    assert_size_stride(buf34, (2, 2, 32, 64), (4096, 2048, 64, 1))
    # Topologically Sorted Source Nodes: [v_5], Original ATen: [_c10d_functional.wait_tensor]
    torch.ops._c10d_functional.wait_tensor.default(buf34)
    buf37 = buf22; del buf22  # reuse
    cpp_fused_clone_8(buf34, buf37)
    del buf34
    buf38 = empty_strided_cpu((8, 32, 64), (2048, 64, 1), torch.float32)
    # Topologically Sorted Source Nodes: [numerator_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf26, (8, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf37, (8, 32, 64), (2048, 64, 1), 0), out=buf38)
    del buf26
    buf41 = buf37; del buf37  # reuse
    cpp_fused_clone_9(buf31, buf41)
    del buf31
    buf42 = empty_strided_cpu((8, 32, 64), (2048, 64, 1), torch.float32)
    # Topologically Sorted Source Nodes: [numerator_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf40, (8, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf41, (8, 32, 64), (2048, 64, 1), 0), out=buf42)
    del buf40
    buf45 = buf41; del buf41  # reuse
    cpp_fused_clone_10(buf28, buf45)
    del buf28
    buf46 = empty_strided_cpu((8, 32, 64), (2048, 64, 1), torch.float32)
    # Topologically Sorted Source Nodes: [numerator_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf44, (8, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf45, (8, 32, 64), (2048, 64, 1), 0), out=buf46)
    del buf44
    buf49 = buf45; del buf45  # reuse
    cpp_fused_clone_11(arg2_1, buf49)
    del arg2_1
    buf50 = empty_strided_cpu((8, 32, 64), (2048, 64, 1), torch.float32)
    # Topologically Sorted Source Nodes: [numerator], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf48, (8, 32, 32), (1024, 32, 1), 0), reinterpret_tensor(buf49, (8, 32, 64), (2048, 64, 1), 0), out=buf50)
    del buf49
    buf58 = empty_strided_cpu((2, 4, 32, 1), (128, 32, 1, 256), torch.float32)
    buf51 = empty_strided_cpu((2, 4, 32, 1), (128, 32, 1, 256), torch.float32)
    buf52 = empty_strided_cpu((2, 4, 32, 1), (128, 32, 1, 256), torch.float32)
    buf39 = empty_strided_cpu((2, 4, 32, 1), (128, 32, 1, 256), torch.float32)
    buf53 = empty_strided_cpu((2, 4, 32, 1), (128, 32, 1, 256), torch.float32)
    buf59 = buf55; del buf55  # reuse
    buf43 = empty_strided_cpu((2, 4, 32, 1), (128, 32, 1, 256), torch.float32)
    buf47 = empty_strided_cpu((2, 4, 32, 1), (128, 32, 1, 256), torch.float32)
    buf54 = reinterpret_tensor(buf38, (2, 4, 32, 64), (8192, 2048, 64, 1), 0); del buf38  # reuse
    buf60 = buf54; del buf54  # reuse
    cpp_fused_add_div_exp_maximum_mul_sub_sum_12(buf59, buf60, buf48, buf2, buf10, buf17, buf24, buf56, buf57, buf42, buf46, buf50, buf58, buf51, buf52, buf39, buf53, buf43, buf47)
    return (buf60, )


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
