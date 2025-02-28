# AOT ID: ['1_inference']
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
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

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


# kernel path: /tmp/torchinductor_goon/ta/cta2ulj2qwenneyzj3dppkdxtrp4t7knzm4kj7c54mdalvqdjqgu.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   %full_default : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%arg6_1, %full_default, %clone), kwargs = {})
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (0)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = 0.0
    tmp4 = tl.where(tmp0, tmp3, tmp2)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_goon/c2/cc2vlgsux57vhu3jqlfmfzdh2b45l3v5jppnhzaivvr7gq2hfbnu.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%arg5_1, 0), kwargs = {})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default, %mm_1), kwargs = {})
triton_poi_fused_1 = async_compile.triton('triton_poi_fused_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_goon/ia/ciat7xvrvmgl3ergj45dh7i7b2evegzyuosblcqxvvwo5tmf3unl.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg11_1, %as_strided_7), kwargs = {})
#   %copy__3 : [num_users=1] = call_function[target=torch.ops.aten.copy_.default](args = (%arg11_1, %add), kwargs = {})
triton_poi_fused_2 = async_compile.triton('triton_poi_fused_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': ['in_ptr0', 'out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '0899F28CADFA4B66E92BC168509FE6F8077C614C8ECE688266CFB3DAFB945502', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, in_ptr1, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr1 + (x0), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1 = args
    args.clear()
    assert_size_stride(arg0_1, (), ())
    assert_size_stride(arg1_1, (1, 16), (16, 1))
    assert_size_stride(arg2_1, (16, 16), (16, 1))
    assert_size_stride(arg3_1, (16, 16), (16, 1))
    assert_size_stride(arg4_1, (1, 16), (16, 1))
    assert_size_stride(arg5_1, (1, 16), (16, 1))
    assert_size_stride(arg6_1, (1, 16), (16, 1))
    assert_size_stride(arg7_1, (16, 16), (16, 1))
    assert_size_stride(arg8_1, (128, ), (1, ))
    assert_size_stride(arg9_1, (128, ), (1, ))
    assert_size_stride(arg10_1, (128, ), (1, ))
    assert_size_stride(arg11_1, (8, 16), (16, 1))
    assert_size_stride(arg12_1, (8, 16), (16, 1))
    assert_size_stride(arg13_1, (8, 16), (16, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf0 = torch.ops.fsdp.all_gather_copy_in.default([arg8_1], [128], 128, 2, 0, torch.bfloat16, device(type='cuda', index=0))
        del arg8_1
        buf1 = buf0[0]
        buf2 = buf0[1]
        del buf0
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._c10d_functional.all_gather_into_tensor_out.default(buf1, 2, '0', out=buf2)
        del buf1
        buf8 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._c10d_functional.wait_tensor.default(buf2)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops.fsdp.split_with_sizes_copy.default(reinterpret_tensor(buf2, (2, 128), (128, 1), 0), [128], 1, out=[reinterpret_tensor(buf8, (2, 128), (128, 1), 0)])
        buf11 = empty_strided_cuda((1, 16), (16, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(arg6_1, arg0_1, buf11, 16, grid=grid(16), stream=stream0)
        del arg0_1
        del arg6_1
        buf12 = reinterpret_tensor(buf2, (16, 16), (16, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf11, (16, 1), (1, 16), 0), arg5_1, out=buf12)
        buf13 = empty_strided_cuda((1, 16), (16, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf11, reinterpret_tensor(buf8, (16, 16), (16, 1), 0), out=buf13)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf14 = torch.ops.fsdp.all_gather_copy_in.default([arg9_1], [128], 128, 2, 0, torch.bfloat16, device(type='cuda', index=0))
        del arg9_1
        buf15 = buf14[0]
        buf16 = buf14[1]
        del buf14
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._c10d_functional.all_gather_into_tensor_out.default(buf15, 2, '0', out=buf16)
        del buf15
        buf22 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._c10d_functional.wait_tensor.default(buf16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops.fsdp.split_with_sizes_copy.default(reinterpret_tensor(buf16, (2, 128), (128, 1), 0), [128], 1, out=[reinterpret_tensor(buf22, (2, 128), (128, 1), 0)])
        buf25 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(buf25, arg5_1, 16, grid=grid(16), stream=stream0)
        del arg5_1
        buf26 = reinterpret_tensor(buf16, (16, 16), (16, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf25, (16, 1), (1, 16), 0), arg4_1, out=buf26)
        buf27 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf25, reinterpret_tensor(buf22, (16, 16), (16, 1), 0), out=buf27)
        del buf25
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf28 = torch.ops.fsdp.all_gather_copy_in.default([arg10_1], [128], 128, 2, 0, torch.bfloat16, device(type='cuda', index=0))
        del arg10_1
        buf29 = buf28[0]
        buf30 = buf28[1]
        del buf28
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._c10d_functional.all_gather_into_tensor_out.default(buf29, 2, '0', out=buf30)
        del buf29
        buf36 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._c10d_functional.wait_tensor.default(buf30)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops.fsdp.split_with_sizes_copy.default(reinterpret_tensor(buf30, (2, 128), (128, 1), 0), [128], 1, out=[reinterpret_tensor(buf36, (2, 128), (128, 1), 0)])
        buf39 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(buf39, arg4_1, 16, grid=grid(16), stream=stream0)
        del arg4_1
        buf40 = reinterpret_tensor(buf36, (16, 16), (16, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (16, 1), (1, 0), 0), arg1_1, out=buf40)
        del arg1_1
        del buf39
        buf41 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops.fsdp.chunk_cat.default([buf40], 0, 2, out=reinterpret_tensor(buf41, (2, 128), (128, 1), 0))
        del buf40
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf45 = torch.ops._c10d_functional.reduce_scatter_tensor.default(buf41, 'avg', 2, '0')
        assert_size_stride(buf45, (128, ), (1, ))
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._c10d_functional.wait_tensor.default(buf45)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(arg11_1, buf45, arg11_1, 128, grid=grid(128), stream=stream0)
        del buf45
        buf48 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops.fsdp.chunk_cat.default([buf26], 0, 2, out=reinterpret_tensor(buf48, (2, 128), (128, 1), 0))
        del buf26
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf52 = torch.ops._c10d_functional.reduce_scatter_tensor.default(buf48, 'avg', 2, '0')
        assert_size_stride(buf52, (128, ), (1, ))
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._c10d_functional.wait_tensor.default(buf52)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(arg12_1, buf52, arg12_1, 128, grid=grid(128), stream=stream0)
        del buf52
        buf55 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops.fsdp.chunk_cat.default([buf12], 0, 2, out=reinterpret_tensor(buf55, (2, 128), (128, 1), 0))
        del buf12
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf59 = torch.ops._c10d_functional.reduce_scatter_tensor.default(buf55, 'avg', 2, '0')
        assert_size_stride(buf59, (128, ), (1, ))
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._c10d_functional.wait_tensor.default(buf59)
        del buf55
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(arg13_1, buf59, arg13_1, 128, grid=grid(128), stream=stream0)
        del buf59
    return (arg11_1, arg12_1, arg13_1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((), (), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((1, 16), (16, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((1, 16), (16, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((1, 16), (16, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((1, 16), (16, 1), device='cuda:0', dtype=torch.bool)
    arg7_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((8, 16), (16, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
