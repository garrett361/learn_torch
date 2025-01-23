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
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, grid_combo_kernels, start_graph, end_graph
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


# kernel path: /tmp/torchinductor_goon/kr/ckrgqt5qp7c7z4p4b6ygvptj2ofhxdpw22billndnxkexkvldjv2.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_default_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_4,), kwargs = {})
triton_poi_fused_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=1, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_goon/xf/cxfka5legbewree7l2hrrhpq45ej4d2ea3bf7rk7ivbntgqlrpac.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:1, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%arg6_1, %full_default, %clone), kwargs = {})
triton_poi_fused_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=1, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_goon/tx/ctxuvmbrwnpkfo4q4czi6di65g2tgmdwzhp3jg537wdfl5khnzlm.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%arg5_1, 0), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:1, pin_memory: False})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default_1, %mm_1), kwargs = {})
triton_poi_fused_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=1, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1 = args
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
    with torch.cuda._DeviceGuard(1):
        torch.cuda.set_device(1)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf1 = torch.ops.fsdp.all_gather_copy_in.default([arg8_1], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1))
        del arg8_1
        buf2 = buf1[0]
        buf3 = buf1[1]
        del buf1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._c10d_functional.all_gather_into_tensor_out.default(buf2, 2, '0', out=buf3)
        del buf2
        buf0 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
        buf9 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream1 = get_raw_stream(1)
        triton_poi_fused_0.run(buf0, buf9, 256, grid=grid(256), stream=stream1)
        buf15 = empty_strided_cuda((1, 16), (16, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(arg6_1, arg0_1, buf15, 16, grid=grid(16), stream=stream1)
        del arg0_1
        del arg6_1
        buf18 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
        buf27 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_0.run(buf18, buf27, 256, grid=grid(256), stream=stream1)
        buf54 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._c10d_functional.wait_tensor.default(buf3)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops.fsdp.split_with_sizes_copy.default(reinterpret_tensor(buf3, (2, 128), (128, 1), 0), [128], 1, out=[reinterpret_tensor(buf9, (2, 128), (128, 1), 0)])
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf12 = torch.ops.aten.set_.source_Tensor(arg3_1, reinterpret_tensor(buf9, (16, 16), (16, 1), 0))
        assert_size_stride(buf12, (16, 16), (16, 1))
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf19 = torch.ops.fsdp.all_gather_copy_in.default([arg9_1], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1))
        del arg9_1
        buf20 = buf19[0]
        buf21 = buf19[1]
        del buf19
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._c10d_functional.all_gather_into_tensor_out.default(buf20, 2, '0', out=buf21)
        del buf20
        buf16 = empty_strided_cuda((1, 16), (16, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf15, arg3_1, out=buf16)
        del arg3_1
        buf33 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf33, arg5_1, 16, grid=grid(16), stream=stream1)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._c10d_functional.wait_tensor.default(buf21)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops.fsdp.split_with_sizes_copy.default(reinterpret_tensor(buf21, (2, 128), (128, 1), 0), [128], 1, out=[reinterpret_tensor(buf27, (2, 128), (128, 1), 0)])
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf30 = torch.ops.aten.set_.source_Tensor(arg2_1, reinterpret_tensor(buf27, (16, 16), (16, 1), 0))
        assert_size_stride(buf30, (16, 16), (16, 1))
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf37 = torch.ops.fsdp.all_gather_copy_in.default([arg10_1], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1))
        del arg10_1
        buf38 = buf37[0]
        buf39 = buf37[1]
        del buf37
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._c10d_functional.all_gather_into_tensor_out.default(buf38, 2, '0', out=buf39)
        del buf38
        buf34 = empty_strided_cuda((1, 16), (16, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf33, arg2_1, out=buf34)
        del arg2_1
        buf51 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(buf51, arg4_1, 16, grid=grid(16), stream=stream1)
        buf52 = reinterpret_tensor(buf21, (16, 16), (16, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (16, 1), (1, 0), 0), arg1_1, out=buf52)
        del arg1_1
        del buf51
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops.fsdp.chunk_cat.default([buf52], 0, 2, out=reinterpret_tensor(buf54, (2, 128), (128, 1), 0))
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf58 = torch.ops._c10d_functional.reduce_scatter_tensor.default(buf54, 'avg', 2, '0')
        assert_size_stride(buf58, (128, ), (1, ))
        buf62 = reinterpret_tensor(buf52, (256, ), (1, ), 0); del buf52  # reuse
        buf35 = reinterpret_tensor(buf3, (16, 16), (16, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf33, (16, 1), (1, 16), 0), arg4_1, out=buf35)
        del arg4_1
        del buf33
        buf17 = empty_strided_cuda((16, 16), (16, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf15, (16, 1), (1, 16), 0), arg5_1, out=buf17)
        del arg5_1
        del buf15
        buf70 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
        buf36 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
        buf45 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_0.run(buf36, buf45, 256, grid=grid(256), stream=stream1)
        inductor_ops.resize_storage_bytes_(buf0, 0)
        inductor_ops.resize_storage_bytes_(buf18, 0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._c10d_functional.wait_tensor.default(buf39)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops.fsdp.split_with_sizes_copy.default(reinterpret_tensor(buf39, (2, 128), (128, 1), 0), [128], 1, out=[reinterpret_tensor(buf45, (2, 128), (128, 1), 0)])
        del buf39
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf48 = torch.ops.aten.set_.source_Tensor(arg7_1, reinterpret_tensor(buf45, (16, 16), (16, 1), 0))
        assert_size_stride(buf48, (16, 16), (16, 1))
        del arg7_1
        inductor_ops.resize_storage_bytes_(buf36, 0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._c10d_functional.wait_tensor.default(buf58)
        del buf54
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops.fsdp.chunk_cat.default([buf35], 0, 2, out=reinterpret_tensor(buf62, (2, 128), (128, 1), 0))
        del buf35
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf66 = torch.ops._c10d_functional.reduce_scatter_tensor.default(buf62, 'avg', 2, '0')
        assert_size_stride(buf66, (128, ), (1, ))
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._c10d_functional.wait_tensor.default(buf66)
        del buf62
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops.fsdp.chunk_cat.default([buf17], 0, 2, out=reinterpret_tensor(buf70, (2, 128), (128, 1), 0))
        del buf17
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf74 = torch.ops._c10d_functional.reduce_scatter_tensor.default(buf70, 'avg', 2, '0')
        assert_size_stride(buf74, (128, ), (1, ))
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._c10d_functional.wait_tensor.default(buf74)
        del buf70
    return (reinterpret_tensor(buf74, (8, 16), (16, 1), 0), reinterpret_tensor(buf66, (8, 16), (16, 1), 0), reinterpret_tensor(buf58, (8, 16), (16, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((), (), device='cuda:1', dtype=torch.bfloat16)
    arg1_1 = rand_strided((1, 16), (16, 1), device='cuda:1', dtype=torch.bfloat16)
    arg2_1 = rand_strided((16, 16), (16, 1), device='cuda:1', dtype=torch.bfloat16)
    arg3_1 = rand_strided((16, 16), (16, 1), device='cuda:1', dtype=torch.bfloat16)
    arg4_1 = rand_strided((1, 16), (16, 1), device='cuda:1', dtype=torch.bfloat16)
    arg5_1 = rand_strided((1, 16), (16, 1), device='cuda:1', dtype=torch.bfloat16)
    arg6_1 = rand_strided((1, 16), (16, 1), device='cuda:1', dtype=torch.bool)
    arg7_1 = rand_strided((16, 16), (16, 1), device='cuda:1', dtype=torch.bfloat16)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:1', dtype=torch.bfloat16)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:1', dtype=torch.bfloat16)
    arg10_1 = rand_strided((128, ), (1, ), device='cuda:1', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
