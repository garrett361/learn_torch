# AOT ID: ['0_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
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


# kernel path: /tmp/torchinductor_goon/jp/cjpb5olyvsnxr23c5kowvg3a7x2znj7llgd3a6ekalcynptscvio.py
# Topologically Sorted Source Nodes: [outputs], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   outputs => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%mm,), kwargs = {})
triton_poi_fused_relu_0 = async_compile.triton('triton_poi_fused_relu_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'BD001C26B1655FF20D01D972BCF6591519D78D60067BF64F26958DF9B224F154', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_0(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_goon/er/cerflmaxpxuwy3ykxn23ita2cjr7g4mrqkt4cxh3xiewqzoezkdj.py
# Topologically Sorted Source Nodes: [outputs_2], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   outputs_2 => relu_2
# Graph fragment:
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%mm_2,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_2, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_1 = async_compile.triton('triton_poi_fused_relu_threshold_backward_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'BD001C26B1655FF20D01D972BCF6591519D78D60067BF64F26958DF9B224F154', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_threshold_backward_1(in_out_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = 0.0
    tmp4 = tmp2 <= tmp3
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8 = args
    args.clear()
    assert_size_stride(primals_1, (1, 16), (16, 1))
    assert_size_stride(primals_2, (128, ), (1, ))
    assert_size_stride(primals_3, (16, 16), (16, 1))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (16, 16), (16, 1))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (16, 16), (16, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [], Original ATen: [fsdp.all_gather_copy_in]
        buf0 = torch.ops.fsdp.all_gather_copy_in.default([primals_2], [128], 128, 2, 0, torch.bfloat16, device(type='cuda', index=0))
        del primals_2
        buf1 = buf0[0]
        assert_size_stride(buf1, (128, ), (1, ))
        buf2 = buf0[1]
        assert_size_stride(buf2, (256, ), (1, ))
        del buf0
        # Topologically Sorted Source Nodes: [all_gather_into_tensor_out], Original ATen: [_c10d_functional.all_gather_into_tensor_out]
        torch.ops._c10d_functional.all_gather_into_tensor_out.default(buf1, 2, '0', out=buf2)
        # Topologically Sorted Source Nodes: [res], Original ATen: [_c10d_functional.wait_tensor]
        torch.ops._c10d_functional.wait_tensor.default(buf2)
        # Topologically Sorted Source Nodes: [], Original ATen: [fsdp.all_gather_copy_in]
        buf13 = torch.ops.fsdp.all_gather_copy_in.default([primals_5], [128], 128, 2, 0, torch.bfloat16, device(type='cuda', index=0))
        del primals_5
        buf14 = buf13[0]
        assert_size_stride(buf14, (128, ), (1, ))
        buf15 = buf13[1]
        assert_size_stride(buf15, (256, ), (1, ))
        del buf13
        # Topologically Sorted Source Nodes: [all_gather_into_tensor_out], Original ATen: [_c10d_functional.all_gather_into_tensor_out]
        torch.ops._c10d_functional.all_gather_into_tensor_out.default(buf14, 2, '0', out=buf15)
        del buf1
        # Topologically Sorted Source Nodes: [res_1], Original ATen: [_c10d_functional.wait_tensor]
        torch.ops._c10d_functional.wait_tensor.default(buf15)
        # Topologically Sorted Source Nodes: [], Original ATen: [fsdp.all_gather_copy_in]
        buf26 = torch.ops.fsdp.all_gather_copy_in.default([primals_7], [128], 128, 2, 0, torch.bfloat16, device(type='cuda', index=0))
        del primals_7
        buf27 = buf26[0]
        assert_size_stride(buf27, (128, ), (1, ))
        buf28 = buf26[1]
        assert_size_stride(buf28, (256, ), (1, ))
        del buf26
        # Topologically Sorted Source Nodes: [all_gather_into_tensor_out], Original ATen: [_c10d_functional.all_gather_into_tensor_out]
        torch.ops._c10d_functional.all_gather_into_tensor_out.default(buf27, 2, '0', out=buf28)
        del buf14
        del buf27
        # Topologically Sorted Source Nodes: [res_2], Original ATen: [_c10d_functional.wait_tensor]
        torch.ops._c10d_functional.wait_tensor.default(buf28)
        buf8 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops.fsdp.split_with_sizes_copy.default(reinterpret_tensor(buf2, (2, 128), (128, 1), 0), [128], 1, out=[reinterpret_tensor(buf8, (2, 128), (128, 1), 0)])
        del buf2
        buf11 = empty_strided_cuda((1, 16), (16, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(primals_1, reinterpret_tensor(buf8, (16, 16), (1, 16), 0), out=buf11)
        buf12 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [outputs], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_0.run(buf12, 16, grid=grid(16), stream=stream0)
        buf21 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops.fsdp.split_with_sizes_copy.default(reinterpret_tensor(buf15, (2, 128), (128, 1), 0), [128], 1, out=[reinterpret_tensor(buf21, (2, 128), (128, 1), 0)])
        del buf15
        buf24 = empty_strided_cuda((1, 16), (16, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf12, reinterpret_tensor(buf21, (16, 16), (1, 16), 0), out=buf24)
        buf25 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [outputs_1], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_0.run(buf25, 16, grid=grid(16), stream=stream0)
        buf34 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops.fsdp.split_with_sizes_copy.default(reinterpret_tensor(buf28, (2, 128), (128, 1), 0), [128], 1, out=[reinterpret_tensor(buf34, (2, 128), (128, 1), 0)])
        del buf28
        buf37 = empty_strided_cuda((1, 16), (16, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf25, reinterpret_tensor(buf34, (16, 16), (1, 16), 0), out=buf37)
        del buf34
        buf38 = buf37; del buf37  # reuse
        buf39 = empty_strided_cuda((1, 16), (16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [outputs_2], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_1.run(buf38, buf39, 16, grid=grid(16), stream=stream0)
    return (buf38, primals_1, primals_6, primals_8, buf12, buf25, buf39, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 16), (16, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_3 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_6 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_8 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_5, primals_6, primals_7, primals_8])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
