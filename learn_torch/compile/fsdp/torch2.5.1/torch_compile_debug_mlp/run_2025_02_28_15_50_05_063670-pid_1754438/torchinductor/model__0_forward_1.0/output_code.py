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


# kernel path: /tmp/torchinductor_goon/3v/c3vilzpnsj6ll55j7s5chijvdc32t4bucofw63lwcf3jm7u26too.py
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
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
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


# kernel path: /tmp/torchinductor_goon/qb/cqbk4cocsibgylvv4iqvd3l35imn364gq4x7ror7btkzfrfzbbsq.py
# Topologically Sorted Source Nodes: [output], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   output => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%mm,), kwargs = {})
triton_poi_fused_relu_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_goon/te/ctecp5vdyj2gw5scwgrgesk2qrgrmdkiu26vyl3oulpuzjgftmmi.py
# Topologically Sorted Source Nodes: [output_2], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   output_2 => relu_2
# Graph fragment:
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%mm_2,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_2, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*i1', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_2', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf0 = torch.ops.fsdp.all_gather_copy_in.default([primals_2], [128], 128, 2, 0, torch.bfloat16, device(type='cuda', index=0))
        del primals_2
        buf1 = buf0[0]
        buf2 = buf0[1]
        del buf0
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._c10d_functional.all_gather_into_tensor_out.default(buf1, 2, '0', out=buf2)
        del buf1
        buf8 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
        buf9 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(buf8, buf9, 256, grid=grid(256), stream=stream0)
        buf26 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
        buf27 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_0.run(buf26, buf27, 256, grid=grid(256), stream=stream0)
        buf44 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
        buf45 = empty_strided_cuda((256, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_0.run(buf44, buf45, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [res], Original ATen: [_c10d_functional.wait_tensor]
        torch.ops._c10d_functional.wait_tensor.default(buf2)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops.fsdp.split_with_sizes_copy.default(reinterpret_tensor(buf2, (2, 128), (128, 1), 0), [128], 1, out=[reinterpret_tensor(buf9, (2, 128), (128, 1), 0)])
        del buf2
        # Topologically Sorted Source Nodes: [set__default], Original ATen: [fsdp.set_]
        buf12 = torch.ops.aten.set_.source_Tensor(primals_3, reinterpret_tensor(buf9, (16, 16), (16, 1), 0))
        assert_size_stride(buf12, (16, 16), (16, 1))
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf18 = torch.ops.fsdp.all_gather_copy_in.default([primals_5], [128], 128, 2, 0, torch.bfloat16, device(type='cuda', index=0))
        del primals_5
        buf19 = buf18[0]
        buf20 = buf18[1]
        del buf18
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._c10d_functional.all_gather_into_tensor_out.default(buf19, 2, '0', out=buf20)
        del buf19
        buf15 = empty_strided_cuda((1, 16), (16, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(primals_1, reinterpret_tensor(primals_3, (16, 16), (1, 16), 0), out=buf15)
        del primals_3
        buf16 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [output], Original ATen: [aten.relu]
        triton_poi_fused_relu_1.run(buf16, 16, grid=grid(16), stream=stream0)
        inductor_ops.resize_storage_bytes_(buf8, 0)
        # Topologically Sorted Source Nodes: [res_1], Original ATen: [_c10d_functional.wait_tensor]
        torch.ops._c10d_functional.wait_tensor.default(buf20)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops.fsdp.split_with_sizes_copy.default(reinterpret_tensor(buf20, (2, 128), (128, 1), 0), [128], 1, out=[reinterpret_tensor(buf27, (2, 128), (128, 1), 0)])
        del buf20
        # Topologically Sorted Source Nodes: [set__default_1], Original ATen: [fsdp.set_]
        buf30 = torch.ops.aten.set_.source_Tensor(primals_6, reinterpret_tensor(buf27, (16, 16), (16, 1), 0))
        assert_size_stride(buf30, (16, 16), (16, 1))
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf36 = torch.ops.fsdp.all_gather_copy_in.default([primals_7], [128], 128, 2, 0, torch.bfloat16, device(type='cuda', index=0))
        del primals_7
        buf37 = buf36[0]
        buf38 = buf36[1]
        del buf36
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._c10d_functional.all_gather_into_tensor_out.default(buf37, 2, '0', out=buf38)
        del buf37
        buf33 = empty_strided_cuda((1, 16), (16, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf16, reinterpret_tensor(primals_6, (16, 16), (1, 16), 0), out=buf33)
        buf34 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [output_1], Original ATen: [aten.relu]
        triton_poi_fused_relu_1.run(buf34, 16, grid=grid(16), stream=stream0)
        inductor_ops.resize_storage_bytes_(buf26, 0)
        # Topologically Sorted Source Nodes: [res_2], Original ATen: [_c10d_functional.wait_tensor]
        torch.ops._c10d_functional.wait_tensor.default(buf38)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops.fsdp.split_with_sizes_copy.default(reinterpret_tensor(buf38, (2, 128), (128, 1), 0), [128], 1, out=[reinterpret_tensor(buf45, (2, 128), (128, 1), 0)])
        del buf38
        # Topologically Sorted Source Nodes: [set__default_2], Original ATen: [fsdp.set_]
        buf48 = torch.ops.aten.set_.source_Tensor(primals_8, reinterpret_tensor(buf45, (16, 16), (16, 1), 0))
        assert_size_stride(buf48, (16, 16), (16, 1))
        buf51 = empty_strided_cuda((1, 16), (16, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf34, reinterpret_tensor(primals_8, (16, 16), (1, 16), 0), out=buf51)
        buf52 = buf51; del buf51  # reuse
        buf54 = empty_strided_cuda((1, 16), (16, 1), torch.bool)
        # Topologically Sorted Source Nodes: [output_2], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_2.run(buf52, buf54, 16, grid=grid(16), stream=stream0)
        inductor_ops.resize_storage_bytes_(buf44, 0)
    return (buf52, primals_1, primals_6, primals_8, buf16, buf34, buf54, )


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
