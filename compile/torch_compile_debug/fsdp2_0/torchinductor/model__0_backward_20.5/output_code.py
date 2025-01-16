
from ctypes import c_void_p, c_long
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
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_goon/ef/cef7p6a4idoi2btcpbv362nailiyswvvi7dt7ylytoynfxfmsd2a.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[128], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': '493e1fe457c358ce90acff64111ef5f4d47f6118d8ea0b6619574e4de66e314f'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


async_compile.wait(globals())
del async_compile

def call(args):
    primals_2, le, tangents_1 = args
    args.clear()
    assert_size_stride(primals_2, (1, 128), (128, 1))
    assert_size_stride(le, (1, 128), (128, 1))
    assert_size_stride(tangents_1, (1, 128), (128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_threshold_backward_0.run(le, tangents_1, buf0, 128, grid=grid(128), stream=stream0)
        del le
        del tangents_1
        buf1 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (128, 1), (1, 0), 0), primals_2, out=buf1)
        del buf0
        del primals_2
    return (reinterpret_tensor(buf1, (128, 128), (128, 1), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.bool)
    tangents_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_2, le, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
