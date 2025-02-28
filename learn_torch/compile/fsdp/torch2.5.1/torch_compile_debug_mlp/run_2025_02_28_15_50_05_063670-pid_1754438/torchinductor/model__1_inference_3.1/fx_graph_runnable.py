
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.skip_fsdp_hooks = False
torch._inductor.config.reorder_for_compute_comm_overlap = True
torch._inductor.config.reorder_for_compute_comm_overlap_passes = ['sink_waits', 'raise_comms', 'reorder_compute_for_overlap']
torch._functorch.config.cse = False
torch._functorch.config.recompute_views = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.debug_partitioner = True



isolate_fails_code_str = None



# torch version: 2.5.1+cu124
# torch cuda version: 12.4
# torch git version: a8d6afb511a69687bbb2b7e88a3cf67917e1697e


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA A100-SXM4-80GB : 8 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1):
        expand = torch.ops.aten.expand.default(arg0_1, [1, 16]);  arg0_1 = None
        clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        all_gather_copy_in = torch.ops.fsdp.all_gather_copy_in.default([arg8_1], [128], 128, 2, 0, torch.bfloat16, device(type='cuda', index=0));  arg8_1 = None
        getitem = all_gather_copy_in[0]
        getitem_1 = all_gather_copy_in[1];  all_gather_copy_in = getitem_1 = None
        all_gather_into_tensor = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem, 2, '0');  getitem = None
        wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
        empty = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1 = torch.ops.aten.view.default(empty, [2, -1])
        view_2 = torch.ops.aten.view.default(wait_tensor, [2, -1]);  wait_tensor = None
        auto_functionalized = torch.ops.higher_order.auto_functionalized(torch.ops.fsdp.split_with_sizes_copy.default, all_gather_output = view_2, all_gather_input_split_sizes = [128], dim = 1, out = [view_1]);  view_2 = view_1 = None
        getitem_3 = auto_functionalized[1];  auto_functionalized = None
        getitem_4 = getitem_3[0];  getitem_3 = None
        view_3 = torch.ops.aten.view.default(getitem_4, [256]);  getitem_4 = None
        as_strided_1 = torch.ops.aten.as_strided.default(view_3, [16, 16], [16, 1], 0);  view_3 = None
        set_ = torch.ops.fsdp.set_.default(arg3_1, as_strided_1);  as_strided_1 = set_ = None
        full_default = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(arg6_1, full_default, clone);  arg6_1 = full_default = clone = None
        permute = torch.ops.aten.permute.default(where, [1, 0])
        mm = torch.ops.aten.mm.default(permute, arg5_1);  permute = None
        permute_1 = torch.ops.aten.permute.default(mm, [1, 0]);  mm = None
        permute_2 = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        permute_3 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_1 = torch.ops.aten.mm.default(where, permute_3);  where = permute_3 = None
        permute_4 = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        all_gather_copy_in_1 = torch.ops.fsdp.all_gather_copy_in.default([arg9_1], [128], 128, 2, 0, torch.bfloat16, device(type='cuda', index=0));  arg9_1 = None
        getitem_5 = all_gather_copy_in_1[0]
        getitem_6 = all_gather_copy_in_1[1];  all_gather_copy_in_1 = getitem_6 = None
        all_gather_into_tensor_1 = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_5, 2, '0');  getitem_5 = None
        wait_tensor_1 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_1);  all_gather_into_tensor_1 = None
        empty_1 = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_6 = torch.ops.aten.view.default(empty_1, [2, -1])
        view_7 = torch.ops.aten.view.default(wait_tensor_1, [2, -1]);  wait_tensor_1 = None
        auto_functionalized_1 = torch.ops.higher_order.auto_functionalized(torch.ops.fsdp.split_with_sizes_copy.default, all_gather_output = view_7, all_gather_input_split_sizes = [128], dim = 1, out = [view_6]);  view_7 = view_6 = None
        getitem_8 = auto_functionalized_1[1];  auto_functionalized_1 = None
        getitem_9 = getitem_8[0];  getitem_8 = None
        view_8 = torch.ops.aten.view.default(getitem_9, [256]);  getitem_9 = None
        as_strided_3 = torch.ops.aten.as_strided.default(view_8, [16, 16], [16, 1], 0);  view_8 = None
        set__1 = torch.ops.fsdp.set_.default(arg2_1, as_strided_3);  as_strided_3 = set__1 = None
        le = torch.ops.aten.le.Scalar(arg5_1, 0);  arg5_1 = None
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(le, full_default_1, mm_1);  le = full_default_1 = mm_1 = None
        permute_5 = torch.ops.aten.permute.default(where_1, [1, 0])
        mm_2 = torch.ops.aten.mm.default(permute_5, arg4_1);  permute_5 = None
        permute_6 = torch.ops.aten.permute.default(mm_2, [1, 0]);  mm_2 = None
        permute_7 = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        permute_8 = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        mm_3 = torch.ops.aten.mm.default(where_1, permute_8);  where_1 = permute_8 = None
        permute_9 = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
        all_gather_copy_in_2 = torch.ops.fsdp.all_gather_copy_in.default([arg10_1], [128], 128, 2, 0, torch.bfloat16, device(type='cuda', index=0));  arg10_1 = None
        getitem_10 = all_gather_copy_in_2[0]
        getitem_11 = all_gather_copy_in_2[1];  all_gather_copy_in_2 = getitem_11 = None
        all_gather_into_tensor_2 = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_10, 2, '0');  getitem_10 = None
        wait_tensor_2 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_2);  all_gather_into_tensor_2 = None
        empty_2 = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_11 = torch.ops.aten.view.default(empty_2, [2, -1])
        view_12 = torch.ops.aten.view.default(wait_tensor_2, [2, -1]);  wait_tensor_2 = None
        auto_functionalized_2 = torch.ops.higher_order.auto_functionalized(torch.ops.fsdp.split_with_sizes_copy.default, all_gather_output = view_12, all_gather_input_split_sizes = [128], dim = 1, out = [view_11]);  view_12 = view_11 = None
        getitem_13 = auto_functionalized_2[1];  auto_functionalized_2 = None
        getitem_14 = getitem_13[0];  getitem_13 = None
        view_13 = torch.ops.aten.view.default(getitem_14, [256]);  getitem_14 = None
        as_strided_5 = torch.ops.aten.as_strided.default(view_13, [16, 16], [16, 1], 0);  view_13 = None
        set__2 = torch.ops.fsdp.set_.default(arg7_1, as_strided_5);  arg7_1 = as_strided_5 = set__2 = None
        le_1 = torch.ops.aten.le.Scalar(arg4_1, 0);  arg4_1 = None
        full_default_2 = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(le_1, full_default_2, mm_3);  le_1 = full_default_2 = mm_3 = None
        permute_10 = torch.ops.aten.permute.default(where_2, [1, 0]);  where_2 = None
        mm_4 = torch.ops.aten.mm.default(permute_10, arg1_1);  permute_10 = arg1_1 = None
        permute_11 = torch.ops.aten.permute.default(mm_4, [1, 0]);  mm_4 = None
        permute_12 = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(empty_2, 0);  empty_2 = resize_storage_bytes_ = None
        empty_3 = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_15 = torch.ops.aten.view.default(empty_3, [2, -1]);  empty_3 = None
        auto_functionalized_3 = torch.ops.higher_order.auto_functionalized(torch.ops.fsdp.chunk_cat.default, tensors = [permute_12], dim = 0, num_chunks = 2, out = view_15);  permute_12 = view_15 = None
        getitem_16 = auto_functionalized_3[1];  auto_functionalized_3 = None
        view_16 = torch.ops.aten.view.default(getitem_16, [256]);  getitem_16 = None
        empty_4 = torch.ops.aten.empty.memory_format([128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  empty_4 = None
        reduce_scatter_tensor = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_16, 'avg', 2, '0');  view_16 = None
        wait_tensor_3 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor);  reduce_scatter_tensor = None
        resize_storage_bytes__1 = torch.ops.inductor.resize_storage_bytes_.default(empty_1, 0);  empty_1 = resize_storage_bytes__1 = None
        empty_5 = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_19 = torch.ops.aten.view.default(empty_5, [2, -1]);  empty_5 = None
        auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops.fsdp.chunk_cat.default, tensors = [permute_9], dim = 0, num_chunks = 2, out = view_19);  permute_9 = view_19 = None
        getitem_18 = auto_functionalized_4[1];  auto_functionalized_4 = None
        view_20 = torch.ops.aten.view.default(getitem_18, [256]);  getitem_18 = None
        empty_6 = torch.ops.aten.empty.memory_format([128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  empty_6 = None
        reduce_scatter_tensor_1 = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_20, 'avg', 2, '0');  view_20 = None
        wait_tensor_4 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_1);  reduce_scatter_tensor_1 = None
        resize_storage_bytes__2 = torch.ops.inductor.resize_storage_bytes_.default(empty, 0);  empty = resize_storage_bytes__2 = None
        empty_7 = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_23 = torch.ops.aten.view.default(empty_7, [2, -1]);  empty_7 = None
        auto_functionalized_5 = torch.ops.higher_order.auto_functionalized(torch.ops.fsdp.chunk_cat.default, tensors = [permute_4], dim = 0, num_chunks = 2, out = view_23);  permute_4 = view_23 = None
        getitem_20 = auto_functionalized_5[1];  auto_functionalized_5 = None
        view_24 = torch.ops.aten.view.default(getitem_20, [256]);  getitem_20 = None
        empty_8 = torch.ops.aten.empty.memory_format([128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  empty_8 = None
        reduce_scatter_tensor_2 = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_24, 'avg', 2, '0');  view_24 = None
        wait_tensor_5 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_2);  reduce_scatter_tensor_2 = None
        as_strided_9 = torch.ops.aten.as_strided.default(wait_tensor_5, [8, 16], [16, 1], 0);  wait_tensor_5 = None
        as_strided_10 = torch.ops.aten.as_strided.default(wait_tensor_4, [8, 16], [16, 1], 0);  wait_tensor_4 = None
        as_strided_11 = torch.ops.aten.as_strided.default(wait_tensor_3, [8, 16], [16, 1], 0);  wait_tensor_3 = None
        return (as_strided_9, as_strided_10, as_strided_11)
        
def load_args(reader):
    buf0 = reader.storage(None, 2, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (), dtype=torch.bfloat16, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 32, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (1, 16), dtype=torch.bfloat16, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 512, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (16, 16), dtype=torch.bfloat16, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 512, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (16, 16), dtype=torch.bfloat16, is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 32, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (1, 16), dtype=torch.bfloat16, is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 32, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf5, (1, 16), dtype=torch.bfloat16, is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf6, (1, 16), dtype=torch.bool, is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 512, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf7, (16, 16), dtype=torch.bfloat16, is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf8, (128,), dtype=torch.bfloat16, is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf9, (128,), dtype=torch.bfloat16, is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf10, (128,), dtype=torch.bfloat16, is_leaf=True)  # arg10_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)