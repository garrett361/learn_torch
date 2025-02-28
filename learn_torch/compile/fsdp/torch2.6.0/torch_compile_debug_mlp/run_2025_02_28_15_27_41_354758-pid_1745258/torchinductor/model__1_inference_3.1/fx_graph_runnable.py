
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
torch._dynamo.config.traceable_tensor_subclasses = set()
torch._dynamo.config.allowed_functions_module_string_ignorelist = {'torch._prims', 'torch.distributions', 'torch._decomp', 'torch.testing', 'torch._refs'}
torch._dynamo.config._ddp_optimization_mode = ['ddp_optimizer', 'python_reducer', 'python_reducer_without_compiled_forward', 'no_optimization']
torch._dynamo.config.skip_fsdp_hooks = False
torch._dynamo.config._save_config_ignore = {'skipfiles_inline_module_allowlist', 'constant_functions', 'repro_after', 'repro_level'}
torch._dynamo.config.reorderable_logging_functions = set()
torch._dynamo.config.ignore_logger_methods = set()
torch._dynamo.config._autograd_backward_strict_mode_banned_ops = ['stride', 'requires_grad', 'storage_offset', 'layout', 'data', 'is_coalesced', 'is_complex', 'is_conj', 'is_contiguous', 'is_cpu', 'is_cuda', 'is_distributed', 'is_floating_point', 'is_inference', 'is_ipu', 'is_leaf', 'is_maia', 'is_meta', 'is_mkldnn', 'is_mps', 'is_mtia', 'is_neg', 'is_nested', 'is_nonzero', 'is_pinned', 'is_quantized', 'is_same_size', 'is_set_to', 'is_shared', 'is_signed', 'is_sparse', 'is_sparse_csr', 'is_vulkan', 'is_xla', 'is_xpu']
torch._dynamo.config.compiled_autograd_kwargs_override = {}
torch._inductor.config.pre_grad_fusion_options = {}
torch._inductor.config.post_grad_fusion_options = {}
torch._inductor.config.fx_passes_numeric_check = {'pre_grad': False, 'precision': 0.0001, 'num_iterations': 1, 'requires_optimizer': True}
torch._inductor.config.reorder_for_compute_comm_overlap_passes = ['reorder_compute_for_overlap', 'sink_waits', 'raise_comms']
torch._inductor.config._fuse_ddp_communication_passes = ['fuse_ddp_with_concat_op', 'schedule_comm_wait']
torch._inductor.config.aot_inductor.metadata = {}
torch._inductor.config.aot_inductor.presets = {}
torch._inductor.config.rocm.arch = []
torch._inductor.config.rocm.ck_supported_arch = ['gfx90a', 'gfx940', 'gfx941', 'gfx942']
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config._save_config_ignore = ['trace.upload_tar', 'joint_custom_pre_pass', 'joint_custom_post_pass', 'pre_grad_custom_pass']
torch._inductor.config._cache_config_ignore_prefix = ['trace', 'cuda.cutlass_dir', 'worker_start_method', 'compile_threads', 'post_grad_custom_post_pass', 'post_grad_custom_pre_pass', 'always_complex_memory_overlap_TESTING_ONLY']
torch._inductor.config.external_matmul = []
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.recompute_views = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.6.0+cu124
# torch cuda version: 12.4
# torch git version: 2236df1770800ffea5697b11b0bb0d910b2e59e1


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA A100-SXM4-80GB : 8 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1):
        expand = torch.ops.aten.expand.default(arg0_1, [1, 16]);  arg0_1 = None
        clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        all_gather_copy_in = torch.ops.fsdp.all_gather_copy_in.default([arg8_1], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1));  arg8_1 = None
        getitem = all_gather_copy_in[0]
        getitem_1 = all_gather_copy_in[1];  all_gather_copy_in = getitem_1 = None
        all_gather_into_tensor = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem, 2, '0');  getitem = None
        wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
        empty = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        view_2 = torch.ops.aten.view.default(wait_tensor, [2, -1]);  wait_tensor = None
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.fsdp.split_with_sizes_copy.default, all_gather_output = view_2, all_gather_input_split_sizes = [128], dim = 1, _out_length = 1, _out_0_base_index = 0, _out_0_size = (2, 128), _out_0_stride = (128, 1), _out_0_storage_offset = 0, _all_bases = [empty]);  view_2 = empty = None
        getitem_3 = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(arg3_1, 512);  resize_storage_bytes_ = None
        as_strided_1 = torch.ops.aten.as_strided.default(getitem_3, [16, 16], [16, 1], 0);  getitem_3 = None
        copy_ = torch.ops.fsdp.copy_.default(arg3_1, as_strided_1);  as_strided_1 = copy_ = None
        full_default = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False)
        where = torch.ops.aten.where.self(arg6_1, full_default, clone);  arg6_1 = clone = None
        permute = torch.ops.aten.permute.default(where, [1, 0])
        mm = torch.ops.aten.mm.default(permute, arg5_1);  permute = None
        mm_1 = torch.ops.aten.mm.default(where, arg3_1);  where = None
        all_gather_copy_in_1 = torch.ops.fsdp.all_gather_copy_in.default([arg9_1], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1));  arg9_1 = None
        getitem_4 = all_gather_copy_in_1[0]
        getitem_5 = all_gather_copy_in_1[1];  all_gather_copy_in_1 = getitem_5 = None
        all_gather_into_tensor_1 = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_4, 2, '0');  getitem_4 = None
        wait_tensor_1 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_1);  all_gather_into_tensor_1 = None
        empty_1 = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        view_5 = torch.ops.aten.view.default(wait_tensor_1, [2, -1]);  wait_tensor_1 = None
        auto_functionalized_v2_1 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.fsdp.split_with_sizes_copy.default, all_gather_output = view_5, all_gather_input_split_sizes = [128], dim = 1, _out_length = 1, _out_0_base_index = 0, _out_0_size = (2, 128), _out_0_stride = (128, 1), _out_0_storage_offset = 0, _all_bases = [empty_1]);  view_5 = empty_1 = None
        getitem_7 = auto_functionalized_v2_1[1];  auto_functionalized_v2_1 = None
        resize_storage_bytes__1 = torch.ops.inductor.resize_storage_bytes_.default(arg2_1, 512);  resize_storage_bytes__1 = None
        as_strided_3 = torch.ops.aten.as_strided.default(getitem_7, [16, 16], [16, 1], 0);  getitem_7 = None
        copy__1 = torch.ops.fsdp.copy_.default(arg2_1, as_strided_3);  as_strided_3 = copy__1 = None
        le = torch.ops.aten.le.Scalar(arg5_1, 0);  arg5_1 = None
        where_1 = torch.ops.aten.where.self(le, full_default, mm_1);  le = mm_1 = None
        permute_3 = torch.ops.aten.permute.default(where_1, [1, 0])
        mm_2 = torch.ops.aten.mm.default(permute_3, arg4_1);  permute_3 = None
        mm_3 = torch.ops.aten.mm.default(where_1, arg2_1);  where_1 = None
        all_gather_copy_in_2 = torch.ops.fsdp.all_gather_copy_in.default([arg10_1], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1));  arg10_1 = None
        getitem_8 = all_gather_copy_in_2[0]
        getitem_9 = all_gather_copy_in_2[1];  all_gather_copy_in_2 = getitem_9 = None
        all_gather_into_tensor_2 = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_8, 2, '0');  getitem_8 = None
        wait_tensor_2 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_2);  all_gather_into_tensor_2 = None
        empty_2 = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        view_8 = torch.ops.aten.view.default(wait_tensor_2, [2, -1]);  wait_tensor_2 = None
        auto_functionalized_v2_2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.fsdp.split_with_sizes_copy.default, all_gather_output = view_8, all_gather_input_split_sizes = [128], dim = 1, _out_length = 1, _out_0_base_index = 0, _out_0_size = (2, 128), _out_0_stride = (128, 1), _out_0_storage_offset = 0, _all_bases = [empty_2]);  view_8 = empty_2 = None
        getitem_11 = auto_functionalized_v2_2[1];  auto_functionalized_v2_2 = None
        resize_storage_bytes__2 = torch.ops.inductor.resize_storage_bytes_.default(arg7_1, 512);  resize_storage_bytes__2 = None
        as_strided_5 = torch.ops.aten.as_strided.default(getitem_11, [16, 16], [16, 1], 0);  getitem_11 = None
        copy__2 = torch.ops.fsdp.copy_.default(arg7_1, as_strided_5);  as_strided_5 = copy__2 = None
        le_1 = torch.ops.aten.le.Scalar(arg4_1, 0);  arg4_1 = None
        where_2 = torch.ops.aten.where.self(le_1, full_default, mm_3);  le_1 = full_default = mm_3 = None
        permute_6 = torch.ops.aten.permute.default(where_2, [1, 0]);  where_2 = None
        mm_4 = torch.ops.aten.mm.default(permute_6, arg1_1);  permute_6 = arg1_1 = None
        resize_storage_bytes__3 = torch.ops.inductor.resize_storage_bytes_.default(arg7_1, 0);  arg7_1 = resize_storage_bytes__3 = None
        empty_3 = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        auto_functionalized_v2_3 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.fsdp.chunk_cat.default, tensors = [mm_4], dim = 0, num_chunks = 2, _out_base_index = 0, _out_size = (2, 128), _out_stride = (128, 1), _out_storage_offset = 0, _all_bases = [empty_3]);  mm_4 = empty_3 = None
        getitem_13 = auto_functionalized_v2_3[1];  auto_functionalized_v2_3 = None
        empty_4 = torch.ops.aten.empty.memory_format([128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False);  empty_4 = None
        reduce_scatter_tensor = torch.ops._c10d_functional.reduce_scatter_tensor.default(getitem_13, 'avg', 2, '0');  getitem_13 = None
        wait_tensor_3 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor);  reduce_scatter_tensor = None
        as_strided_7 = torch.ops.aten.as_strided.default(wait_tensor_3, [8, 16], [16, 1], 0);  wait_tensor_3 = None
        add = torch.ops.aten.add.Tensor(arg11_1, as_strided_7);  as_strided_7 = None
        resize_storage_bytes__4 = torch.ops.inductor.resize_storage_bytes_.default(arg2_1, 0);  arg2_1 = resize_storage_bytes__4 = None
        empty_5 = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        auto_functionalized_v2_4 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.fsdp.chunk_cat.default, tensors = [mm_2], dim = 0, num_chunks = 2, _out_base_index = 0, _out_size = (2, 128), _out_stride = (128, 1), _out_storage_offset = 0, _all_bases = [empty_5]);  mm_2 = empty_5 = None
        getitem_15 = auto_functionalized_v2_4[1];  auto_functionalized_v2_4 = None
        empty_6 = torch.ops.aten.empty.memory_format([128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False);  empty_6 = None
        reduce_scatter_tensor_1 = torch.ops._c10d_functional.reduce_scatter_tensor.default(getitem_15, 'avg', 2, '0');  getitem_15 = None
        wait_tensor_4 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_1);  reduce_scatter_tensor_1 = None
        as_strided_9 = torch.ops.aten.as_strided.default(wait_tensor_4, [8, 16], [16, 1], 0);  wait_tensor_4 = None
        add_1 = torch.ops.aten.add.Tensor(arg12_1, as_strided_9);  as_strided_9 = None
        resize_storage_bytes__5 = torch.ops.inductor.resize_storage_bytes_.default(arg3_1, 0);  arg3_1 = resize_storage_bytes__5 = None
        empty_7 = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        auto_functionalized_v2_5 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.fsdp.chunk_cat.default, tensors = [mm], dim = 0, num_chunks = 2, _out_base_index = 0, _out_size = (2, 128), _out_stride = (128, 1), _out_storage_offset = 0, _all_bases = [empty_7]);  mm = empty_7 = None
        getitem_17 = auto_functionalized_v2_5[1];  auto_functionalized_v2_5 = None
        empty_8 = torch.ops.aten.empty.memory_format([128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False);  empty_8 = None
        reduce_scatter_tensor_2 = torch.ops._c10d_functional.reduce_scatter_tensor.default(getitem_17, 'avg', 2, '0');  getitem_17 = None
        wait_tensor_5 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_2);  reduce_scatter_tensor_2 = None
        as_strided_11 = torch.ops.aten.as_strided.default(wait_tensor_5, [8, 16], [16, 1], 0);  wait_tensor_5 = None
        add_2 = torch.ops.aten.add.Tensor(arg13_1, as_strided_11);  as_strided_11 = None
        copy__3 = torch.ops.aten.copy_.default(arg11_1, add);  arg11_1 = add = None
        copy__4 = torch.ops.aten.copy_.default(arg12_1, add_1);  arg12_1 = add_1 = None
        copy__5 = torch.ops.aten.copy_.default(arg13_1, add_2);  arg13_1 = add_2 = None
        return (copy__3, copy__4, copy__5)
        
def load_args(reader):
    buf0 = reader.storage(None, 2, device=device(type='cuda', index=1), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (), dtype=torch.bfloat16, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 32, device=device(type='cuda', index=1), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (1, 16), dtype=torch.bfloat16, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 0, device=device(type='cuda', index=1), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (16, 16), dtype=torch.bfloat16, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 0, device=device(type='cuda', index=1), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (16, 16), dtype=torch.bfloat16, is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 32, device=device(type='cuda', index=1), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (1, 16), dtype=torch.bfloat16, is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 32, device=device(type='cuda', index=1), dtype_hint=torch.bfloat16)
    reader.tensor(buf5, (1, 16), dtype=torch.bfloat16, is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 16, device=device(type='cuda', index=1), dtype_hint=torch.bool)
    reader.tensor(buf6, (1, 16), dtype=torch.bool, is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 0, device=device(type='cuda', index=1), dtype_hint=torch.bfloat16)
    reader.tensor(buf7, (16, 16), dtype=torch.bfloat16, is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 256, device=device(type='cuda', index=1), dtype_hint=torch.bfloat16)
    reader.tensor(buf8, (128,), dtype=torch.bfloat16, is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 256, device=device(type='cuda', index=1), dtype_hint=torch.bfloat16)
    reader.tensor(buf9, (128,), dtype=torch.bfloat16, is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 256, device=device(type='cuda', index=1), dtype_hint=torch.bfloat16)
    reader.tensor(buf10, (128,), dtype=torch.bfloat16, is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 256, device=device(type='cuda', index=1), dtype_hint=torch.bfloat16)
    reader.tensor(buf11, (8, 16), dtype=torch.bfloat16, is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 256, device=device(type='cuda', index=1), dtype_hint=torch.bfloat16)
    reader.tensor(buf12, (8, 16), dtype=torch.bfloat16, is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 256, device=device(type='cuda', index=1), dtype_hint=torch.bfloat16)
    reader.tensor(buf13, (8, 16), dtype=torch.bfloat16, is_leaf=True)  # arg13_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)