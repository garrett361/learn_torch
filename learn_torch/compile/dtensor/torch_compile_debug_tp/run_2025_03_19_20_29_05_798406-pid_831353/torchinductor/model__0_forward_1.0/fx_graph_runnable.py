
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
torch._dynamo.config._ddp_optimization_mode = ['ddp_optimizer', 'python_reducer', 'python_reducer_without_compiled_forward', 'no_optimization']
torch._dynamo.config._save_config_ignore = {'repro_level', 'repro_after', 'skipfiles_inline_module_allowlist', 'constant_functions'}
torch._dynamo.config.reorderable_logging_functions = set()
torch._inductor.config.pre_grad_fusion_options = {}
torch._inductor.config.post_grad_fusion_options = {}
torch._inductor.config.fx_passes_numeric_check = {'pre_grad': False, 'precision': 0.0001, 'num_iterations': 1, 'requires_optimizer': True}
torch._inductor.config.reorder_for_compute_comm_overlap_passes = ['reorder_compute_for_overlap', 'sink_waits', 'raise_comms']
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
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.6.0+cu124
# torch cuda version: 12.4
# torch git version: 2236df1770800ffea5697b11b0bb0d910b2e59e1


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA A100-SXM4-80GB : 4 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3):
        device_put = torch.ops.prims.device_put.default(primals_1, device(type='cuda', index=2));  primals_1 = None
        permute = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        mm = torch.ops.aten.mm.default(device_put, permute);  permute = None
        relu = torch.ops.aten.relu.default(mm);  mm = None
        permute_1 = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        mm_1 = torch.ops.aten.mm.default(relu, permute_1)
        all_reduce = torch.ops._c10d_functional.all_reduce.default(mm_1, 'sum', '0');  mm_1 = None
        wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_reduce);  all_reduce = None
        permute_6 = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return (wait_tensor, device_put, relu, permute_6)
        
def load_args(reader):
    buf0 = reader.storage(None, 128)
    reader.tensor(buf0, (1, 32), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 1024, device=device(type='cuda', index=2))
    reader.tensor(buf1, (8, 32), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 1024, device=device(type='cuda', index=2))
    reader.tensor(buf2, (32, 8), is_leaf=True)  # primals_3
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)