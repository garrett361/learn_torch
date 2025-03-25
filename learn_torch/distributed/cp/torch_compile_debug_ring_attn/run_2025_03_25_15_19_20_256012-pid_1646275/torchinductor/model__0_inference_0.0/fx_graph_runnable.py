
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


torch._functorch.config.debug_partitioner = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None



# torch version: 2.5.1+cu124
# torch cuda version: 12.4
# torch git version: a8d6afb511a69687bbb2b7e88a3cf67917e1697e


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA A100-SXM4-80GB : 4 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1):
        unsqueeze = torch.ops.aten.unsqueeze.default(arg1_1, 2)
        expand = torch.ops.aten.expand.default(unsqueeze, [-1, -1, 2, -1, -1]);  unsqueeze = None
        clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view = torch.ops.aten.view.default(clone, [2, 4, 32, 64]);  clone = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(arg2_1, 2)
        expand_1 = torch.ops.aten.expand.default(unsqueeze_1, [-1, -1, 2, -1, -1]);  unsqueeze_1 = None
        clone_1 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_1 = torch.ops.aten.view.default(clone_1, [2, 4, 32, 64]);  clone_1 = None
        permute = torch.ops.aten.permute.default(view, [0, 1, 3, 2]);  view = None
        expand_2 = torch.ops.aten.expand.default(arg0_1, [2, 4, 32, 64]);  arg0_1 = None
        view_2 = torch.ops.aten.view.default(expand_2, [8, 32, 64]);  expand_2 = None
        expand_3 = torch.ops.aten.expand.default(permute, [2, 4, 64, 32]);  permute = None
        view_3 = torch.ops.aten.view.default(expand_3, [8, 64, 32]);  expand_3 = None
        bmm = torch.ops.aten.bmm.default(view_2, view_3);  view_2 = view_3 = None
        view_4 = torch.ops.aten.view.default(bmm, [2, 4, 32, 32]);  bmm = None
        div = torch.ops.aten.div.Tensor(view_4, 8.0);  view_4 = None
        full_default = torch.ops.aten.full.default([32, 32], True, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        iota = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
        iota_1 = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        sub = torch.ops.aten.sub.Tensor(unsqueeze_2, unsqueeze_3);  unsqueeze_2 = unsqueeze_3 = None
        ge = torch.ops.aten.ge.Scalar(sub, 1);  sub = None
        logical_and = torch.ops.aten.logical_and.default(ge, full_default);  ge = full_default = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(logical_and, 0);  logical_and = None
        full_default_1 = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where = torch.ops.aten.where.self(unsqueeze_5, full_default_1, div);  unsqueeze_5 = full_default_1 = div = None
        max_1 = torch.ops.aten.max.dim(where, -1, True)
        getitem = max_1[0];  max_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(where, getitem);  where = getitem = None
        exp = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        expand_4 = torch.ops.aten.expand.default(exp, [2, 4, 32, 32])
        view_5 = torch.ops.aten.view.default(expand_4, [8, 32, 32]);  expand_4 = None
        expand_5 = torch.ops.aten.expand.default(view_1, [2, 4, 32, 64]);  view_1 = None
        view_6 = torch.ops.aten.view.default(expand_5, [8, 32, 64]);  expand_5 = None
        bmm_1 = torch.ops.aten.bmm.default(view_5, view_6);  view_5 = view_6 = None
        view_7 = torch.ops.aten.view.default(bmm_1, [2, 4, 32, 64]);  bmm_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True);  exp = None
        clone_2 = torch.ops.aten.clone.default(arg1_1, memory_format = torch.contiguous_format);  arg1_1 = None
        all_to_all_single = torch.ops._c10d_functional.all_to_all_single.default(clone_2, [0, 0, 0, 2], [0, 2, 0, 0], '0');  clone_2 = None
        wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single);  all_to_all_single = None
        clone_3 = torch.ops.aten.clone.default(arg2_1, memory_format = torch.contiguous_format);  arg2_1 = None
        all_to_all_single_1 = torch.ops._c10d_functional.all_to_all_single.default(clone_3, [0, 0, 0, 2], [0, 2, 0, 0], '0');  clone_3 = None
        wait_tensor_1 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_1);  all_to_all_single_1 = None
        all_to_all_single_2 = torch.ops._c10d_functional.all_to_all_single.default(wait_tensor, [0, 0, 0, 2], [0, 2, 0, 0], '0');  wait_tensor = None
        wait_tensor_2 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_2);  all_to_all_single_2 = None
        all_to_all_single_3 = torch.ops._c10d_functional.all_to_all_single.default(wait_tensor_1, [0, 0, 0, 2], [0, 2, 0, 0], '0');  wait_tensor_1 = None
        wait_tensor_3 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_3);  all_to_all_single_3 = None
        all_to_all_single_4 = torch.ops._c10d_functional.all_to_all_single.default(wait_tensor_2, [0, 0, 0, 2], [0, 2, 0, 0], '0');  wait_tensor_2 = None
        wait_tensor_4 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_4);  all_to_all_single_4 = wait_tensor_4 = None
        all_to_all_single_5 = torch.ops._c10d_functional.all_to_all_single.default(wait_tensor_3, [0, 0, 0, 2], [0, 2, 0, 0], '0');  wait_tensor_3 = None
        wait_tensor_5 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_5);  all_to_all_single_5 = wait_tensor_5 = None
        div_1 = torch.ops.aten.div.Tensor(view_7, sum_1);  view_7 = sum_1 = None
        return (div_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 262144)
    reader.tensor(buf0, (2, 4, 32, 64), (32768, 8192, 64, 1), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 262144)
    reader.tensor(buf1, (2, 2, 32, 64), (32768, 16384, 128, 1), is_leaf=True)  # arg1_1
    reader.tensor(buf1, (2, 2, 32, 64), (32768, 16384, 128, 1), storage_offset=64, is_leaf=True)  # arg2_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)