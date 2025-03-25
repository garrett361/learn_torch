
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
        expand_2 = torch.ops.aten.expand.default(arg0_1, [2, 4, 32, 64])
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
        sub_1 = torch.ops.aten.sub.Tensor(where, getitem);  where = None
        exp = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        expand_4 = torch.ops.aten.expand.default(exp, [2, 4, 32, 32])
        view_5 = torch.ops.aten.view.default(expand_4, [8, 32, 32]);  expand_4 = None
        expand_5 = torch.ops.aten.expand.default(view_1, [2, 4, 32, 64]);  view_1 = None
        view_6 = torch.ops.aten.view.default(expand_5, [8, 32, 64]);  expand_5 = None
        bmm_1 = torch.ops.aten.bmm.default(view_5, view_6);  view_5 = view_6 = None
        view_7 = torch.ops.aten.view.default(bmm_1, [2, 4, 32, 64]);  bmm_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True);  exp = None
        clone_2 = torch.ops.aten.clone.default(arg1_1, memory_format = torch.contiguous_format);  arg1_1 = None
        all_to_all_single = torch.ops._c10d_functional.all_to_all_single.default(clone_2, [0, 2, 0, 0], [0, 0, 0, 2], '0');  clone_2 = None
        wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single);  all_to_all_single = None
        clone_3 = torch.ops.aten.clone.default(arg2_1, memory_format = torch.contiguous_format);  arg2_1 = None
        all_to_all_single_1 = torch.ops._c10d_functional.all_to_all_single.default(clone_3, [0, 2, 0, 0], [0, 0, 0, 2], '0');  clone_3 = None
        wait_tensor_1 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_1);  all_to_all_single_1 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(wait_tensor, 2)
        expand_6 = torch.ops.aten.expand.default(unsqueeze_6, [-1, -1, 2, -1, -1]);  unsqueeze_6 = None
        clone_4 = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_8 = torch.ops.aten.view.default(clone_4, [2, 4, 32, 64]);  clone_4 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(wait_tensor_1, 2)
        expand_7 = torch.ops.aten.expand.default(unsqueeze_7, [-1, -1, 2, -1, -1]);  unsqueeze_7 = None
        clone_5 = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_9 = torch.ops.aten.view.default(clone_5, [2, 4, 32, 64]);  clone_5 = None
        permute_1 = torch.ops.aten.permute.default(view_8, [0, 1, 3, 2]);  view_8 = None
        expand_8 = torch.ops.aten.expand.default(arg0_1, [2, 4, 32, 64])
        view_10 = torch.ops.aten.view.default(expand_8, [8, 32, 64]);  expand_8 = None
        expand_9 = torch.ops.aten.expand.default(permute_1, [2, 4, 64, 32]);  permute_1 = None
        view_11 = torch.ops.aten.view.default(expand_9, [8, 64, 32]);  expand_9 = None
        bmm_2 = torch.ops.aten.bmm.default(view_10, view_11);  view_10 = view_11 = None
        view_12 = torch.ops.aten.view.default(bmm_2, [2, 4, 32, 32]);  bmm_2 = None
        div_1 = torch.ops.aten.div.Tensor(view_12, 8.0);  view_12 = None
        max_2 = torch.ops.aten.max.dim(div_1, -1, True)
        getitem_2 = max_2[0];  max_2 = None
        sub_2 = torch.ops.aten.sub.Tensor(div_1, getitem_2);  div_1 = None
        exp_1 = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        expand_10 = torch.ops.aten.expand.default(exp_1, [2, 4, 32, 32])
        view_13 = torch.ops.aten.view.default(expand_10, [8, 32, 32]);  expand_10 = None
        expand_11 = torch.ops.aten.expand.default(view_9, [2, 4, 32, 64]);  view_9 = None
        view_14 = torch.ops.aten.view.default(expand_11, [8, 32, 64]);  expand_11 = None
        bmm_3 = torch.ops.aten.bmm.default(view_13, view_14);  view_13 = view_14 = None
        view_15 = torch.ops.aten.view.default(bmm_3, [2, 4, 32, 64]);  bmm_3 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True);  exp_1 = None
        maximum = torch.ops.aten.maximum.default(getitem, getitem_2)
        sub_3 = torch.ops.aten.sub.Tensor(getitem_2, maximum)
        exp_2 = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        mul = torch.ops.aten.mul.Tensor(exp_2, view_15);  exp_2 = view_15 = None
        sub_4 = torch.ops.aten.sub.Tensor(getitem, maximum)
        exp_3 = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        mul_1 = torch.ops.aten.mul.Tensor(exp_3, view_7);  exp_3 = view_7 = None
        add = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
        sub_5 = torch.ops.aten.sub.Tensor(getitem_2, maximum);  getitem_2 = None
        exp_4 = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        mul_2 = torch.ops.aten.mul.Tensor(exp_4, sum_2);  exp_4 = sum_2 = None
        sub_6 = torch.ops.aten.sub.Tensor(getitem, maximum);  getitem = None
        exp_5 = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        mul_3 = torch.ops.aten.mul.Tensor(exp_5, sum_1);  exp_5 = sum_1 = None
        add_1 = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        all_to_all_single_2 = torch.ops._c10d_functional.all_to_all_single.default(wait_tensor, [0, 2, 0, 0], [0, 0, 0, 2], '0');  wait_tensor = None
        wait_tensor_2 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_2);  all_to_all_single_2 = None
        all_to_all_single_3 = torch.ops._c10d_functional.all_to_all_single.default(wait_tensor_1, [0, 2, 0, 0], [0, 0, 0, 2], '0');  wait_tensor_1 = None
        wait_tensor_3 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_3);  all_to_all_single_3 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(wait_tensor_2, 2)
        expand_12 = torch.ops.aten.expand.default(unsqueeze_8, [-1, -1, 2, -1, -1]);  unsqueeze_8 = None
        clone_6 = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_16 = torch.ops.aten.view.default(clone_6, [2, 4, 32, 64]);  clone_6 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(wait_tensor_3, 2)
        expand_13 = torch.ops.aten.expand.default(unsqueeze_9, [-1, -1, 2, -1, -1]);  unsqueeze_9 = None
        clone_7 = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_17 = torch.ops.aten.view.default(clone_7, [2, 4, 32, 64]);  clone_7 = None
        permute_2 = torch.ops.aten.permute.default(view_16, [0, 1, 3, 2]);  view_16 = None
        expand_14 = torch.ops.aten.expand.default(arg0_1, [2, 4, 32, 64]);  arg0_1 = None
        view_18 = torch.ops.aten.view.default(expand_14, [8, 32, 64]);  expand_14 = None
        expand_15 = torch.ops.aten.expand.default(permute_2, [2, 4, 64, 32]);  permute_2 = None
        view_19 = torch.ops.aten.view.default(expand_15, [8, 64, 32]);  expand_15 = None
        bmm_4 = torch.ops.aten.bmm.default(view_18, view_19);  view_18 = view_19 = None
        view_20 = torch.ops.aten.view.default(bmm_4, [2, 4, 32, 32]);  bmm_4 = None
        div_2 = torch.ops.aten.div.Tensor(view_20, 8.0);  view_20 = None
        max_3 = torch.ops.aten.max.dim(div_2, -1, True)
        getitem_4 = max_3[0];  max_3 = None
        sub_7 = torch.ops.aten.sub.Tensor(div_2, getitem_4);  div_2 = None
        exp_6 = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        expand_16 = torch.ops.aten.expand.default(exp_6, [2, 4, 32, 32])
        view_21 = torch.ops.aten.view.default(expand_16, [8, 32, 32]);  expand_16 = None
        expand_17 = torch.ops.aten.expand.default(view_17, [2, 4, 32, 64]);  view_17 = None
        view_22 = torch.ops.aten.view.default(expand_17, [8, 32, 64]);  expand_17 = None
        bmm_5 = torch.ops.aten.bmm.default(view_21, view_22);  view_21 = view_22 = None
        view_23 = torch.ops.aten.view.default(bmm_5, [2, 4, 32, 64]);  bmm_5 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True);  exp_6 = None
        maximum_1 = torch.ops.aten.maximum.default(maximum, getitem_4)
        sub_8 = torch.ops.aten.sub.Tensor(getitem_4, maximum_1)
        exp_7 = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        mul_4 = torch.ops.aten.mul.Tensor(exp_7, view_23);  exp_7 = view_23 = None
        sub_9 = torch.ops.aten.sub.Tensor(maximum, maximum_1)
        exp_8 = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        mul_5 = torch.ops.aten.mul.Tensor(exp_8, add);  exp_8 = add = None
        add_2 = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        sub_10 = torch.ops.aten.sub.Tensor(getitem_4, maximum_1);  getitem_4 = None
        exp_9 = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        mul_6 = torch.ops.aten.mul.Tensor(exp_9, sum_3);  exp_9 = sum_3 = None
        sub_11 = torch.ops.aten.sub.Tensor(maximum, maximum_1);  maximum = maximum_1 = None
        exp_10 = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        mul_7 = torch.ops.aten.mul.Tensor(exp_10, add_1);  exp_10 = add_1 = None
        add_3 = torch.ops.aten.add.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
        all_to_all_single_4 = torch.ops._c10d_functional.all_to_all_single.default(wait_tensor_2, [0, 2, 0, 0], [0, 0, 0, 2], '0');  wait_tensor_2 = None
        wait_tensor_4 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_4);  all_to_all_single_4 = wait_tensor_4 = None
        all_to_all_single_5 = torch.ops._c10d_functional.all_to_all_single.default(wait_tensor_3, [0, 2, 0, 0], [0, 0, 0, 2], '0');  wait_tensor_3 = None
        wait_tensor_5 = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_5);  all_to_all_single_5 = wait_tensor_5 = None
        div_3 = torch.ops.aten.div.Tensor(add_2, add_3);  add_2 = add_3 = None
        return (div_3,)
        
def load_args(reader):
    buf0 = reader.storage(None, 262144)
    reader.tensor(buf0, (2, 4, 32, 64), (32768, 8192, 64, 1), storage_offset=4096, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 262144)
    reader.tensor(buf1, (2, 2, 32, 64), (32768, 16384, 128, 1), storage_offset=8192, is_leaf=True)  # arg1_1
    reader.tensor(buf1, (2, 2, 32, 64), (32768, 16384, 128, 1), storage_offset=8256, is_leaf=True)  # arg2_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)