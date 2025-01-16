
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



isolate_fails_code_str = None



# torch version: 2.3.1+cu121
# torch cuda version: 12.1
# torch git version: d44533f9d073df13895333e70b66f81c513c1889


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA A100-SXM4-80GB : 4 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, primals_1, primals_2):
        permute = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        mm = torch.ops.aten.mm.default(primals_2, permute)
        relu = torch.ops.aten.relu.default(mm);  mm = None
        alias = torch.ops.aten.alias.default(relu)
        alias_1 = torch.ops.aten.alias.default(alias);  alias = None
        alias_2 = torch.ops.aten.alias.default(alias_1);  alias_1 = None
        alias_3 = torch.ops.aten.alias.default(alias_2);  alias_2 = None
        le = torch.ops.aten.le.Scalar(alias_3, 0);  alias_3 = None
        permute_3 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return [relu, primals_2, le, permute_3]
        
def load_args(reader):
    buf0 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf0, (128, 128), requires_grad=True, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1, (1, 128), requires_grad=True)  # primals_2
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
