
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

    
    
    def forward(self, primals_2, le, tangents_1):
        full_default = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(le, full_default, tangents_1);  le = full_default = tangents_1 = None
        permute_1 = torch.ops.aten.permute.default(where, [1, 0]);  where = None
        mm_1 = torch.ops.aten.mm.default(permute_1, primals_2);  permute_1 = primals_2 = None
        permute_2 = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        permute_3 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        return [permute_3, None]
        
def load_args(reader):
    buf0 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf0, (1, 128), is_leaf=True)  # primals_2
    buf1 = reader.storage(None, 128, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf1, (1, 128), dtype=torch.bool, is_leaf=True)  # le
    buf2 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1, 128), is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
