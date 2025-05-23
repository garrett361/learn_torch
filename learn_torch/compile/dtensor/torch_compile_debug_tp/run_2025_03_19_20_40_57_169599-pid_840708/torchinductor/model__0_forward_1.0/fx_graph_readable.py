class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1, 32]", primals_2: "f32[16, 32]", primals_3: "f32[32, 16]"):
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/compile/dtensor/tp.py:18 in forward, code: return self.lin1(self.lin0(inputs).relu())
        permute: "f32[32, 16]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        mm: "f32[1, 16]" = torch.ops.aten.mm.default(primals_1, permute);  permute = None
        relu: "f32[1, 16]" = torch.ops.aten.relu.default(mm);  mm = None
        permute_1: "f32[16, 32]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        mm_1: "f32[1, 32]" = torch.ops.aten.mm.default(relu, permute_1)
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/learn_torch/lib/python3.12/site-packages/torch/distributed/tensor/parallel/style.py:251 in _prepare_output_fn, code: outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        all_reduce: "f32[1, 32]" = torch.ops._c10d_functional.all_reduce.default(mm_1, 'sum', '1');  mm_1 = None
        wait_tensor: "f32[1, 32]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce);  all_reduce = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/compile/dtensor/tp.py:18 in forward, code: return self.lin1(self.lin0(inputs).relu())
        permute_6: "f32[32, 16]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return (wait_tensor, primals_1, relu, permute_6)
        