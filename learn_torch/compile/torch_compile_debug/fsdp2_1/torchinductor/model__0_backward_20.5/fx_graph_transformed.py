class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "f32[1, 128]", le: "b8[1, 128]", tangents_1: "f32[1, 128]"):
        # File: /gpfs/users/goon/github/garrett361/learn_torch/compile/basic_fsdp2.py:23 in forward, code: outputs = self.lin0(inputs).relu()
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False)
        where: "f32[1, 128]" = torch.ops.aten.where.self(le, full_default, tangents_1);  le = full_default = tangents_1 = None
        
        # File: /gpfs/users/goon/.pyenv/versions/3.9.20/envs/learn_torch/lib/python3.9/site-packages/torch/nn/modules/linear.py:116 in forward, code: return F.linear(input, self.weight, self.bias)
        permute_1: "f32[128, 1]" = torch.ops.aten.permute.default(where, [1, 0]);  where = None
        mm_1: "f32[128, 128]" = torch.ops.aten.mm.default(permute_1, primals_2);  permute_1 = primals_2 = None
        permute_2: "f32[128, 128]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        permute_3: "f32[128, 128]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        return [permute_3, None]
        