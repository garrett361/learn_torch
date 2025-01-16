class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[128, 128]", primals_2: "f32[1, 128]"):
        # File: /gpfs/users/goon/.pyenv/versions/3.9.20/envs/learn_torch/lib/python3.9/site-packages/torch/nn/modules/linear.py:116 in forward, code: return F.linear(input, self.weight, self.bias)
        permute: "f32[128, 128]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        mm: "f32[1, 128]" = torch.ops.aten.mm.default(primals_2, permute);  permute = None
        
        # File: /gpfs/users/goon/github/garrett361/learn_torch/compile/basic_fsdp2.py:23 in forward, code: outputs = self.lin0(inputs).relu()
        relu: "f32[1, 128]" = torch.ops.aten.relu.default(mm);  mm = None
        alias: "f32[1, 128]" = torch.ops.aten.alias.default(relu)
        alias_1: "f32[1, 128]" = torch.ops.aten.alias.default(alias);  alias = None
        alias_2: "f32[1, 128]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
        alias_3: "f32[1, 128]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
        le: "b8[1, 128]" = torch.ops.aten.le.Scalar(alias_3, 0);  alias_3 = None
        return [relu, primals_2, le]
        