class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[256, 128]", primals_2: "f32[1, 128]"):
        # File: /gpfs/users/goon/github/garrett361/learn_torch/compile/basic.py:29 in forward, code: outputs = self.lin0(inputs).relu()
        permute: "f32[128, 256]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        mm: "f32[1, 256]" = torch.ops.aten.mm.default(primals_2, permute);  permute = None
        relu: "f32[1, 256]" = torch.ops.aten.relu.default(mm);  mm = None
        alias: "f32[1, 256]" = torch.ops.aten.alias.default(relu)
        alias_1: "f32[1, 256]" = torch.ops.aten.alias.default(alias);  alias = None
        alias_2: "f32[1, 256]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
        alias_3: "f32[1, 256]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
        le: "b8[1, 256]" = torch.ops.aten.le.Scalar(alias_3, 0);  alias_3 = None
        return [relu, primals_2, le]
        