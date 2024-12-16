class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1, 256]"):
        # File: /gpfs/users/goon/github/garrett361/learn_torch/compile/basic_fsdp2.py:23 in torch_dynamo_resume_in_forward_at_23, code: outputs = self.lin0(inputs).relu()
        relu: "f32[1, 256]" = torch.ops.aten.relu.default(primals_1);  primals_1 = None
        alias: "f32[1, 256]" = torch.ops.aten.alias.default(relu)
        alias_1: "f32[1, 256]" = torch.ops.aten.alias.default(alias);  alias = None
        alias_2: "f32[1, 256]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
        alias_3: "f32[1, 256]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
        le: "b8[1, 256]" = torch.ops.aten.le.Scalar(alias_3, 0);  alias_3 = None
        return [relu, le]
        