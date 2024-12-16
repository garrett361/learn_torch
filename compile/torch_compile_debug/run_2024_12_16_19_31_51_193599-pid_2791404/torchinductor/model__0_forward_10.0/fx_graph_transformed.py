class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1, 256]"):
        # File: /gpfs/users/goon/github/garrett361/learn_torch/compile/basic_fsdp2.py:23 in torch_dynamo_resume_in_forward_at_23, code: outputs = self.lin0(inputs).relu()
        relu: "f32[1, 256]" = torch.ops.aten.relu.default(primals_1);  primals_1 = None
        le: "b8[1, 256]" = torch.ops.aten.le.Scalar(relu, 0)
        return [relu, le]
        