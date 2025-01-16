class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[256, 128]", primals_2: "f32[1, 128]"):
        # File: /gpfs/users/goon/github/garrett361/learn_torch/compile/basic.py:29 in forward, code: outputs = self.lin0(inputs).relu()
        permute: "f32[128, 256]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        mm: "f32[1, 256]" = torch.ops.aten.mm.default(primals_2, permute);  permute = None
        relu: "f32[1, 256]" = torch.ops.aten.relu.default(mm);  mm = None
        le: "b8[1, 256]" = torch.ops.aten.le.Scalar(relu, 0)
        return [relu, primals_2, le]
        