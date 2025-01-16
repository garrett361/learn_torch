class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[128, 128]", primals_2: "f32[1, 128]"):
        # File: /gpfs/users/goon/.pyenv/versions/3.9.20/envs/learn_torch/lib/python3.9/site-packages/torch/nn/modules/linear.py:116 in forward, code: return F.linear(input, self.weight, self.bias)
        permute: "f32[128, 128]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        mm: "f32[1, 128]" = torch.ops.aten.mm.default(primals_2, permute);  permute = None
        
        # File: /gpfs/users/goon/github/garrett361/learn_torch/compile/basic_fsdp2.py:23 in forward, code: outputs = self.lin0(inputs).relu()
        relu: "f32[1, 128]" = torch.ops.aten.relu.default(mm);  mm = None
        le: "b8[1, 128]" = torch.ops.aten.le.Scalar(relu, 0)
        return [relu, primals_2, le]
        