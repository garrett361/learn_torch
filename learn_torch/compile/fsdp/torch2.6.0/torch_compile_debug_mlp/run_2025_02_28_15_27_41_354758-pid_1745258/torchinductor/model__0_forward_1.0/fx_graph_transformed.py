class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[1, 16]", primals_2: "bf16[128]", primals_3: "bf16[16, 16]", primals_4, primals_5: "bf16[128]", primals_6: "bf16[16, 16]", primals_7: "bf16[128]", primals_8: "bf16[16, 16]"):
        # No stacktrace found for following nodes
        all_gather_copy_in_default_2 = torch.ops.fsdp.all_gather_copy_in.default([primals_2], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1));  primals_2 = None
        getitem_16: "bf16[128]" = all_gather_copy_in_default_2[0]
        getitem_17: "bf16[256]" = all_gather_copy_in_default_2[1];  all_gather_copy_in_default_2 = None
        all_gather_into_tensor_out_default_2: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor_out.default(getitem_16, 2, '0', out = getitem_17);  getitem_16 = getitem_17 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.6.0/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:141 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_out_default_2);  all_gather_into_tensor_out_default_2 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.6.0/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py:449 in init_all_gather_outputs, code: torch.empty(torch.Size([numel * world_size]), dtype=dtype, device=device)
        empty: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.6.0/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:288 in foreach_all_gather_copy_out, code: torch.ops.fsdp.split_with_sizes_copy(
        view_2: "bf16[2, 128]" = torch.ops.aten.reshape.default(wait_tensor, [2, -1]);  wait_tensor = None
        
        # No stacktrace found for following nodes
        as_strided_default_2: "bf16[2, 128]" = torch.ops.aten.as_strided.default(empty, [2, 128], [128, 1], 0)
        split_with_sizes_copy_default_2 = torch.ops.fsdp.split_with_sizes_copy.default(view_2, [128], 1, out = [as_strided_default_2]);  view_2 = as_strided_default_2 = split_with_sizes_copy_default_2 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.6.0/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py:525 in init_unsharded_param, code: torch.ops.fsdp.copy_(self._unsharded_param, unsharded_param)
        as_strided_1: "bf16[16, 16]" = torch.ops.aten.as_strided.default(empty, [16, 16], [16, 1], 0);  empty = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/compile/fsdp/torch2.6.0/mlp_fwd_only.py:48 in forward, code: outputs = self.lin0(inputs).relu()
        permute: "bf16[16, 16]" = torch.ops.aten.permute.default(as_strided_1, [1, 0]);  as_strided_1 = None
        mm: "bf16[1, 16]" = torch.ops.aten.mm.default(primals_1, permute);  permute = None
        relu: "bf16[1, 16]" = torch.ops.aten.relu.default(mm);  mm = None
        
        # No stacktrace found for following nodes
        all_gather_copy_in_default_1 = torch.ops.fsdp.all_gather_copy_in.default([primals_5], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1));  primals_5 = None
        getitem_14: "bf16[128]" = all_gather_copy_in_default_1[0]
        getitem_15: "bf16[256]" = all_gather_copy_in_default_1[1];  all_gather_copy_in_default_1 = None
        all_gather_into_tensor_out_default_1: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor_out.default(getitem_14, 2, '0', out = getitem_15);  getitem_14 = getitem_15 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.6.0/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:141 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_1: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_out_default_1);  all_gather_into_tensor_out_default_1 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.6.0/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py:449 in init_all_gather_outputs, code: torch.empty(torch.Size([numel * world_size]), dtype=dtype, device=device)
        empty_1: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.6.0/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:288 in foreach_all_gather_copy_out, code: torch.ops.fsdp.split_with_sizes_copy(
        view_5: "bf16[2, 128]" = torch.ops.aten.reshape.default(wait_tensor_1, [2, -1]);  wait_tensor_1 = None
        
        # No stacktrace found for following nodes
        as_strided_default_1: "bf16[2, 128]" = torch.ops.aten.as_strided.default(empty_1, [2, 128], [128, 1], 0)
        split_with_sizes_copy_default_1 = torch.ops.fsdp.split_with_sizes_copy.default(view_5, [128], 1, out = [as_strided_default_1]);  view_5 = as_strided_default_1 = split_with_sizes_copy_default_1 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.6.0/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py:525 in init_unsharded_param, code: torch.ops.fsdp.copy_(self._unsharded_param, unsharded_param)
        as_strided_3: "bf16[16, 16]" = torch.ops.aten.as_strided.default(empty_1, [16, 16], [16, 1], 0);  empty_1 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/compile/fsdp/torch2.6.0/mlp_fwd_only.py:48 in forward, code: outputs = self.lin0(inputs).relu()
        permute_1: "bf16[16, 16]" = torch.ops.aten.permute.default(as_strided_3, [1, 0]);  as_strided_3 = None
        mm_1: "bf16[1, 16]" = torch.ops.aten.mm.default(relu, permute_1);  permute_1 = None
        relu_1: "bf16[1, 16]" = torch.ops.aten.relu.default(mm_1);  mm_1 = None
        
        # No stacktrace found for following nodes
        all_gather_copy_in_default = torch.ops.fsdp.all_gather_copy_in.default([primals_7], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1));  primals_7 = None
        getitem_12: "bf16[128]" = all_gather_copy_in_default[0]
        getitem_13: "bf16[256]" = all_gather_copy_in_default[1];  all_gather_copy_in_default = None
        all_gather_into_tensor_out_default: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor_out.default(getitem_12, 2, '0', out = getitem_13);  getitem_12 = getitem_13 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.6.0/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:141 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_2: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_out_default);  all_gather_into_tensor_out_default = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.6.0/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py:449 in init_all_gather_outputs, code: torch.empty(torch.Size([numel * world_size]), dtype=dtype, device=device)
        empty_2: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.6.0/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:288 in foreach_all_gather_copy_out, code: torch.ops.fsdp.split_with_sizes_copy(
        view_8: "bf16[2, 128]" = torch.ops.aten.reshape.default(wait_tensor_2, [2, -1]);  wait_tensor_2 = None
        
        # No stacktrace found for following nodes
        as_strided_default: "bf16[2, 128]" = torch.ops.aten.as_strided.default(empty_2, [2, 128], [128, 1], 0)
        split_with_sizes_copy_default = torch.ops.fsdp.split_with_sizes_copy.default(view_8, [128], 1, out = [as_strided_default]);  view_8 = as_strided_default = split_with_sizes_copy_default = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.6.0/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py:525 in init_unsharded_param, code: torch.ops.fsdp.copy_(self._unsharded_param, unsharded_param)
        as_strided_5: "bf16[16, 16]" = torch.ops.aten.as_strided.default(empty_2, [16, 16], [16, 1], 0);  empty_2 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/compile/fsdp/torch2.6.0/mlp_fwd_only.py:48 in forward, code: outputs = self.lin0(inputs).relu()
        permute_2: "bf16[16, 16]" = torch.ops.aten.permute.default(as_strided_5, [1, 0]);  as_strided_5 = None
        mm_2: "bf16[1, 16]" = torch.ops.aten.mm.default(relu_1, permute_2);  permute_2 = None
        relu_2: "bf16[1, 16]" = torch.ops.aten.relu.default(mm_2);  mm_2 = None
        le: "b8[1, 16]" = torch.ops.aten.le.Scalar(relu_2, 0)
        return (relu_2, primals_1, primals_6, primals_8, relu, relu_1, le)
        