class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[1, 16]", primals_2: "bf16[128]", primals_3: "bf16[16, 16]", primals_4, primals_5: "bf16[128]", primals_6: "bf16[16, 16]", primals_7: "bf16[128]", primals_8: "bf16[16, 16]"):
        # No stacktrace found for following nodes
        all_gather_copy_in_default_2 = torch.ops.fsdp.all_gather_copy_in.default([primals_2], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1));  primals_2 = None
        getitem_19: "bf16[128]" = all_gather_copy_in_default_2[0]
        getitem_20: "bf16[256]" = all_gather_copy_in_default_2[1];  all_gather_copy_in_default_2 = None
        all_gather_into_tensor_out_default_2: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor_out.default(getitem_19, 2, '0', out = getitem_20);  getitem_19 = getitem_20 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:140 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_out_default_2);  all_gather_into_tensor_out_default_2 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_param.py:409 in init_all_gather_outputs, code: torch.empty(torch.Size([numel * world_size]), dtype=dtype, device=device)
        empty: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_collectives.py:274 in foreach_all_gather_copy_out, code: out = [t.view(world_size, -1) for t in gen]
        view_1: "bf16[2, 128]" = torch.ops.aten.reshape.default(empty, [2, -1])
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_collectives.py:275 in foreach_all_gather_copy_out, code: torch.ops.fsdp.split_with_sizes_copy(
        view_2: "bf16[2, 128]" = torch.ops.aten.reshape.default(wait_tensor, [2, -1]);  wait_tensor = None
        
        # No stacktrace found for following nodes
        as_strided_default_4: "bf16[256]" = torch.ops.aten.as_strided.default(view_1, [256], [1], 0);  view_1 = None
        clone_default_2: "bf16[256]" = torch.ops.aten.clone.default(as_strided_default_4);  as_strided_default_4 = None
        as_strided_default_5: "bf16[2, 128]" = torch.ops.aten.as_strided.default(clone_default_2, [2, 128], [128, 1], 0);  clone_default_2 = None
        split_with_sizes_copy_default_2 = torch.ops.fsdp.split_with_sizes_copy.default(view_2, [128], 1, out = [as_strided_default_5]);  view_2 = split_with_sizes_copy_default_2 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_collectives.py:275 in foreach_all_gather_copy_out, code: torch.ops.fsdp.split_with_sizes_copy(
        view_3: "bf16[256]" = torch.ops.aten.reshape.default(as_strided_default_5, [256]);  as_strided_default_5 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_param.py:478 in init_unsharded_param, code: torch.ops.fsdp.set_.default(self._unsharded_param, unsharded_param)
        as_strided_1: "bf16[16, 16]" = torch.ops.aten.as_strided.default(view_3, [16, 16], [16, 1], 0);  view_3 = None
        set_ = torch.ops.fsdp.set_.default(primals_3, as_strided_1);  as_strided_1 = set_ = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/compile/fsdp/mlp.py:97 in forward, code: outputs = self.lin0(inputs).relu()
        permute: "bf16[16, 16]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        mm: "bf16[1, 16]" = torch.ops.aten.mm.default(primals_1, permute);  permute = None
        relu: "bf16[1, 16]" = torch.ops.aten.relu.default(mm);  mm = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_param.py:733 in free_storage, code: storage.resize_(0)
        resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(empty, 0);  empty = resize_storage_bytes_ = None
        
        # No stacktrace found for following nodes
        all_gather_copy_in_default_1 = torch.ops.fsdp.all_gather_copy_in.default([primals_5], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1));  primals_5 = None
        getitem_17: "bf16[128]" = all_gather_copy_in_default_1[0]
        getitem_18: "bf16[256]" = all_gather_copy_in_default_1[1];  all_gather_copy_in_default_1 = None
        all_gather_into_tensor_out_default_1: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor_out.default(getitem_17, 2, '0', out = getitem_18);  getitem_17 = getitem_18 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:140 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_1: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_out_default_1);  all_gather_into_tensor_out_default_1 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_param.py:409 in init_all_gather_outputs, code: torch.empty(torch.Size([numel * world_size]), dtype=dtype, device=device)
        empty_1: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_collectives.py:274 in foreach_all_gather_copy_out, code: out = [t.view(world_size, -1) for t in gen]
        view_6: "bf16[2, 128]" = torch.ops.aten.reshape.default(empty_1, [2, -1])
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_collectives.py:275 in foreach_all_gather_copy_out, code: torch.ops.fsdp.split_with_sizes_copy(
        view_7: "bf16[2, 128]" = torch.ops.aten.reshape.default(wait_tensor_1, [2, -1]);  wait_tensor_1 = None
        
        # No stacktrace found for following nodes
        as_strided_default_2: "bf16[256]" = torch.ops.aten.as_strided.default(view_6, [256], [1], 0);  view_6 = None
        clone_default_1: "bf16[256]" = torch.ops.aten.clone.default(as_strided_default_2);  as_strided_default_2 = None
        as_strided_default_3: "bf16[2, 128]" = torch.ops.aten.as_strided.default(clone_default_1, [2, 128], [128, 1], 0);  clone_default_1 = None
        split_with_sizes_copy_default_1 = torch.ops.fsdp.split_with_sizes_copy.default(view_7, [128], 1, out = [as_strided_default_3]);  view_7 = split_with_sizes_copy_default_1 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_collectives.py:275 in foreach_all_gather_copy_out, code: torch.ops.fsdp.split_with_sizes_copy(
        view_8: "bf16[256]" = torch.ops.aten.reshape.default(as_strided_default_3, [256]);  as_strided_default_3 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_param.py:478 in init_unsharded_param, code: torch.ops.fsdp.set_.default(self._unsharded_param, unsharded_param)
        as_strided_3: "bf16[16, 16]" = torch.ops.aten.as_strided.default(view_8, [16, 16], [16, 1], 0);  view_8 = None
        set__1 = torch.ops.fsdp.set_.default(primals_6, as_strided_3);  as_strided_3 = set__1 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/compile/fsdp/mlp.py:97 in forward, code: outputs = self.lin0(inputs).relu()
        permute_1: "bf16[16, 16]" = torch.ops.aten.permute.default(primals_6, [1, 0])
        mm_1: "bf16[1, 16]" = torch.ops.aten.mm.default(relu, permute_1);  permute_1 = None
        relu_1: "bf16[1, 16]" = torch.ops.aten.relu.default(mm_1);  mm_1 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_param.py:733 in free_storage, code: storage.resize_(0)
        resize_storage_bytes__1 = torch.ops.inductor.resize_storage_bytes_.default(empty_1, 0);  empty_1 = resize_storage_bytes__1 = None
        
        # No stacktrace found for following nodes
        all_gather_copy_in_default = torch.ops.fsdp.all_gather_copy_in.default([primals_7], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1));  primals_7 = None
        getitem_15: "bf16[128]" = all_gather_copy_in_default[0]
        getitem_16: "bf16[256]" = all_gather_copy_in_default[1];  all_gather_copy_in_default = None
        all_gather_into_tensor_out_default: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor_out.default(getitem_15, 2, '0', out = getitem_16);  getitem_15 = getitem_16 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:140 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_2: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_out_default);  all_gather_into_tensor_out_default = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_param.py:409 in init_all_gather_outputs, code: torch.empty(torch.Size([numel * world_size]), dtype=dtype, device=device)
        empty_2: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_collectives.py:274 in foreach_all_gather_copy_out, code: out = [t.view(world_size, -1) for t in gen]
        view_11: "bf16[2, 128]" = torch.ops.aten.reshape.default(empty_2, [2, -1])
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_collectives.py:275 in foreach_all_gather_copy_out, code: torch.ops.fsdp.split_with_sizes_copy(
        view_12: "bf16[2, 128]" = torch.ops.aten.reshape.default(wait_tensor_2, [2, -1]);  wait_tensor_2 = None
        
        # No stacktrace found for following nodes
        as_strided_default: "bf16[256]" = torch.ops.aten.as_strided.default(view_11, [256], [1], 0);  view_11 = None
        clone_default: "bf16[256]" = torch.ops.aten.clone.default(as_strided_default);  as_strided_default = None
        as_strided_default_1: "bf16[2, 128]" = torch.ops.aten.as_strided.default(clone_default, [2, 128], [128, 1], 0);  clone_default = None
        split_with_sizes_copy_default = torch.ops.fsdp.split_with_sizes_copy.default(view_12, [128], 1, out = [as_strided_default_1]);  view_12 = split_with_sizes_copy_default = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_collectives.py:275 in foreach_all_gather_copy_out, code: torch.ops.fsdp.split_with_sizes_copy(
        view_13: "bf16[256]" = torch.ops.aten.reshape.default(as_strided_default_1, [256]);  as_strided_default_1 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_param.py:478 in init_unsharded_param, code: torch.ops.fsdp.set_.default(self._unsharded_param, unsharded_param)
        as_strided_5: "bf16[16, 16]" = torch.ops.aten.as_strided.default(view_13, [16, 16], [16, 1], 0);  view_13 = None
        set__2 = torch.ops.fsdp.set_.default(primals_8, as_strided_5);  as_strided_5 = set__2 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/compile/fsdp/mlp.py:97 in forward, code: outputs = self.lin0(inputs).relu()
        permute_2: "bf16[16, 16]" = torch.ops.aten.permute.default(primals_8, [1, 0])
        mm_2: "bf16[1, 16]" = torch.ops.aten.mm.default(relu_1, permute_2);  permute_2 = None
        relu_2: "bf16[1, 16]" = torch.ops.aten.relu.default(mm_2);  mm_2 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_param.py:733 in free_storage, code: storage.resize_(0)
        resize_storage_bytes__2 = torch.ops.inductor.resize_storage_bytes_.default(empty_2, 0);  empty_2 = resize_storage_bytes__2 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/compile/fsdp/mlp.py:97 in forward, code: outputs = self.lin0(inputs).relu()
        le: "b8[1, 16]" = torch.ops.aten.le.Scalar(relu_2, 0)
        return (relu_2, primals_1, primals_6, primals_8, relu, relu_1, le)
        