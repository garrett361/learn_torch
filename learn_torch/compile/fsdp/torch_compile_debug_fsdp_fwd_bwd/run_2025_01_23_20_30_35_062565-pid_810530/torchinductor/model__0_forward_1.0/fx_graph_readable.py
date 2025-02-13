class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[1, 16]", primals_2: "bf16[128]", primals_3: "bf16[16, 16]", primals_4, primals_5: "bf16[128]", primals_6: "bf16[16, 16]", primals_7: "bf16[128]", primals_8: "bf16[16, 16]"):
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_collectives.py:151 in foreach_all_gather, code: all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(
        all_gather_copy_in = torch.ops.fsdp.all_gather_copy_in.default([primals_2], [128], 128, 2, 0, torch.bfloat16, device(type='cuda', index=0));  primals_2 = None
        getitem: "bf16[128]" = all_gather_copy_in[0];  all_gather_copy_in = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:204 in all_gather_tensor, code: tensor = torch.ops._c10d_functional.all_gather_into_tensor(
        all_gather_into_tensor: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem, 2, '0');  getitem = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:140 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_param.py:409 in init_all_gather_outputs, code: torch.empty(torch.Size([numel * world_size]), dtype=dtype, device=device)
        empty: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_collectives.py:274 in foreach_all_gather_copy_out, code: out = [t.view(world_size, -1) for t in gen]
        view_1: "bf16[2, 128]" = torch.ops.aten.view.default(empty, [2, -1])
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_collectives.py:275 in foreach_all_gather_copy_out, code: torch.ops.fsdp.split_with_sizes_copy(
        view_2: "bf16[2, 128]" = torch.ops.aten.view.default(wait_tensor, [2, -1]);  wait_tensor = None
        auto_functionalized = torch.ops.higher_order.auto_functionalized(torch.ops.fsdp.split_with_sizes_copy.default, all_gather_output = view_2, all_gather_input_split_sizes = [128], dim = 1, out = [view_1]);  view_2 = view_1 = None
        getitem_3 = auto_functionalized[1];  auto_functionalized = None
        getitem_4: "bf16[2, 128]" = getitem_3[0];  getitem_3 = None
        view_3: "bf16[256]" = torch.ops.aten.view.default(getitem_4, [256]);  getitem_4 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_param.py:478 in init_unsharded_param, code: torch.ops.fsdp.set_.default(self._unsharded_param, unsharded_param)
        as_strided_1: "bf16[16, 16]" = torch.ops.aten.as_strided.default(view_3, [16, 16], [16, 1], 0);  view_3 = None
        set_ = torch.ops.fsdp.set_.default(primals_3, as_strided_1);  as_strided_1 = set_ = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/compile/fsdp/mlp.py:97 in forward, code: outputs = self.lin0(inputs).relu()
        permute: "bf16[16, 16]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        mm: "bf16[1, 16]" = torch.ops.aten.mm.default(primals_1, permute);  permute = None
        relu: "bf16[1, 16]" = torch.ops.aten.relu.default(mm);  mm = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_param.py:733 in free_storage, code: storage.resize_(0)
        resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(empty, 0);  empty = resize_storage_bytes_ = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_collectives.py:151 in foreach_all_gather, code: all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(
        all_gather_copy_in_1 = torch.ops.fsdp.all_gather_copy_in.default([primals_5], [128], 128, 2, 0, torch.bfloat16, device(type='cuda', index=0));  primals_5 = None
        getitem_5: "bf16[128]" = all_gather_copy_in_1[0];  all_gather_copy_in_1 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:204 in all_gather_tensor, code: tensor = torch.ops._c10d_functional.all_gather_into_tensor(
        all_gather_into_tensor_1: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_5, 2, '0');  getitem_5 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:140 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_1: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_1);  all_gather_into_tensor_1 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_param.py:409 in init_all_gather_outputs, code: torch.empty(torch.Size([numel * world_size]), dtype=dtype, device=device)
        empty_1: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_collectives.py:274 in foreach_all_gather_copy_out, code: out = [t.view(world_size, -1) for t in gen]
        view_6: "bf16[2, 128]" = torch.ops.aten.view.default(empty_1, [2, -1])
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_collectives.py:275 in foreach_all_gather_copy_out, code: torch.ops.fsdp.split_with_sizes_copy(
        view_7: "bf16[2, 128]" = torch.ops.aten.view.default(wait_tensor_1, [2, -1]);  wait_tensor_1 = None
        auto_functionalized_1 = torch.ops.higher_order.auto_functionalized(torch.ops.fsdp.split_with_sizes_copy.default, all_gather_output = view_7, all_gather_input_split_sizes = [128], dim = 1, out = [view_6]);  view_7 = view_6 = None
        getitem_8 = auto_functionalized_1[1];  auto_functionalized_1 = None
        getitem_9: "bf16[2, 128]" = getitem_8[0];  getitem_8 = None
        view_8: "bf16[256]" = torch.ops.aten.view.default(getitem_9, [256]);  getitem_9 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_param.py:478 in init_unsharded_param, code: torch.ops.fsdp.set_.default(self._unsharded_param, unsharded_param)
        as_strided_3: "bf16[16, 16]" = torch.ops.aten.as_strided.default(view_8, [16, 16], [16, 1], 0);  view_8 = None
        set__1 = torch.ops.fsdp.set_.default(primals_6, as_strided_3);  as_strided_3 = set__1 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/compile/fsdp/mlp.py:97 in forward, code: outputs = self.lin0(inputs).relu()
        permute_1: "bf16[16, 16]" = torch.ops.aten.permute.default(primals_6, [1, 0])
        mm_1: "bf16[1, 16]" = torch.ops.aten.mm.default(relu, permute_1);  permute_1 = None
        relu_1: "bf16[1, 16]" = torch.ops.aten.relu.default(mm_1);  mm_1 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_param.py:733 in free_storage, code: storage.resize_(0)
        resize_storage_bytes__1 = torch.ops.inductor.resize_storage_bytes_.default(empty_1, 0);  empty_1 = resize_storage_bytes__1 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_collectives.py:151 in foreach_all_gather, code: all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(
        all_gather_copy_in_2 = torch.ops.fsdp.all_gather_copy_in.default([primals_7], [128], 128, 2, 0, torch.bfloat16, device(type='cuda', index=0));  primals_7 = None
        getitem_10: "bf16[128]" = all_gather_copy_in_2[0];  all_gather_copy_in_2 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:204 in all_gather_tensor, code: tensor = torch.ops._c10d_functional.all_gather_into_tensor(
        all_gather_into_tensor_2: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_10, 2, '0');  getitem_10 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:140 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_2: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_2);  all_gather_into_tensor_2 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_param.py:409 in init_all_gather_outputs, code: torch.empty(torch.Size([numel * world_size]), dtype=dtype, device=device)
        empty_2: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_collectives.py:274 in foreach_all_gather_copy_out, code: out = [t.view(world_size, -1) for t in gen]
        view_11: "bf16[2, 128]" = torch.ops.aten.view.default(empty_2, [2, -1])
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.7/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_composable/fsdp/_fsdp_collectives.py:275 in foreach_all_gather_copy_out, code: torch.ops.fsdp.split_with_sizes_copy(
        view_12: "bf16[2, 128]" = torch.ops.aten.view.default(wait_tensor_2, [2, -1]);  wait_tensor_2 = None
        auto_functionalized_2 = torch.ops.higher_order.auto_functionalized(torch.ops.fsdp.split_with_sizes_copy.default, all_gather_output = view_12, all_gather_input_split_sizes = [128], dim = 1, out = [view_11]);  view_12 = view_11 = None
        getitem_13 = auto_functionalized_2[1];  auto_functionalized_2 = None
        getitem_14: "bf16[2, 128]" = getitem_13[0];  getitem_13 = None
        view_13: "bf16[256]" = torch.ops.aten.view.default(getitem_14, [256]);  getitem_14 = None
        
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
        