class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[1, 16]", primals_2: "bf16[128]", primals_3: "bf16[16, 16]", primals_4, primals_5: "bf16[128]", primals_6: "bf16[16, 16]", primals_7: "bf16[128]", primals_8: "bf16[16, 16]"):
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:158 in foreach_all_gather, code: all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(
        all_gather_copy_in = torch.ops.fsdp.all_gather_copy_in.default([primals_2], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1));  primals_2 = None
        getitem: "bf16[128]" = all_gather_copy_in[0];  all_gather_copy_in = None
        
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:205 in all_gather_tensor, code: tensor = torch.ops._c10d_functional.all_gather_into_tensor(
        all_gather_into_tensor: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem, 2, '0');  getitem = None
        
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:141 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
        
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py:457 in init_all_gather_outputs, code: torch.empty(torch.Size([numel * world_size]), dtype=dtype, device=device)
        empty: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:298 in foreach_all_gather_copy_out, code: torch.ops.fsdp.split_with_sizes_copy(
        view_2: "bf16[2, 128]" = torch.ops.aten.view.default(wait_tensor, [2, -1]);  wait_tensor = None
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.fsdp.split_with_sizes_copy.default, all_gather_output = view_2, all_gather_input_split_sizes = [128], dim = 1, _out_length = 1, _out_0_base_index = 0, _out_0_size = (2, 128), _out_0_stride = (128, 1), _out_0_storage_offset = 0, _all_bases = [empty]);  view_2 = empty = None
        getitem_3: "bf16[256]" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py:530 in init_unsharded_param, code: self._unsharded_param.untyped_storage().resize_(
        resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(primals_3, 512);  resize_storage_bytes_ = None
        
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py:533 in init_unsharded_param, code: torch.ops.fsdp.copy_(self._unsharded_param, unsharded_param)
        as_strided_1: "bf16[16, 16]" = torch.ops.aten.as_strided.default(getitem_3, [16, 16], [16, 1], 0);  getitem_3 = None
        copy_ = torch.ops.fsdp.copy_.default(primals_3, as_strided_1);  as_strided_1 = copy_ = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/compile/fsdp/mlp_fwd_only.py:48 in forward, code: outputs = self.lin0(inputs).relu()
        permute: "bf16[16, 16]" = torch.ops.aten.permute.default(primals_3, [1, 0])
        mm: "bf16[1, 16]" = torch.ops.aten.mm.default(primals_1, permute);  permute = None
        relu: "bf16[1, 16]" = torch.ops.aten.relu.default(mm);  mm = None
        
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py:675 in free_unsharded_param, code: self._unsharded_param.untyped_storage().resize_(0)
        resize_storage_bytes__1 = torch.ops.inductor.resize_storage_bytes_.default(primals_3, 0);  primals_3 = resize_storage_bytes__1 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:158 in foreach_all_gather, code: all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(
        all_gather_copy_in_1 = torch.ops.fsdp.all_gather_copy_in.default([primals_5], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1));  primals_5 = None
        getitem_4: "bf16[128]" = all_gather_copy_in_1[0];  all_gather_copy_in_1 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:205 in all_gather_tensor, code: tensor = torch.ops._c10d_functional.all_gather_into_tensor(
        all_gather_into_tensor_1: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_4, 2, '0');  getitem_4 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:141 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_1: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_1);  all_gather_into_tensor_1 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py:457 in init_all_gather_outputs, code: torch.empty(torch.Size([numel * world_size]), dtype=dtype, device=device)
        empty_1: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:298 in foreach_all_gather_copy_out, code: torch.ops.fsdp.split_with_sizes_copy(
        view_5: "bf16[2, 128]" = torch.ops.aten.view.default(wait_tensor_1, [2, -1]);  wait_tensor_1 = None
        auto_functionalized_v2_1 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.fsdp.split_with_sizes_copy.default, all_gather_output = view_5, all_gather_input_split_sizes = [128], dim = 1, _out_length = 1, _out_0_base_index = 0, _out_0_size = (2, 128), _out_0_stride = (128, 1), _out_0_storage_offset = 0, _all_bases = [empty_1]);  view_5 = empty_1 = None
        getitem_7: "bf16[256]" = auto_functionalized_v2_1[1];  auto_functionalized_v2_1 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py:530 in init_unsharded_param, code: self._unsharded_param.untyped_storage().resize_(
        resize_storage_bytes__2 = torch.ops.inductor.resize_storage_bytes_.default(primals_6, 512);  resize_storage_bytes__2 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py:533 in init_unsharded_param, code: torch.ops.fsdp.copy_(self._unsharded_param, unsharded_param)
        as_strided_3: "bf16[16, 16]" = torch.ops.aten.as_strided.default(getitem_7, [16, 16], [16, 1], 0);  getitem_7 = None
        copy__1 = torch.ops.fsdp.copy_.default(primals_6, as_strided_3);  as_strided_3 = copy__1 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/compile/fsdp/mlp_fwd_only.py:48 in forward, code: outputs = self.lin0(inputs).relu()
        permute_1: "bf16[16, 16]" = torch.ops.aten.permute.default(primals_6, [1, 0])
        mm_1: "bf16[1, 16]" = torch.ops.aten.mm.default(relu, permute_1);  permute_1 = None
        relu_1: "bf16[1, 16]" = torch.ops.aten.relu.default(mm_1);  mm_1 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py:675 in free_unsharded_param, code: self._unsharded_param.untyped_storage().resize_(0)
        resize_storage_bytes__3 = torch.ops.inductor.resize_storage_bytes_.default(primals_6, 0);  resize_storage_bytes__3 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:158 in foreach_all_gather, code: all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(
        all_gather_copy_in_2 = torch.ops.fsdp.all_gather_copy_in.default([primals_7], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1));  primals_7 = None
        getitem_8: "bf16[128]" = all_gather_copy_in_2[0];  all_gather_copy_in_2 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:205 in all_gather_tensor, code: tensor = torch.ops._c10d_functional.all_gather_into_tensor(
        all_gather_into_tensor_2: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_8, 2, '0');  getitem_8 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:141 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_2: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_2);  all_gather_into_tensor_2 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py:457 in init_all_gather_outputs, code: torch.empty(torch.Size([numel * world_size]), dtype=dtype, device=device)
        empty_2: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:298 in foreach_all_gather_copy_out, code: torch.ops.fsdp.split_with_sizes_copy(
        view_8: "bf16[2, 128]" = torch.ops.aten.view.default(wait_tensor_2, [2, -1]);  wait_tensor_2 = None
        auto_functionalized_v2_2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.fsdp.split_with_sizes_copy.default, all_gather_output = view_8, all_gather_input_split_sizes = [128], dim = 1, _out_length = 1, _out_0_base_index = 0, _out_0_size = (2, 128), _out_0_stride = (128, 1), _out_0_storage_offset = 0, _all_bases = [empty_2]);  view_8 = empty_2 = None
        getitem_11: "bf16[256]" = auto_functionalized_v2_2[1];  auto_functionalized_v2_2 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py:530 in init_unsharded_param, code: self._unsharded_param.untyped_storage().resize_(
        resize_storage_bytes__4 = torch.ops.inductor.resize_storage_bytes_.default(primals_8, 512);  resize_storage_bytes__4 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py:533 in init_unsharded_param, code: torch.ops.fsdp.copy_(self._unsharded_param, unsharded_param)
        as_strided_5: "bf16[16, 16]" = torch.ops.aten.as_strided.default(getitem_11, [16, 16], [16, 1], 0);  getitem_11 = None
        copy__2 = torch.ops.fsdp.copy_.default(primals_8, as_strided_5);  as_strided_5 = copy__2 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/compile/fsdp/mlp_fwd_only.py:48 in forward, code: outputs = self.lin0(inputs).relu()
        permute_2: "bf16[16, 16]" = torch.ops.aten.permute.default(primals_8, [1, 0])
        mm_2: "bf16[1, 16]" = torch.ops.aten.mm.default(relu_1, permute_2);  permute_2 = None
        relu_2: "bf16[1, 16]" = torch.ops.aten.relu.default(mm_2);  mm_2 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/torch-nightly/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py:675 in free_unsharded_param, code: self._unsharded_param.untyped_storage().resize_(0)
        resize_storage_bytes__5 = torch.ops.inductor.resize_storage_bytes_.default(primals_8, 0);  resize_storage_bytes__5 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/compile/fsdp/mlp_fwd_only.py:48 in forward, code: outputs = self.lin0(inputs).relu()
        le: "b8[1, 16]" = torch.ops.aten.le.Scalar(relu_2, 0)
        return (relu_2, primals_1, primals_6, primals_8, relu, relu_1, le)
        