class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 4, 32, 64]", arg1_1: "f32[2, 2, 32, 64]", arg2_1: "f32[2, 2, 32, 64]"):
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:37 in torch_attn_primitives, code: mask = torch.ones(seqlen, seqlen, dtype=torch.bool, device=scores.device).triu_(1)
        iota: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        unsqueeze_2: "i64[1, 32]" = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
        iota_1: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        unsqueeze_3: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        sub: "i64[32, 32]" = torch.ops.aten.sub.Tensor(unsqueeze_2, unsqueeze_3);  unsqueeze_2 = unsqueeze_3 = None
        ge: "b8[32, 32]" = torch.ops.aten.ge.Scalar(sub, 1);  sub = None
        full_default: "b8[32, 32]" = torch.ops.aten.full.default([32, 32], True, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        logical_and: "b8[32, 32]" = torch.ops.aten.logical_and.default(ge, full_default);  ge = full_default = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:38 in torch_attn_primitives, code: scores.masked_fill_(mask[None], float("-inf"))
        unsqueeze_5: "b8[1, 32, 32]" = torch.ops.aten.unsqueeze.default(logical_and, 0);  logical_and = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:35 in torch_attn_primitives, code: scores = (q @ k.transpose(-1, -2)) / scale
        expand_2: "f32[2, 4, 32, 64]" = torch.ops.aten.expand.default(arg0_1, [2, 4, 32, 64])
        view_2: "f32[8, 32, 64]" = torch.ops.aten.reshape.default(expand_2, [8, 32, 64]);  expand_2 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:31 in torch_attn_primitives, code: k = k.unsqueeze(2).expand(-1, -1, gqa_ratio, -1, -1).flatten(1, 2)
        unsqueeze: "f32[2, 2, 1, 32, 64]" = torch.ops.aten.unsqueeze.default(arg1_1, 2)
        expand: "f32[2, 2, 2, 32, 64]" = torch.ops.aten.expand.default(unsqueeze, [-1, -1, 2, -1, -1]);  unsqueeze = None
        clone: "f32[2, 2, 2, 32, 64]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view: "f32[2, 4, 32, 64]" = torch.ops.aten.reshape.default(clone, [2, 4, 32, 64]);  clone = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:35 in torch_attn_primitives, code: scores = (q @ k.transpose(-1, -2)) / scale
        permute: "f32[2, 4, 64, 32]" = torch.ops.aten.permute.default(view, [0, 1, 3, 2]);  view = None
        expand_3: "f32[2, 4, 64, 32]" = torch.ops.aten.expand.default(permute, [2, 4, 64, 32]);  permute = None
        view_3: "f32[8, 64, 32]" = torch.ops.aten.reshape.default(expand_3, [8, 64, 32]);  expand_3 = None
        bmm: "f32[8, 32, 32]" = torch.ops.aten.bmm.default(view_2, view_3);  view_2 = view_3 = None
        view_4: "f32[2, 4, 32, 32]" = torch.ops.aten.reshape.default(bmm, [2, 4, 32, 32]);  bmm = None
        div: "f32[2, 4, 32, 32]" = torch.ops.aten.div.Tensor(view_4, 8.0);  view_4 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:38 in torch_attn_primitives, code: scores.masked_fill_(mask[None], float("-inf"))
        where: "f32[2, 4, 32, 32]" = torch.ops.aten.where.self(unsqueeze_5, full_default_1, div);  unsqueeze_5 = full_default_1 = div = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:39 in torch_attn_primitives, code: max_score = scores.max(dim=-1, keepdim=True).values
        max_1 = torch.ops.aten.max.dim(where, -1, True)
        getitem: "f32[2, 4, 32, 1]" = max_1[0];  max_1 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:35 in torch_attn_primitives, code: scores = (q @ k.transpose(-1, -2)) / scale
        expand_8: "f32[2, 4, 32, 64]" = torch.ops.aten.expand.default(arg0_1, [2, 4, 32, 64])
        view_10: "f32[8, 32, 64]" = torch.ops.aten.reshape.default(expand_8, [8, 32, 64]);  expand_8 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:77 in ring_send_recv, code: send_buffer.contiguous(),
        clone_2: "f32[2, 2, 32, 64]" = torch.ops.aten.clone.default(arg1_1, memory_format = torch.contiguous_format);  arg1_1 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:483 in all_to_all_single, code: tensor = torch.ops._c10d_functional.all_to_all_single(  # type: ignore[attr-defined]
        all_to_all_single: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.all_to_all_single.default(clone_2, [0, 0, 2, 0], [2, 0, 0, 0], '0');  clone_2 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:140 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single);  all_to_all_single = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:31 in torch_attn_primitives, code: k = k.unsqueeze(2).expand(-1, -1, gqa_ratio, -1, -1).flatten(1, 2)
        unsqueeze_6: "f32[2, 2, 1, 32, 64]" = torch.ops.aten.unsqueeze.default(wait_tensor, 2)
        expand_6: "f32[2, 2, 2, 32, 64]" = torch.ops.aten.expand.default(unsqueeze_6, [-1, -1, 2, -1, -1]);  unsqueeze_6 = None
        clone_4: "f32[2, 2, 2, 32, 64]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_8: "f32[2, 4, 32, 64]" = torch.ops.aten.reshape.default(clone_4, [2, 4, 32, 64]);  clone_4 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:35 in torch_attn_primitives, code: scores = (q @ k.transpose(-1, -2)) / scale
        permute_1: "f32[2, 4, 64, 32]" = torch.ops.aten.permute.default(view_8, [0, 1, 3, 2]);  view_8 = None
        expand_9: "f32[2, 4, 64, 32]" = torch.ops.aten.expand.default(permute_1, [2, 4, 64, 32]);  permute_1 = None
        view_11: "f32[8, 64, 32]" = torch.ops.aten.reshape.default(expand_9, [8, 64, 32]);  expand_9 = None
        bmm_2: "f32[8, 32, 32]" = torch.ops.aten.bmm.default(view_10, view_11);  view_10 = view_11 = None
        view_12: "f32[2, 4, 32, 32]" = torch.ops.aten.reshape.default(bmm_2, [2, 4, 32, 32]);  bmm_2 = None
        div_1: "f32[2, 4, 32, 32]" = torch.ops.aten.div.Tensor(view_12, 8.0);  view_12 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:39 in torch_attn_primitives, code: max_score = scores.max(dim=-1, keepdim=True).values
        max_2 = torch.ops.aten.max.dim(div_1, -1, True)
        getitem_2: "f32[2, 4, 32, 1]" = max_2[0];  max_2 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:35 in torch_attn_primitives, code: scores = (q @ k.transpose(-1, -2)) / scale
        expand_14: "f32[2, 4, 32, 64]" = torch.ops.aten.expand.default(arg0_1, [2, 4, 32, 64])
        view_18: "f32[8, 32, 64]" = torch.ops.aten.reshape.default(expand_14, [8, 32, 64]);  expand_14 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:483 in all_to_all_single, code: tensor = torch.ops._c10d_functional.all_to_all_single(  # type: ignore[attr-defined]
        all_to_all_single_2: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.all_to_all_single.default(wait_tensor, [0, 0, 2, 0], [2, 0, 0, 0], '0');  wait_tensor = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:140 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_2: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_2);  all_to_all_single_2 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:31 in torch_attn_primitives, code: k = k.unsqueeze(2).expand(-1, -1, gqa_ratio, -1, -1).flatten(1, 2)
        unsqueeze_8: "f32[2, 2, 1, 32, 64]" = torch.ops.aten.unsqueeze.default(wait_tensor_2, 2)
        expand_12: "f32[2, 2, 2, 32, 64]" = torch.ops.aten.expand.default(unsqueeze_8, [-1, -1, 2, -1, -1]);  unsqueeze_8 = None
        clone_6: "f32[2, 2, 2, 32, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_16: "f32[2, 4, 32, 64]" = torch.ops.aten.reshape.default(clone_6, [2, 4, 32, 64]);  clone_6 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:35 in torch_attn_primitives, code: scores = (q @ k.transpose(-1, -2)) / scale
        permute_2: "f32[2, 4, 64, 32]" = torch.ops.aten.permute.default(view_16, [0, 1, 3, 2]);  view_16 = None
        expand_15: "f32[2, 4, 64, 32]" = torch.ops.aten.expand.default(permute_2, [2, 4, 64, 32]);  permute_2 = None
        view_19: "f32[8, 64, 32]" = torch.ops.aten.reshape.default(expand_15, [8, 64, 32]);  expand_15 = None
        bmm_4: "f32[8, 32, 32]" = torch.ops.aten.bmm.default(view_18, view_19);  view_18 = view_19 = None
        view_20: "f32[2, 4, 32, 32]" = torch.ops.aten.reshape.default(bmm_4, [2, 4, 32, 32]);  bmm_4 = None
        div_2: "f32[2, 4, 32, 32]" = torch.ops.aten.div.Tensor(view_20, 8.0);  view_20 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:39 in torch_attn_primitives, code: max_score = scores.max(dim=-1, keepdim=True).values
        max_3 = torch.ops.aten.max.dim(div_2, -1, True)
        getitem_4: "f32[2, 4, 32, 1]" = max_3[0];  max_3 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:35 in torch_attn_primitives, code: scores = (q @ k.transpose(-1, -2)) / scale
        expand_20: "f32[2, 4, 32, 64]" = torch.ops.aten.expand.default(arg0_1, [2, 4, 32, 64]);  arg0_1 = None
        view_26: "f32[8, 32, 64]" = torch.ops.aten.reshape.default(expand_20, [8, 32, 64]);  expand_20 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:483 in all_to_all_single, code: tensor = torch.ops._c10d_functional.all_to_all_single(  # type: ignore[attr-defined]
        all_to_all_single_4: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.all_to_all_single.default(wait_tensor_2, [0, 0, 2, 0], [2, 0, 0, 0], '0');  wait_tensor_2 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:140 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_4: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_4);  all_to_all_single_4 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:31 in torch_attn_primitives, code: k = k.unsqueeze(2).expand(-1, -1, gqa_ratio, -1, -1).flatten(1, 2)
        unsqueeze_10: "f32[2, 2, 1, 32, 64]" = torch.ops.aten.unsqueeze.default(wait_tensor_4, 2);  wait_tensor_4 = None
        expand_18: "f32[2, 2, 2, 32, 64]" = torch.ops.aten.expand.default(unsqueeze_10, [-1, -1, 2, -1, -1]);  unsqueeze_10 = None
        clone_8: "f32[2, 2, 2, 32, 64]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_24: "f32[2, 4, 32, 64]" = torch.ops.aten.reshape.default(clone_8, [2, 4, 32, 64]);  clone_8 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:35 in torch_attn_primitives, code: scores = (q @ k.transpose(-1, -2)) / scale
        permute_3: "f32[2, 4, 64, 32]" = torch.ops.aten.permute.default(view_24, [0, 1, 3, 2]);  view_24 = None
        expand_21: "f32[2, 4, 64, 32]" = torch.ops.aten.expand.default(permute_3, [2, 4, 64, 32]);  permute_3 = None
        view_27: "f32[8, 64, 32]" = torch.ops.aten.reshape.default(expand_21, [8, 64, 32]);  expand_21 = None
        bmm_6: "f32[8, 32, 32]" = torch.ops.aten.bmm.default(view_26, view_27);  view_26 = view_27 = None
        view_28: "f32[2, 4, 32, 32]" = torch.ops.aten.reshape.default(bmm_6, [2, 4, 32, 32]);  bmm_6 = None
        div_3: "f32[2, 4, 32, 32]" = torch.ops.aten.div.Tensor(view_28, 8.0);  view_28 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:39 in torch_attn_primitives, code: max_score = scores.max(dim=-1, keepdim=True).values
        max_4 = torch.ops.aten.max.dim(div_3, -1, True)
        getitem_6: "f32[2, 4, 32, 1]" = max_4[0];  max_4 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:114 in torch_ring_attn_prefill, code: new_max_score = torch.maximum(max_score, ring_max_score)
        maximum: "f32[2, 4, 32, 1]" = torch.ops.aten.maximum.default(getitem, getitem_2)
        maximum_1: "f32[2, 4, 32, 1]" = torch.ops.aten.maximum.default(maximum, getitem_4)
        maximum_2: "f32[2, 4, 32, 1]" = torch.ops.aten.maximum.default(maximum_1, getitem_6)
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:115 in torch_ring_attn_prefill, code: numerator = (ring_max_score - new_max_score).exp() * ring_numerator + (
        sub_13: "f32[2, 4, 32, 1]" = torch.ops.aten.sub.Tensor(getitem_6, maximum_2)
        exp_12: "f32[2, 4, 32, 1]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:40 in torch_attn_primitives, code: scores = (scores - max_score).exp()
        sub_12: "f32[2, 4, 32, 32]" = torch.ops.aten.sub.Tensor(div_3, getitem_6);  div_3 = None
        exp_11: "f32[2, 4, 32, 32]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:41 in torch_attn_primitives, code: numerator = scores @ v
        expand_22: "f32[2, 4, 32, 32]" = torch.ops.aten.expand.default(exp_11, [2, 4, 32, 32])
        view_29: "f32[8, 32, 32]" = torch.ops.aten.reshape.default(expand_22, [8, 32, 32]);  expand_22 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:77 in ring_send_recv, code: send_buffer.contiguous(),
        clone_3: "f32[2, 2, 32, 64]" = torch.ops.aten.clone.default(arg2_1, memory_format = torch.contiguous_format)
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:483 in all_to_all_single, code: tensor = torch.ops._c10d_functional.all_to_all_single(  # type: ignore[attr-defined]
        all_to_all_single_1: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.all_to_all_single.default(clone_3, [0, 0, 2, 0], [2, 0, 0, 0], '0');  clone_3 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:140 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_1: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_1);  all_to_all_single_1 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:483 in all_to_all_single, code: tensor = torch.ops._c10d_functional.all_to_all_single(  # type: ignore[attr-defined]
        all_to_all_single_3: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.all_to_all_single.default(wait_tensor_1, [0, 0, 2, 0], [2, 0, 0, 0], '0')
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:140 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_3: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_3);  all_to_all_single_3 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:483 in all_to_all_single, code: tensor = torch.ops._c10d_functional.all_to_all_single(  # type: ignore[attr-defined]
        all_to_all_single_5: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.all_to_all_single.default(wait_tensor_3, [0, 0, 2, 0], [2, 0, 0, 0], '0')
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:140 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_5: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_5);  all_to_all_single_5 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:32 in torch_attn_primitives, code: v = v.unsqueeze(2).expand(-1, -1, gqa_ratio, -1, -1).flatten(1, 2)
        unsqueeze_11: "f32[2, 2, 1, 32, 64]" = torch.ops.aten.unsqueeze.default(wait_tensor_5, 2);  wait_tensor_5 = None
        expand_19: "f32[2, 2, 2, 32, 64]" = torch.ops.aten.expand.default(unsqueeze_11, [-1, -1, 2, -1, -1]);  unsqueeze_11 = None
        clone_9: "f32[2, 2, 2, 32, 64]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_25: "f32[2, 4, 32, 64]" = torch.ops.aten.reshape.default(clone_9, [2, 4, 32, 64]);  clone_9 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:41 in torch_attn_primitives, code: numerator = scores @ v
        expand_23: "f32[2, 4, 32, 64]" = torch.ops.aten.expand.default(view_25, [2, 4, 32, 64]);  view_25 = None
        view_30: "f32[8, 32, 64]" = torch.ops.aten.reshape.default(expand_23, [8, 32, 64]);  expand_23 = None
        bmm_7: "f32[8, 32, 64]" = torch.ops.aten.bmm.default(view_29, view_30);  view_29 = view_30 = None
        view_31: "f32[2, 4, 32, 64]" = torch.ops.aten.reshape.default(bmm_7, [2, 4, 32, 64]);  bmm_7 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:115 in torch_ring_attn_prefill, code: numerator = (ring_max_score - new_max_score).exp() * ring_numerator + (
        mul_8: "f32[2, 4, 32, 64]" = torch.ops.aten.mul.Tensor(exp_12, view_31);  exp_12 = view_31 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:116 in torch_ring_attn_prefill, code: max_score - new_max_score
        sub_14: "f32[2, 4, 32, 1]" = torch.ops.aten.sub.Tensor(maximum_1, maximum_2)
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:117 in torch_ring_attn_prefill, code: ).exp() * numerator
        exp_13: "f32[2, 4, 32, 1]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:115 in torch_ring_attn_prefill, code: numerator = (ring_max_score - new_max_score).exp() * ring_numerator + (
        sub_8: "f32[2, 4, 32, 1]" = torch.ops.aten.sub.Tensor(getitem_4, maximum_1)
        exp_7: "f32[2, 4, 32, 1]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:40 in torch_attn_primitives, code: scores = (scores - max_score).exp()
        sub_7: "f32[2, 4, 32, 32]" = torch.ops.aten.sub.Tensor(div_2, getitem_4);  div_2 = None
        exp_6: "f32[2, 4, 32, 32]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:41 in torch_attn_primitives, code: numerator = scores @ v
        expand_16: "f32[2, 4, 32, 32]" = torch.ops.aten.expand.default(exp_6, [2, 4, 32, 32])
        view_21: "f32[8, 32, 32]" = torch.ops.aten.reshape.default(expand_16, [8, 32, 32]);  expand_16 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:32 in torch_attn_primitives, code: v = v.unsqueeze(2).expand(-1, -1, gqa_ratio, -1, -1).flatten(1, 2)
        unsqueeze_9: "f32[2, 2, 1, 32, 64]" = torch.ops.aten.unsqueeze.default(wait_tensor_3, 2);  wait_tensor_3 = None
        expand_13: "f32[2, 2, 2, 32, 64]" = torch.ops.aten.expand.default(unsqueeze_9, [-1, -1, 2, -1, -1]);  unsqueeze_9 = None
        clone_7: "f32[2, 2, 2, 32, 64]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_17: "f32[2, 4, 32, 64]" = torch.ops.aten.reshape.default(clone_7, [2, 4, 32, 64]);  clone_7 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:41 in torch_attn_primitives, code: numerator = scores @ v
        expand_17: "f32[2, 4, 32, 64]" = torch.ops.aten.expand.default(view_17, [2, 4, 32, 64]);  view_17 = None
        view_22: "f32[8, 32, 64]" = torch.ops.aten.reshape.default(expand_17, [8, 32, 64]);  expand_17 = None
        bmm_5: "f32[8, 32, 64]" = torch.ops.aten.bmm.default(view_21, view_22);  view_21 = view_22 = None
        view_23: "f32[2, 4, 32, 64]" = torch.ops.aten.reshape.default(bmm_5, [2, 4, 32, 64]);  bmm_5 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:115 in torch_ring_attn_prefill, code: numerator = (ring_max_score - new_max_score).exp() * ring_numerator + (
        mul_4: "f32[2, 4, 32, 64]" = torch.ops.aten.mul.Tensor(exp_7, view_23);  exp_7 = view_23 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:116 in torch_ring_attn_prefill, code: max_score - new_max_score
        sub_9: "f32[2, 4, 32, 1]" = torch.ops.aten.sub.Tensor(maximum, maximum_1)
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:117 in torch_ring_attn_prefill, code: ).exp() * numerator
        exp_8: "f32[2, 4, 32, 1]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:115 in torch_ring_attn_prefill, code: numerator = (ring_max_score - new_max_score).exp() * ring_numerator + (
        sub_3: "f32[2, 4, 32, 1]" = torch.ops.aten.sub.Tensor(getitem_2, maximum)
        exp_2: "f32[2, 4, 32, 1]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:40 in torch_attn_primitives, code: scores = (scores - max_score).exp()
        sub_2: "f32[2, 4, 32, 32]" = torch.ops.aten.sub.Tensor(div_1, getitem_2);  div_1 = None
        exp_1: "f32[2, 4, 32, 32]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:41 in torch_attn_primitives, code: numerator = scores @ v
        expand_10: "f32[2, 4, 32, 32]" = torch.ops.aten.expand.default(exp_1, [2, 4, 32, 32])
        view_13: "f32[8, 32, 32]" = torch.ops.aten.reshape.default(expand_10, [8, 32, 32]);  expand_10 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:32 in torch_attn_primitives, code: v = v.unsqueeze(2).expand(-1, -1, gqa_ratio, -1, -1).flatten(1, 2)
        unsqueeze_7: "f32[2, 2, 1, 32, 64]" = torch.ops.aten.unsqueeze.default(wait_tensor_1, 2);  wait_tensor_1 = None
        expand_7: "f32[2, 2, 2, 32, 64]" = torch.ops.aten.expand.default(unsqueeze_7, [-1, -1, 2, -1, -1]);  unsqueeze_7 = None
        clone_5: "f32[2, 2, 2, 32, 64]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_9: "f32[2, 4, 32, 64]" = torch.ops.aten.reshape.default(clone_5, [2, 4, 32, 64]);  clone_5 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:41 in torch_attn_primitives, code: numerator = scores @ v
        expand_11: "f32[2, 4, 32, 64]" = torch.ops.aten.expand.default(view_9, [2, 4, 32, 64]);  view_9 = None
        view_14: "f32[8, 32, 64]" = torch.ops.aten.reshape.default(expand_11, [8, 32, 64]);  expand_11 = None
        bmm_3: "f32[8, 32, 64]" = torch.ops.aten.bmm.default(view_13, view_14);  view_13 = view_14 = None
        view_15: "f32[2, 4, 32, 64]" = torch.ops.aten.reshape.default(bmm_3, [2, 4, 32, 64]);  bmm_3 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:115 in torch_ring_attn_prefill, code: numerator = (ring_max_score - new_max_score).exp() * ring_numerator + (
        mul: "f32[2, 4, 32, 64]" = torch.ops.aten.mul.Tensor(exp_2, view_15);  exp_2 = view_15 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:116 in torch_ring_attn_prefill, code: max_score - new_max_score
        sub_4: "f32[2, 4, 32, 1]" = torch.ops.aten.sub.Tensor(getitem, maximum)
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:117 in torch_ring_attn_prefill, code: ).exp() * numerator
        exp_3: "f32[2, 4, 32, 1]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:40 in torch_attn_primitives, code: scores = (scores - max_score).exp()
        sub_1: "f32[2, 4, 32, 32]" = torch.ops.aten.sub.Tensor(where, getitem);  where = None
        exp: "f32[2, 4, 32, 32]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:41 in torch_attn_primitives, code: numerator = scores @ v
        expand_4: "f32[2, 4, 32, 32]" = torch.ops.aten.expand.default(exp, [2, 4, 32, 32])
        view_5: "f32[8, 32, 32]" = torch.ops.aten.reshape.default(expand_4, [8, 32, 32]);  expand_4 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:32 in torch_attn_primitives, code: v = v.unsqueeze(2).expand(-1, -1, gqa_ratio, -1, -1).flatten(1, 2)
        unsqueeze_1: "f32[2, 2, 1, 32, 64]" = torch.ops.aten.unsqueeze.default(arg2_1, 2);  arg2_1 = None
        expand_1: "f32[2, 2, 2, 32, 64]" = torch.ops.aten.expand.default(unsqueeze_1, [-1, -1, 2, -1, -1]);  unsqueeze_1 = None
        clone_1: "f32[2, 2, 2, 32, 64]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_1: "f32[2, 4, 32, 64]" = torch.ops.aten.reshape.default(clone_1, [2, 4, 32, 64]);  clone_1 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:41 in torch_attn_primitives, code: numerator = scores @ v
        expand_5: "f32[2, 4, 32, 64]" = torch.ops.aten.expand.default(view_1, [2, 4, 32, 64]);  view_1 = None
        view_6: "f32[8, 32, 64]" = torch.ops.aten.reshape.default(expand_5, [8, 32, 64]);  expand_5 = None
        bmm_1: "f32[8, 32, 64]" = torch.ops.aten.bmm.default(view_5, view_6);  view_5 = view_6 = None
        view_7: "f32[2, 4, 32, 64]" = torch.ops.aten.reshape.default(bmm_1, [2, 4, 32, 64]);  bmm_1 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:115 in torch_ring_attn_prefill, code: numerator = (ring_max_score - new_max_score).exp() * ring_numerator + (
        mul_1: "f32[2, 4, 32, 64]" = torch.ops.aten.mul.Tensor(exp_3, view_7);  exp_3 = view_7 = None
        add: "f32[2, 4, 32, 64]" = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
        mul_5: "f32[2, 4, 32, 64]" = torch.ops.aten.mul.Tensor(exp_8, add);  exp_8 = add = None
        add_2: "f32[2, 4, 32, 64]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        mul_9: "f32[2, 4, 32, 64]" = torch.ops.aten.mul.Tensor(exp_13, add_2);  exp_13 = add_2 = None
        add_4: "f32[2, 4, 32, 64]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:118 in torch_ring_attn_prefill, code: denominator = (ring_max_score - new_max_score).exp() * ring_denominator + (
        sub_15: "f32[2, 4, 32, 1]" = torch.ops.aten.sub.Tensor(getitem_6, maximum_2);  getitem_6 = None
        exp_14: "f32[2, 4, 32, 1]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:42 in torch_attn_primitives, code: denominator = scores.sum(dim=-1, keepdim=True)
        sum_4: "f32[2, 4, 32, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True);  exp_11 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:118 in torch_ring_attn_prefill, code: denominator = (ring_max_score - new_max_score).exp() * ring_denominator + (
        mul_10: "f32[2, 4, 32, 1]" = torch.ops.aten.mul.Tensor(exp_14, sum_4);  exp_14 = sum_4 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:119 in torch_ring_attn_prefill, code: max_score - new_max_score
        sub_16: "f32[2, 4, 32, 1]" = torch.ops.aten.sub.Tensor(maximum_1, maximum_2);  maximum_2 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:120 in torch_ring_attn_prefill, code: ).exp() * denominator
        exp_15: "f32[2, 4, 32, 1]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:118 in torch_ring_attn_prefill, code: denominator = (ring_max_score - new_max_score).exp() * ring_denominator + (
        sub_10: "f32[2, 4, 32, 1]" = torch.ops.aten.sub.Tensor(getitem_4, maximum_1);  getitem_4 = None
        exp_9: "f32[2, 4, 32, 1]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:42 in torch_attn_primitives, code: denominator = scores.sum(dim=-1, keepdim=True)
        sum_3: "f32[2, 4, 32, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True);  exp_6 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:118 in torch_ring_attn_prefill, code: denominator = (ring_max_score - new_max_score).exp() * ring_denominator + (
        mul_6: "f32[2, 4, 32, 1]" = torch.ops.aten.mul.Tensor(exp_9, sum_3);  exp_9 = sum_3 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:119 in torch_ring_attn_prefill, code: max_score - new_max_score
        sub_11: "f32[2, 4, 32, 1]" = torch.ops.aten.sub.Tensor(maximum, maximum_1);  maximum_1 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:120 in torch_ring_attn_prefill, code: ).exp() * denominator
        exp_10: "f32[2, 4, 32, 1]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:118 in torch_ring_attn_prefill, code: denominator = (ring_max_score - new_max_score).exp() * ring_denominator + (
        sub_5: "f32[2, 4, 32, 1]" = torch.ops.aten.sub.Tensor(getitem_2, maximum);  getitem_2 = None
        exp_4: "f32[2, 4, 32, 1]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:42 in torch_attn_primitives, code: denominator = scores.sum(dim=-1, keepdim=True)
        sum_2: "f32[2, 4, 32, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True);  exp_1 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:118 in torch_ring_attn_prefill, code: denominator = (ring_max_score - new_max_score).exp() * ring_denominator + (
        mul_2: "f32[2, 4, 32, 1]" = torch.ops.aten.mul.Tensor(exp_4, sum_2);  exp_4 = sum_2 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:119 in torch_ring_attn_prefill, code: max_score - new_max_score
        sub_6: "f32[2, 4, 32, 1]" = torch.ops.aten.sub.Tensor(getitem, maximum);  getitem = maximum = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:120 in torch_ring_attn_prefill, code: ).exp() * denominator
        exp_5: "f32[2, 4, 32, 1]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:42 in torch_attn_primitives, code: denominator = scores.sum(dim=-1, keepdim=True)
        sum_1: "f32[2, 4, 32, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True);  exp = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:118 in torch_ring_attn_prefill, code: denominator = (ring_max_score - new_max_score).exp() * ring_denominator + (
        mul_3: "f32[2, 4, 32, 1]" = torch.ops.aten.mul.Tensor(exp_5, sum_1);  exp_5 = sum_1 = None
        add_1: "f32[2, 4, 32, 1]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        mul_7: "f32[2, 4, 32, 1]" = torch.ops.aten.mul.Tensor(exp_10, add_1);  exp_10 = add_1 = None
        add_3: "f32[2, 4, 32, 1]" = torch.ops.aten.add.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
        mul_11: "f32[2, 4, 32, 1]" = torch.ops.aten.mul.Tensor(exp_15, add_3);  exp_15 = add_3 = None
        add_5: "f32[2, 4, 32, 1]" = torch.ops.aten.add.Tensor(mul_10, mul_11);  mul_10 = mul_11 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:124 in torch_ring_attn_prefill, code: return numerator / denominator
        div_4: "f32[2, 4, 32, 64]" = torch.ops.aten.div.Tensor(add_4, add_5);  add_4 = add_5 = None
        return (div_4,)
        