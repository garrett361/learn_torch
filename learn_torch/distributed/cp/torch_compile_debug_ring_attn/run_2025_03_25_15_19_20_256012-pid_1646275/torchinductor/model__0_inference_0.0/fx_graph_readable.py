class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 4, 32, 64]", arg1_1: "f32[2, 2, 32, 64]", arg2_1: "f32[2, 2, 32, 64]"):
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:31 in torch_attn_primitives, code: k = k.unsqueeze(2).expand(-1, -1, gqa_ratio, -1, -1).flatten(1, 2)
        unsqueeze: "f32[2, 2, 1, 32, 64]" = torch.ops.aten.unsqueeze.default(arg1_1, 2)
        expand: "f32[2, 2, 2, 32, 64]" = torch.ops.aten.expand.default(unsqueeze, [-1, -1, 2, -1, -1]);  unsqueeze = None
        clone: "f32[2, 2, 2, 32, 64]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view: "f32[2, 4, 32, 64]" = torch.ops.aten.view.default(clone, [2, 4, 32, 64]);  clone = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:32 in torch_attn_primitives, code: v = v.unsqueeze(2).expand(-1, -1, gqa_ratio, -1, -1).flatten(1, 2)
        unsqueeze_1: "f32[2, 2, 1, 32, 64]" = torch.ops.aten.unsqueeze.default(arg2_1, 2)
        expand_1: "f32[2, 2, 2, 32, 64]" = torch.ops.aten.expand.default(unsqueeze_1, [-1, -1, 2, -1, -1]);  unsqueeze_1 = None
        clone_1: "f32[2, 2, 2, 32, 64]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_1: "f32[2, 4, 32, 64]" = torch.ops.aten.view.default(clone_1, [2, 4, 32, 64]);  clone_1 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:35 in torch_attn_primitives, code: scores = (q @ k.transpose(-1, -2)) / scale
        permute: "f32[2, 4, 64, 32]" = torch.ops.aten.permute.default(view, [0, 1, 3, 2]);  view = None
        expand_2: "f32[2, 4, 32, 64]" = torch.ops.aten.expand.default(arg0_1, [2, 4, 32, 64]);  arg0_1 = None
        view_2: "f32[8, 32, 64]" = torch.ops.aten.view.default(expand_2, [8, 32, 64]);  expand_2 = None
        expand_3: "f32[2, 4, 64, 32]" = torch.ops.aten.expand.default(permute, [2, 4, 64, 32]);  permute = None
        view_3: "f32[8, 64, 32]" = torch.ops.aten.view.default(expand_3, [8, 64, 32]);  expand_3 = None
        bmm: "f32[8, 32, 32]" = torch.ops.aten.bmm.default(view_2, view_3);  view_2 = view_3 = None
        view_4: "f32[2, 4, 32, 32]" = torch.ops.aten.view.default(bmm, [2, 4, 32, 32]);  bmm = None
        div: "f32[2, 4, 32, 32]" = torch.ops.aten.div.Tensor(view_4, 8.0);  view_4 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:37 in torch_attn_primitives, code: mask = torch.ones(seqlen, seqlen, dtype=torch.bool, device=scores.device).triu_(1)
        full_default: "b8[32, 32]" = torch.ops.aten.full.default([32, 32], True, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        iota: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        unsqueeze_2: "i64[1, 32]" = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
        iota_1: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        unsqueeze_3: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        sub: "i64[32, 32]" = torch.ops.aten.sub.Tensor(unsqueeze_2, unsqueeze_3);  unsqueeze_2 = unsqueeze_3 = None
        ge: "b8[32, 32]" = torch.ops.aten.ge.Scalar(sub, 1);  sub = None
        logical_and: "b8[32, 32]" = torch.ops.aten.logical_and.default(ge, full_default);  ge = full_default = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:38 in torch_attn_primitives, code: scores.masked_fill_(mask[None], float("-inf"))
        unsqueeze_5: "b8[1, 32, 32]" = torch.ops.aten.unsqueeze.default(logical_and, 0);  logical_and = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where: "f32[2, 4, 32, 32]" = torch.ops.aten.where.self(unsqueeze_5, full_default_1, div);  unsqueeze_5 = full_default_1 = div = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:39 in torch_attn_primitives, code: max_score = scores.max(dim=-1, keepdim=True).values
        max_1 = torch.ops.aten.max.dim(where, -1, True)
        getitem: "f32[2, 4, 32, 1]" = max_1[0];  max_1 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:40 in torch_attn_primitives, code: scores = (scores - max_score).exp()
        sub_1: "f32[2, 4, 32, 32]" = torch.ops.aten.sub.Tensor(where, getitem);  where = getitem = None
        exp: "f32[2, 4, 32, 32]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:41 in torch_attn_primitives, code: numerator = scores @ v
        expand_4: "f32[2, 4, 32, 32]" = torch.ops.aten.expand.default(exp, [2, 4, 32, 32])
        view_5: "f32[8, 32, 32]" = torch.ops.aten.view.default(expand_4, [8, 32, 32]);  expand_4 = None
        expand_5: "f32[2, 4, 32, 64]" = torch.ops.aten.expand.default(view_1, [2, 4, 32, 64]);  view_1 = None
        view_6: "f32[8, 32, 64]" = torch.ops.aten.view.default(expand_5, [8, 32, 64]);  expand_5 = None
        bmm_1: "f32[8, 32, 64]" = torch.ops.aten.bmm.default(view_5, view_6);  view_5 = view_6 = None
        view_7: "f32[2, 4, 32, 64]" = torch.ops.aten.view.default(bmm_1, [2, 4, 32, 64]);  bmm_1 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:42 in torch_attn_primitives, code: denominator = scores.sum(dim=-1, keepdim=True)
        sum_1: "f32[2, 4, 32, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True);  exp = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:77 in ring_send_recv, code: send_buffer.contiguous(),
        clone_2: "f32[2, 2, 32, 64]" = torch.ops.aten.clone.default(arg1_1, memory_format = torch.contiguous_format);  arg1_1 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:483 in all_to_all_single, code: tensor = torch.ops._c10d_functional.all_to_all_single(  # type: ignore[attr-defined]
        all_to_all_single: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.all_to_all_single.default(clone_2, [0, 0, 0, 2], [0, 2, 0, 0], '0');  clone_2 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:140 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single);  all_to_all_single = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:77 in ring_send_recv, code: send_buffer.contiguous(),
        clone_3: "f32[2, 2, 32, 64]" = torch.ops.aten.clone.default(arg2_1, memory_format = torch.contiguous_format);  arg2_1 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:483 in all_to_all_single, code: tensor = torch.ops._c10d_functional.all_to_all_single(  # type: ignore[attr-defined]
        all_to_all_single_1: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.all_to_all_single.default(clone_3, [0, 0, 0, 2], [0, 2, 0, 0], '0');  clone_3 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:140 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_1: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_1);  all_to_all_single_1 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:483 in all_to_all_single, code: tensor = torch.ops._c10d_functional.all_to_all_single(  # type: ignore[attr-defined]
        all_to_all_single_2: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.all_to_all_single.default(wait_tensor, [0, 0, 0, 2], [0, 2, 0, 0], '0');  wait_tensor = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:140 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_2: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_2);  all_to_all_single_2 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:483 in all_to_all_single, code: tensor = torch.ops._c10d_functional.all_to_all_single(  # type: ignore[attr-defined]
        all_to_all_single_3: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.all_to_all_single.default(wait_tensor_1, [0, 0, 0, 2], [0, 2, 0, 0], '0');  wait_tensor_1 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:140 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_3: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_3);  all_to_all_single_3 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:483 in all_to_all_single, code: tensor = torch.ops._c10d_functional.all_to_all_single(  # type: ignore[attr-defined]
        all_to_all_single_4: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.all_to_all_single.default(wait_tensor_2, [0, 0, 0, 2], [0, 2, 0, 0], '0');  wait_tensor_2 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:140 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_4: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_4);  all_to_all_single_4 = wait_tensor_4 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:483 in all_to_all_single, code: tensor = torch.ops._c10d_functional.all_to_all_single(  # type: ignore[attr-defined]
        all_to_all_single_5: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.all_to_all_single.default(wait_tensor_3, [0, 0, 0, 2], [0, 2, 0, 0], '0');  wait_tensor_3 = None
        
         # File: /gpfs/users/goon/.pyenv/versions/3.12.8/envs/torch-2.5.1/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:140 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_5: "f32[2, 2, 32, 64]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_5);  all_to_all_single_5 = wait_tensor_5 = None
        
         # File: /gpfs/users/goon/github/garrett361/learn_torch/learn_torch/distributed/cp/torch_cp_inference.py:124 in torch_ring_attn_prefill, code: return numerator / denominator
        div_1: "f32[2, 4, 32, 64]" = torch.ops.aten.div.Tensor(view_7, sum_1);  view_7 = sum_1 = None
        return (div_1,)
        