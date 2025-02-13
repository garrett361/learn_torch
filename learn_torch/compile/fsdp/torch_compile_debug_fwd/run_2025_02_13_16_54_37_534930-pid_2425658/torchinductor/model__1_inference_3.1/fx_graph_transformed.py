class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[]", arg1_1: "bf16[1, 16]", arg2_1: "bf16[16, 16]", arg3_1: "bf16[16, 16]", arg4_1: "bf16[1, 16]", arg5_1: "bf16[1, 16]", arg6_1: "b8[1, 16]", arg7_1: "bf16[16, 16]", arg8_1: "bf16[128]", arg9_1: "bf16[128]", arg10_1: "bf16[128]", arg11_1: "bf16[8, 16]", arg12_1: "bf16[8, 16]", arg13_1: "bf16[8, 16]"):
        # No stacktrace found for following nodes
        expand: "bf16[1, 16]" = torch.ops.aten.expand.default(arg0_1, [1, 16]);  arg0_1 = None
        clone: "bf16[1, 16]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        all_gather_copy_in_default_2 = torch.ops.fsdp.all_gather_copy_in.default([arg8_1], [128], 128, 2, 0, torch.bfloat16, device(type='cuda', index=0));  arg8_1 = None
        getitem_22: "bf16[128]" = all_gather_copy_in_default_2[0]
        getitem_23: "bf16[256]" = all_gather_copy_in_default_2[1];  all_gather_copy_in_default_2 = None
        all_gather_into_tensor_out_default_2: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor_out.default(getitem_22, 2, '0', out = getitem_23);  getitem_22 = getitem_23 = None
        wait_tensor: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_out_default_2);  all_gather_into_tensor_out_default_2 = None
        view_2: "bf16[2, 128]" = torch.ops.aten.reshape.default(wait_tensor, [2, -1]);  wait_tensor = None
        empty: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_default_5: "bf16[2, 128]" = torch.ops.aten.as_strided.default(empty, [2, 128], [128, 1], 0)
        split_with_sizes_copy_default_2 = torch.ops.fsdp.split_with_sizes_copy.default(view_2, [128], 1, out = [as_strided_default_5]);  view_2 = as_strided_default_5 = split_with_sizes_copy_default_2 = None
        as_strided_1: "bf16[16, 16]" = torch.ops.aten.as_strided.default(empty, [16, 16], [16, 1], 0);  empty = None
        full_default: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[1, 16]" = torch.ops.aten.where.self(arg6_1, full_default, clone);  arg6_1 = clone = None
        permute: "bf16[16, 1]" = torch.ops.aten.permute.default(where, [1, 0])
        mm: "bf16[16, 16]" = torch.ops.aten.mm.default(permute, arg5_1);  permute = None
        mm_1: "bf16[1, 16]" = torch.ops.aten.mm.default(where, as_strided_1);  where = as_strided_1 = None
        all_gather_copy_in_default_1 = torch.ops.fsdp.all_gather_copy_in.default([arg9_1], [128], 128, 2, 0, torch.bfloat16, device(type='cuda', index=0));  arg9_1 = None
        getitem_20: "bf16[128]" = all_gather_copy_in_default_1[0]
        getitem_21: "bf16[256]" = all_gather_copy_in_default_1[1];  all_gather_copy_in_default_1 = None
        all_gather_into_tensor_out_default_1: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor_out.default(getitem_20, 2, '0', out = getitem_21);  getitem_20 = getitem_21 = None
        wait_tensor_1: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_out_default_1);  all_gather_into_tensor_out_default_1 = None
        view_5: "bf16[2, 128]" = torch.ops.aten.reshape.default(wait_tensor_1, [2, -1]);  wait_tensor_1 = None
        empty_1: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_default_4: "bf16[2, 128]" = torch.ops.aten.as_strided.default(empty_1, [2, 128], [128, 1], 0)
        split_with_sizes_copy_default_1 = torch.ops.fsdp.split_with_sizes_copy.default(view_5, [128], 1, out = [as_strided_default_4]);  view_5 = as_strided_default_4 = split_with_sizes_copy_default_1 = None
        as_strided_3: "bf16[16, 16]" = torch.ops.aten.as_strided.default(empty_1, [16, 16], [16, 1], 0);  empty_1 = None
        le: "b8[1, 16]" = torch.ops.aten.le.Scalar(arg5_1, 0);  arg5_1 = None
        where_1: "bf16[1, 16]" = torch.ops.aten.where.self(le, full_default, mm_1);  le = mm_1 = None
        permute_3: "bf16[16, 1]" = torch.ops.aten.permute.default(where_1, [1, 0])
        mm_2: "bf16[16, 16]" = torch.ops.aten.mm.default(permute_3, arg4_1);  permute_3 = None
        mm_3: "bf16[1, 16]" = torch.ops.aten.mm.default(where_1, as_strided_3);  where_1 = as_strided_3 = None
        all_gather_copy_in_default = torch.ops.fsdp.all_gather_copy_in.default([arg10_1], [128], 128, 2, 0, torch.bfloat16, device(type='cuda', index=0));  arg10_1 = None
        getitem_18: "bf16[128]" = all_gather_copy_in_default[0]
        getitem_19: "bf16[256]" = all_gather_copy_in_default[1];  all_gather_copy_in_default = None
        all_gather_into_tensor_out_default: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor_out.default(getitem_18, 2, '0', out = getitem_19);  getitem_18 = getitem_19 = None
        wait_tensor_2: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_out_default);  all_gather_into_tensor_out_default = None
        view_8: "bf16[2, 128]" = torch.ops.aten.reshape.default(wait_tensor_2, [2, -1]);  wait_tensor_2 = None
        empty_2: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_default_3: "bf16[2, 128]" = torch.ops.aten.as_strided.default(empty_2, [2, 128], [128, 1], 0)
        split_with_sizes_copy_default = torch.ops.fsdp.split_with_sizes_copy.default(view_8, [128], 1, out = [as_strided_default_3]);  view_8 = as_strided_default_3 = split_with_sizes_copy_default = None
        as_strided_5: "bf16[16, 16]" = torch.ops.aten.as_strided.default(empty_2, [16, 16], [16, 1], 0);  empty_2 = as_strided_5 = None
        le_1: "b8[1, 16]" = torch.ops.aten.le.Scalar(arg4_1, 0);  arg4_1 = None
        where_2: "bf16[1, 16]" = torch.ops.aten.where.self(le_1, full_default, mm_3);  le_1 = full_default = mm_3 = None
        permute_6: "bf16[16, 1]" = torch.ops.aten.permute.default(where_2, [1, 0]);  where_2 = None
        mm_4: "bf16[16, 16]" = torch.ops.aten.mm.default(permute_6, arg1_1);  permute_6 = arg1_1 = None
        empty_3: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_default_2: "bf16[2, 128]" = torch.ops.aten.as_strided.default(empty_3, [2, 128], [128, 1], 0)
        chunk_cat_default_2 = torch.ops.fsdp.chunk_cat.default([mm_4], 0, 2, out = as_strided_default_2);  mm_4 = as_strided_default_2 = chunk_cat_default_2 = None
        empty_4: "bf16[128]" = torch.ops.aten.empty.memory_format([128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  empty_4 = None
        reduce_scatter_tensor: "bf16[128]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(empty_3, 'avg', 2, '0');  empty_3 = None
        wait_tensor_3: "bf16[128]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor);  reduce_scatter_tensor = None
        as_strided_7: "bf16[8, 16]" = torch.ops.aten.as_strided.default(wait_tensor_3, [8, 16], [16, 1], 0);  wait_tensor_3 = None
        add: "bf16[8, 16]" = torch.ops.aten.add.Tensor(arg11_1, as_strided_7);  as_strided_7 = None
        empty_5: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_default_1: "bf16[2, 128]" = torch.ops.aten.as_strided.default(empty_5, [2, 128], [128, 1], 0)
        chunk_cat_default_1 = torch.ops.fsdp.chunk_cat.default([mm_2], 0, 2, out = as_strided_default_1);  mm_2 = as_strided_default_1 = chunk_cat_default_1 = None
        empty_6: "bf16[128]" = torch.ops.aten.empty.memory_format([128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  empty_6 = None
        reduce_scatter_tensor_1: "bf16[128]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(empty_5, 'avg', 2, '0');  empty_5 = None
        wait_tensor_4: "bf16[128]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_1);  reduce_scatter_tensor_1 = None
        as_strided_9: "bf16[8, 16]" = torch.ops.aten.as_strided.default(wait_tensor_4, [8, 16], [16, 1], 0);  wait_tensor_4 = None
        add_1: "bf16[8, 16]" = torch.ops.aten.add.Tensor(arg12_1, as_strided_9);  as_strided_9 = None
        empty_7: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_default: "bf16[2, 128]" = torch.ops.aten.as_strided.default(empty_7, [2, 128], [128, 1], 0)
        chunk_cat_default = torch.ops.fsdp.chunk_cat.default([mm], 0, 2, out = as_strided_default);  mm = as_strided_default = chunk_cat_default = None
        empty_8: "bf16[128]" = torch.ops.aten.empty.memory_format([128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  empty_8 = None
        reduce_scatter_tensor_2: "bf16[128]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(empty_7, 'avg', 2, '0');  empty_7 = None
        wait_tensor_5: "bf16[128]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_2);  reduce_scatter_tensor_2 = None
        as_strided_11: "bf16[8, 16]" = torch.ops.aten.as_strided.default(wait_tensor_5, [8, 16], [16, 1], 0);  wait_tensor_5 = None
        add_2: "bf16[8, 16]" = torch.ops.aten.add.Tensor(arg13_1, as_strided_11);  as_strided_11 = None
        copy__3: "bf16[8, 16]" = torch.ops.aten.copy_.default(arg11_1, add);  arg11_1 = add = None
        copy__4: "bf16[8, 16]" = torch.ops.aten.copy_.default(arg12_1, add_1);  arg12_1 = add_1 = None
        copy__5: "bf16[8, 16]" = torch.ops.aten.copy_.default(arg13_1, add_2);  arg13_1 = add_2 = None
        return (copy__3, copy__4, copy__5)
        