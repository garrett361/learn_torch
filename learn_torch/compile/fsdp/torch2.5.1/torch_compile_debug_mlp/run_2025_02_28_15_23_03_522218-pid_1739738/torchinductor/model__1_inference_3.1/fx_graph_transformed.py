class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[]", arg1_1: "bf16[1, 16]", arg2_1: "bf16[16, 16]", arg3_1: "bf16[16, 16]", arg4_1: "bf16[1, 16]", arg5_1: "bf16[1, 16]", arg6_1: "b8[1, 16]", arg7_1: "bf16[16, 16]", arg8_1: "bf16[128]", arg9_1: "bf16[128]", arg10_1: "bf16[128]"):
        # No stacktrace found for following nodes
        expand: "bf16[1, 16]" = torch.ops.aten.expand.default(arg0_1, [1, 16]);  arg0_1 = None
        clone: "bf16[1, 16]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        empty: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        view_1: "bf16[2, 128]" = torch.ops.aten.reshape.default(empty, [2, -1])
        all_gather_copy_in_default_2 = torch.ops.fsdp.all_gather_copy_in.default([arg8_1], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1));  arg8_1 = None
        getitem_25: "bf16[128]" = all_gather_copy_in_default_2[0]
        getitem_26: "bf16[256]" = all_gather_copy_in_default_2[1];  all_gather_copy_in_default_2 = None
        all_gather_into_tensor_out_default_2: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor_out.default(getitem_25, 2, '0', out = getitem_26);  getitem_25 = getitem_26 = None
        wait_tensor: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_out_default_2);  all_gather_into_tensor_out_default_2 = None
        view_2: "bf16[2, 128]" = torch.ops.aten.reshape.default(wait_tensor, [2, -1]);  wait_tensor = None
        as_strided_default_4: "bf16[256]" = torch.ops.aten.as_strided.default(view_1, [256], [1], 0);  view_1 = None
        clone_default_2: "bf16[256]" = torch.ops.aten.clone.default(as_strided_default_4);  as_strided_default_4 = None
        as_strided_default_5: "bf16[2, 128]" = torch.ops.aten.as_strided.default(clone_default_2, [2, 128], [128, 1], 0);  clone_default_2 = None
        split_with_sizes_copy_default_2 = torch.ops.fsdp.split_with_sizes_copy.default(view_2, [128], 1, out = [as_strided_default_5]);  view_2 = split_with_sizes_copy_default_2 = None
        view_3: "bf16[256]" = torch.ops.aten.reshape.default(as_strided_default_5, [256]);  as_strided_default_5 = None
        as_strided_1: "bf16[16, 16]" = torch.ops.aten.as_strided.default(view_3, [16, 16], [16, 1], 0);  view_3 = None
        set_ = torch.ops.fsdp.set_.default(arg3_1, as_strided_1);  as_strided_1 = set_ = None
        full_default: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False)
        where: "bf16[1, 16]" = torch.ops.aten.where.self(arg6_1, full_default, clone);  arg6_1 = full_default = clone = None
        permute_2: "bf16[16, 16]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        permute_3: "bf16[16, 16]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_1: "bf16[1, 16]" = torch.ops.aten.mm.default(where, permute_3);  permute_3 = None
        permute: "bf16[16, 1]" = torch.ops.aten.permute.default(where, [1, 0]);  where = None
        mm: "bf16[16, 16]" = torch.ops.aten.mm.default(permute, arg5_1);  permute = None
        permute_1: "bf16[16, 16]" = torch.ops.aten.permute.default(mm, [1, 0]);  mm = None
        permute_4: "bf16[16, 16]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        empty_1: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        view_6: "bf16[2, 128]" = torch.ops.aten.reshape.default(empty_1, [2, -1])
        all_gather_copy_in_default_1 = torch.ops.fsdp.all_gather_copy_in.default([arg9_1], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1));  arg9_1 = None
        getitem_23: "bf16[128]" = all_gather_copy_in_default_1[0]
        getitem_24: "bf16[256]" = all_gather_copy_in_default_1[1];  all_gather_copy_in_default_1 = None
        all_gather_into_tensor_out_default_1: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor_out.default(getitem_23, 2, '0', out = getitem_24);  getitem_23 = getitem_24 = None
        wait_tensor_1: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_out_default_1);  all_gather_into_tensor_out_default_1 = None
        view_7: "bf16[2, 128]" = torch.ops.aten.reshape.default(wait_tensor_1, [2, -1]);  wait_tensor_1 = None
        as_strided_default_2: "bf16[256]" = torch.ops.aten.as_strided.default(view_6, [256], [1], 0);  view_6 = None
        clone_default_1: "bf16[256]" = torch.ops.aten.clone.default(as_strided_default_2);  as_strided_default_2 = None
        as_strided_default_3: "bf16[2, 128]" = torch.ops.aten.as_strided.default(clone_default_1, [2, 128], [128, 1], 0);  clone_default_1 = None
        split_with_sizes_copy_default_1 = torch.ops.fsdp.split_with_sizes_copy.default(view_7, [128], 1, out = [as_strided_default_3]);  view_7 = split_with_sizes_copy_default_1 = None
        view_8: "bf16[256]" = torch.ops.aten.reshape.default(as_strided_default_3, [256]);  as_strided_default_3 = None
        as_strided_3: "bf16[16, 16]" = torch.ops.aten.as_strided.default(view_8, [16, 16], [16, 1], 0);  view_8 = None
        set__1 = torch.ops.fsdp.set_.default(arg2_1, as_strided_3);  as_strided_3 = set__1 = None
        le: "b8[1, 16]" = torch.ops.aten.le.Scalar(arg5_1, 0);  arg5_1 = None
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False)
        where_1: "bf16[1, 16]" = torch.ops.aten.where.self(le, full_default_1, mm_1);  le = full_default_1 = mm_1 = None
        permute_7: "bf16[16, 16]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        permute_8: "bf16[16, 16]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        mm_3: "bf16[1, 16]" = torch.ops.aten.mm.default(where_1, permute_8);  permute_8 = None
        permute_5: "bf16[16, 1]" = torch.ops.aten.permute.default(where_1, [1, 0]);  where_1 = None
        mm_2: "bf16[16, 16]" = torch.ops.aten.mm.default(permute_5, arg4_1);  permute_5 = None
        permute_6: "bf16[16, 16]" = torch.ops.aten.permute.default(mm_2, [1, 0]);  mm_2 = None
        permute_9: "bf16[16, 16]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
        empty_2: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        view_11: "bf16[2, 128]" = torch.ops.aten.reshape.default(empty_2, [2, -1])
        all_gather_copy_in_default = torch.ops.fsdp.all_gather_copy_in.default([arg10_1], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1));  arg10_1 = None
        getitem_21: "bf16[128]" = all_gather_copy_in_default[0]
        getitem_22: "bf16[256]" = all_gather_copy_in_default[1];  all_gather_copy_in_default = None
        all_gather_into_tensor_out_default: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor_out.default(getitem_21, 2, '0', out = getitem_22);  getitem_21 = getitem_22 = None
        wait_tensor_2: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_out_default);  all_gather_into_tensor_out_default = None
        view_12: "bf16[2, 128]" = torch.ops.aten.reshape.default(wait_tensor_2, [2, -1]);  wait_tensor_2 = None
        as_strided_default: "bf16[256]" = torch.ops.aten.as_strided.default(view_11, [256], [1], 0);  view_11 = None
        clone_default: "bf16[256]" = torch.ops.aten.clone.default(as_strided_default);  as_strided_default = None
        as_strided_default_1: "bf16[2, 128]" = torch.ops.aten.as_strided.default(clone_default, [2, 128], [128, 1], 0);  clone_default = None
        split_with_sizes_copy_default = torch.ops.fsdp.split_with_sizes_copy.default(view_12, [128], 1, out = [as_strided_default_1]);  view_12 = split_with_sizes_copy_default = None
        view_13: "bf16[256]" = torch.ops.aten.reshape.default(as_strided_default_1, [256]);  as_strided_default_1 = None
        as_strided_5: "bf16[16, 16]" = torch.ops.aten.as_strided.default(view_13, [16, 16], [16, 1], 0);  view_13 = None
        set__2 = torch.ops.fsdp.set_.default(arg7_1, as_strided_5);  arg7_1 = as_strided_5 = set__2 = None
        le_1: "b8[1, 16]" = torch.ops.aten.le.Scalar(arg4_1, 0);  arg4_1 = None
        full_default_2: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False)
        where_2: "bf16[1, 16]" = torch.ops.aten.where.self(le_1, full_default_2, mm_3);  le_1 = full_default_2 = mm_3 = None
        permute_10: "bf16[16, 1]" = torch.ops.aten.permute.default(where_2, [1, 0]);  where_2 = None
        mm_4: "bf16[16, 16]" = torch.ops.aten.mm.default(permute_10, arg1_1);  permute_10 = arg1_1 = None
        permute_11: "bf16[16, 16]" = torch.ops.aten.permute.default(mm_4, [1, 0]);  mm_4 = None
        permute_12: "bf16[16, 16]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(empty_2, 0);  empty_2 = resize_storage_bytes_ = None
        empty_3: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        view_15: "bf16[2, 128]" = torch.ops.aten.reshape.default(empty_3, [2, -1]);  empty_3 = None
        chunk_cat_default_2 = torch.ops.fsdp.chunk_cat.default([permute_12], 0, 2, out = view_15);  permute_12 = chunk_cat_default_2 = None
        empty_4: "bf16[128]" = torch.ops.aten.empty.memory_format([128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False);  empty_4 = None
        view_16: "bf16[256]" = torch.ops.aten.reshape.default(view_15, [256]);  view_15 = None
        reduce_scatter_tensor: "bf16[128]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_16, 'avg', 2, '0');  view_16 = None
        wait_tensor_3: "bf16[128]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor);  reduce_scatter_tensor = None
        resize_storage_bytes__1 = torch.ops.inductor.resize_storage_bytes_.default(empty_1, 0);  empty_1 = resize_storage_bytes__1 = None
        empty_5: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        view_19: "bf16[2, 128]" = torch.ops.aten.reshape.default(empty_5, [2, -1]);  empty_5 = None
        chunk_cat_default_1 = torch.ops.fsdp.chunk_cat.default([permute_9], 0, 2, out = view_19);  permute_9 = chunk_cat_default_1 = None
        empty_6: "bf16[128]" = torch.ops.aten.empty.memory_format([128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False);  empty_6 = None
        view_20: "bf16[256]" = torch.ops.aten.reshape.default(view_19, [256]);  view_19 = None
        reduce_scatter_tensor_1: "bf16[128]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_20, 'avg', 2, '0');  view_20 = None
        wait_tensor_4: "bf16[128]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_1);  reduce_scatter_tensor_1 = None
        resize_storage_bytes__2 = torch.ops.inductor.resize_storage_bytes_.default(empty, 0);  empty = resize_storage_bytes__2 = None
        empty_7: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        view_23: "bf16[2, 128]" = torch.ops.aten.reshape.default(empty_7, [2, -1]);  empty_7 = None
        chunk_cat_default = torch.ops.fsdp.chunk_cat.default([permute_4], 0, 2, out = view_23);  permute_4 = chunk_cat_default = None
        empty_8: "bf16[128]" = torch.ops.aten.empty.memory_format([128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False);  empty_8 = None
        view_24: "bf16[256]" = torch.ops.aten.reshape.default(view_23, [256]);  view_23 = None
        reduce_scatter_tensor_2: "bf16[128]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_24, 'avg', 2, '0');  view_24 = None
        wait_tensor_5: "bf16[128]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_2);  reduce_scatter_tensor_2 = None
        as_strided_9: "bf16[8, 16]" = torch.ops.aten.as_strided.default(wait_tensor_5, [8, 16], [16, 1], 0);  wait_tensor_5 = None
        as_strided_10: "bf16[8, 16]" = torch.ops.aten.as_strided.default(wait_tensor_4, [8, 16], [16, 1], 0);  wait_tensor_4 = None
        as_strided_11: "bf16[8, 16]" = torch.ops.aten.as_strided.default(wait_tensor_3, [8, 16], [16, 1], 0);  wait_tensor_3 = None
        return (as_strided_9, as_strided_10, as_strided_11)
        