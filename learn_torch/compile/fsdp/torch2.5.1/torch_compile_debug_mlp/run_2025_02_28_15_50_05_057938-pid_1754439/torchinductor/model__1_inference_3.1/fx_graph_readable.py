class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[]", arg1_1: "bf16[1, 16]", arg2_1: "bf16[16, 16]", arg3_1: "bf16[16, 16]", arg4_1: "bf16[1, 16]", arg5_1: "bf16[1, 16]", arg6_1: "b8[1, 16]", arg7_1: "bf16[16, 16]", arg8_1: "bf16[128]", arg9_1: "bf16[128]", arg10_1: "bf16[128]"):
        # No stacktrace found for following nodes
        expand: "bf16[1, 16]" = torch.ops.aten.expand.default(arg0_1, [1, 16]);  arg0_1 = None
        clone: "bf16[1, 16]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        all_gather_copy_in = torch.ops.fsdp.all_gather_copy_in.default([arg8_1], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1));  arg8_1 = None
        getitem: "bf16[128]" = all_gather_copy_in[0]
        getitem_1: "bf16[256]" = all_gather_copy_in[1];  all_gather_copy_in = getitem_1 = None
        all_gather_into_tensor: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem, 2, '0');  getitem = None
        wait_tensor: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
        empty: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        view_1: "bf16[2, 128]" = torch.ops.aten.view.default(empty, [2, -1])
        view_2: "bf16[2, 128]" = torch.ops.aten.view.default(wait_tensor, [2, -1]);  wait_tensor = None
        auto_functionalized = torch.ops.higher_order.auto_functionalized(torch.ops.fsdp.split_with_sizes_copy.default, all_gather_output = view_2, all_gather_input_split_sizes = [128], dim = 1, out = [view_1]);  view_2 = view_1 = None
        getitem_3 = auto_functionalized[1];  auto_functionalized = None
        getitem_4: "bf16[2, 128]" = getitem_3[0];  getitem_3 = None
        view_3: "bf16[256]" = torch.ops.aten.view.default(getitem_4, [256]);  getitem_4 = None
        as_strided_1: "bf16[16, 16]" = torch.ops.aten.as_strided.default(view_3, [16, 16], [16, 1], 0);  view_3 = None
        set_ = torch.ops.fsdp.set_.default(arg3_1, as_strided_1);  as_strided_1 = set_ = None
        full_default: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False)
        where: "bf16[1, 16]" = torch.ops.aten.where.self(arg6_1, full_default, clone);  arg6_1 = full_default = clone = None
        permute: "bf16[16, 1]" = torch.ops.aten.permute.default(where, [1, 0])
        mm: "bf16[16, 16]" = torch.ops.aten.mm.default(permute, arg5_1);  permute = None
        permute_1: "bf16[16, 16]" = torch.ops.aten.permute.default(mm, [1, 0]);  mm = None
        permute_2: "bf16[16, 16]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        permute_3: "bf16[16, 16]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_1: "bf16[1, 16]" = torch.ops.aten.mm.default(where, permute_3);  where = permute_3 = None
        permute_4: "bf16[16, 16]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        all_gather_copy_in_1 = torch.ops.fsdp.all_gather_copy_in.default([arg9_1], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1));  arg9_1 = None
        getitem_5: "bf16[128]" = all_gather_copy_in_1[0]
        getitem_6: "bf16[256]" = all_gather_copy_in_1[1];  all_gather_copy_in_1 = getitem_6 = None
        all_gather_into_tensor_1: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_5, 2, '0');  getitem_5 = None
        wait_tensor_1: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_1);  all_gather_into_tensor_1 = None
        empty_1: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        view_6: "bf16[2, 128]" = torch.ops.aten.view.default(empty_1, [2, -1])
        view_7: "bf16[2, 128]" = torch.ops.aten.view.default(wait_tensor_1, [2, -1]);  wait_tensor_1 = None
        auto_functionalized_1 = torch.ops.higher_order.auto_functionalized(torch.ops.fsdp.split_with_sizes_copy.default, all_gather_output = view_7, all_gather_input_split_sizes = [128], dim = 1, out = [view_6]);  view_7 = view_6 = None
        getitem_8 = auto_functionalized_1[1];  auto_functionalized_1 = None
        getitem_9: "bf16[2, 128]" = getitem_8[0];  getitem_8 = None
        view_8: "bf16[256]" = torch.ops.aten.view.default(getitem_9, [256]);  getitem_9 = None
        as_strided_3: "bf16[16, 16]" = torch.ops.aten.as_strided.default(view_8, [16, 16], [16, 1], 0);  view_8 = None
        set__1 = torch.ops.fsdp.set_.default(arg2_1, as_strided_3);  as_strided_3 = set__1 = None
        le: "b8[1, 16]" = torch.ops.aten.le.Scalar(arg5_1, 0);  arg5_1 = None
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False)
        where_1: "bf16[1, 16]" = torch.ops.aten.where.self(le, full_default_1, mm_1);  le = full_default_1 = mm_1 = None
        permute_5: "bf16[16, 1]" = torch.ops.aten.permute.default(where_1, [1, 0])
        mm_2: "bf16[16, 16]" = torch.ops.aten.mm.default(permute_5, arg4_1);  permute_5 = None
        permute_6: "bf16[16, 16]" = torch.ops.aten.permute.default(mm_2, [1, 0]);  mm_2 = None
        permute_7: "bf16[16, 16]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        permute_8: "bf16[16, 16]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        mm_3: "bf16[1, 16]" = torch.ops.aten.mm.default(where_1, permute_8);  where_1 = permute_8 = None
        permute_9: "bf16[16, 16]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
        all_gather_copy_in_2 = torch.ops.fsdp.all_gather_copy_in.default([arg10_1], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1));  arg10_1 = None
        getitem_10: "bf16[128]" = all_gather_copy_in_2[0]
        getitem_11: "bf16[256]" = all_gather_copy_in_2[1];  all_gather_copy_in_2 = getitem_11 = None
        all_gather_into_tensor_2: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_10, 2, '0');  getitem_10 = None
        wait_tensor_2: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_2);  all_gather_into_tensor_2 = None
        empty_2: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        view_11: "bf16[2, 128]" = torch.ops.aten.view.default(empty_2, [2, -1])
        view_12: "bf16[2, 128]" = torch.ops.aten.view.default(wait_tensor_2, [2, -1]);  wait_tensor_2 = None
        auto_functionalized_2 = torch.ops.higher_order.auto_functionalized(torch.ops.fsdp.split_with_sizes_copy.default, all_gather_output = view_12, all_gather_input_split_sizes = [128], dim = 1, out = [view_11]);  view_12 = view_11 = None
        getitem_13 = auto_functionalized_2[1];  auto_functionalized_2 = None
        getitem_14: "bf16[2, 128]" = getitem_13[0];  getitem_13 = None
        view_13: "bf16[256]" = torch.ops.aten.view.default(getitem_14, [256]);  getitem_14 = None
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
        view_15: "bf16[2, 128]" = torch.ops.aten.view.default(empty_3, [2, -1]);  empty_3 = None
        auto_functionalized_3 = torch.ops.higher_order.auto_functionalized(torch.ops.fsdp.chunk_cat.default, tensors = [permute_12], dim = 0, num_chunks = 2, out = view_15);  permute_12 = view_15 = None
        getitem_16: "bf16[2, 128]" = auto_functionalized_3[1];  auto_functionalized_3 = None
        view_16: "bf16[256]" = torch.ops.aten.view.default(getitem_16, [256]);  getitem_16 = None
        empty_4: "bf16[128]" = torch.ops.aten.empty.memory_format([128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False);  empty_4 = None
        reduce_scatter_tensor: "bf16[128]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_16, 'avg', 2, '0');  view_16 = None
        wait_tensor_3: "bf16[128]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor);  reduce_scatter_tensor = None
        resize_storage_bytes__1 = torch.ops.inductor.resize_storage_bytes_.default(empty_1, 0);  empty_1 = resize_storage_bytes__1 = None
        empty_5: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        view_19: "bf16[2, 128]" = torch.ops.aten.view.default(empty_5, [2, -1]);  empty_5 = None
        auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops.fsdp.chunk_cat.default, tensors = [permute_9], dim = 0, num_chunks = 2, out = view_19);  permute_9 = view_19 = None
        getitem_18: "bf16[2, 128]" = auto_functionalized_4[1];  auto_functionalized_4 = None
        view_20: "bf16[256]" = torch.ops.aten.view.default(getitem_18, [256]);  getitem_18 = None
        empty_6: "bf16[128]" = torch.ops.aten.empty.memory_format([128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False);  empty_6 = None
        reduce_scatter_tensor_1: "bf16[128]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_20, 'avg', 2, '0');  view_20 = None
        wait_tensor_4: "bf16[128]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_1);  reduce_scatter_tensor_1 = None
        resize_storage_bytes__2 = torch.ops.inductor.resize_storage_bytes_.default(empty, 0);  empty = resize_storage_bytes__2 = None
        empty_7: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        view_23: "bf16[2, 128]" = torch.ops.aten.view.default(empty_7, [2, -1]);  empty_7 = None
        auto_functionalized_5 = torch.ops.higher_order.auto_functionalized(torch.ops.fsdp.chunk_cat.default, tensors = [permute_4], dim = 0, num_chunks = 2, out = view_23);  permute_4 = view_23 = None
        getitem_20: "bf16[2, 128]" = auto_functionalized_5[1];  auto_functionalized_5 = None
        view_24: "bf16[256]" = torch.ops.aten.view.default(getitem_20, [256]);  getitem_20 = None
        empty_8: "bf16[128]" = torch.ops.aten.empty.memory_format([128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False);  empty_8 = None
        reduce_scatter_tensor_2: "bf16[128]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_24, 'avg', 2, '0');  view_24 = None
        wait_tensor_5: "bf16[128]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_2);  reduce_scatter_tensor_2 = None
        as_strided_9: "bf16[8, 16]" = torch.ops.aten.as_strided.default(wait_tensor_5, [8, 16], [16, 1], 0);  wait_tensor_5 = None
        as_strided_10: "bf16[8, 16]" = torch.ops.aten.as_strided.default(wait_tensor_4, [8, 16], [16, 1], 0);  wait_tensor_4 = None
        as_strided_11: "bf16[8, 16]" = torch.ops.aten.as_strided.default(wait_tensor_3, [8, 16], [16, 1], 0);  wait_tensor_3 = None
        return (as_strided_9, as_strided_10, as_strided_11)
        