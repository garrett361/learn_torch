class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[]", arg1_1: "bf16[1, 16]", arg2_1: "bf16[16, 16]", arg3_1: "bf16[16, 16]", arg4_1: "bf16[1, 16]", arg5_1: "bf16[1, 16]", arg6_1: "b8[1, 16]", arg7_1: "bf16[16, 16]", arg8_1: "bf16[128]", arg9_1: "bf16[128]", arg10_1: "bf16[128]", arg11_1: "bf16[8, 16]", arg12_1: "bf16[8, 16]", arg13_1: "bf16[8, 16]"):
        # No stacktrace found for following nodes
        expand: "bf16[1, 16]" = torch.ops.aten.expand.default(arg0_1, [1, 16]);  arg0_1 = None
        clone: "bf16[1, 16]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        all_gather_copy_in = torch.ops.fsdp.all_gather_copy_in.default([arg8_1], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1));  arg8_1 = None
        getitem: "bf16[128]" = all_gather_copy_in[0]
        getitem_1: "bf16[256]" = all_gather_copy_in[1];  all_gather_copy_in = getitem_1 = None
        all_gather_into_tensor: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem, 2, '0');  getitem = None
        wait_tensor: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
        empty: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        view_2: "bf16[2, 128]" = torch.ops.aten.view.default(wait_tensor, [2, -1]);  wait_tensor = None
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.fsdp.split_with_sizes_copy.default, all_gather_output = view_2, all_gather_input_split_sizes = [128], dim = 1, _out_length = 1, _out_0_base_index = 0, _out_0_size = (2, 128), _out_0_stride = (128, 1), _out_0_storage_offset = 0, _all_bases = [empty]);  view_2 = empty = None
        getitem_3: "bf16[256]" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(arg3_1, 512);  resize_storage_bytes_ = None
        as_strided_1: "bf16[16, 16]" = torch.ops.aten.as_strided.default(getitem_3, [16, 16], [16, 1], 0);  getitem_3 = None
        copy_ = torch.ops.fsdp.copy_.default(arg3_1, as_strided_1);  as_strided_1 = copy_ = None
        full_default: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False)
        where: "bf16[1, 16]" = torch.ops.aten.where.self(arg6_1, full_default, clone);  arg6_1 = clone = None
        permute: "bf16[16, 1]" = torch.ops.aten.permute.default(where, [1, 0])
        mm: "bf16[16, 16]" = torch.ops.aten.mm.default(permute, arg5_1);  permute = None
        mm_1: "bf16[1, 16]" = torch.ops.aten.mm.default(where, arg3_1);  where = None
        all_gather_copy_in_1 = torch.ops.fsdp.all_gather_copy_in.default([arg9_1], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1));  arg9_1 = None
        getitem_4: "bf16[128]" = all_gather_copy_in_1[0]
        getitem_5: "bf16[256]" = all_gather_copy_in_1[1];  all_gather_copy_in_1 = getitem_5 = None
        all_gather_into_tensor_1: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_4, 2, '0');  getitem_4 = None
        wait_tensor_1: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_1);  all_gather_into_tensor_1 = None
        empty_1: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        view_5: "bf16[2, 128]" = torch.ops.aten.view.default(wait_tensor_1, [2, -1]);  wait_tensor_1 = None
        auto_functionalized_v2_1 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.fsdp.split_with_sizes_copy.default, all_gather_output = view_5, all_gather_input_split_sizes = [128], dim = 1, _out_length = 1, _out_0_base_index = 0, _out_0_size = (2, 128), _out_0_stride = (128, 1), _out_0_storage_offset = 0, _all_bases = [empty_1]);  view_5 = empty_1 = None
        getitem_7: "bf16[256]" = auto_functionalized_v2_1[1];  auto_functionalized_v2_1 = None
        resize_storage_bytes__1 = torch.ops.inductor.resize_storage_bytes_.default(arg2_1, 512);  resize_storage_bytes__1 = None
        as_strided_3: "bf16[16, 16]" = torch.ops.aten.as_strided.default(getitem_7, [16, 16], [16, 1], 0);  getitem_7 = None
        copy__1 = torch.ops.fsdp.copy_.default(arg2_1, as_strided_3);  as_strided_3 = copy__1 = None
        le: "b8[1, 16]" = torch.ops.aten.le.Scalar(arg5_1, 0);  arg5_1 = None
        where_1: "bf16[1, 16]" = torch.ops.aten.where.self(le, full_default, mm_1);  le = mm_1 = None
        permute_3: "bf16[16, 1]" = torch.ops.aten.permute.default(where_1, [1, 0])
        mm_2: "bf16[16, 16]" = torch.ops.aten.mm.default(permute_3, arg4_1);  permute_3 = None
        mm_3: "bf16[1, 16]" = torch.ops.aten.mm.default(where_1, arg2_1);  where_1 = None
        all_gather_copy_in_2 = torch.ops.fsdp.all_gather_copy_in.default([arg10_1], [128], 128, 2, 1, torch.bfloat16, device(type='cuda', index=1));  arg10_1 = None
        getitem_8: "bf16[128]" = all_gather_copy_in_2[0]
        getitem_9: "bf16[256]" = all_gather_copy_in_2[1];  all_gather_copy_in_2 = getitem_9 = None
        all_gather_into_tensor_2: "bf16[256]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_8, 2, '0');  getitem_8 = None
        wait_tensor_2: "bf16[256]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_2);  all_gather_into_tensor_2 = None
        empty_2: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        view_8: "bf16[2, 128]" = torch.ops.aten.view.default(wait_tensor_2, [2, -1]);  wait_tensor_2 = None
        auto_functionalized_v2_2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.fsdp.split_with_sizes_copy.default, all_gather_output = view_8, all_gather_input_split_sizes = [128], dim = 1, _out_length = 1, _out_0_base_index = 0, _out_0_size = (2, 128), _out_0_stride = (128, 1), _out_0_storage_offset = 0, _all_bases = [empty_2]);  view_8 = empty_2 = None
        getitem_11: "bf16[256]" = auto_functionalized_v2_2[1];  auto_functionalized_v2_2 = None
        resize_storage_bytes__2 = torch.ops.inductor.resize_storage_bytes_.default(arg7_1, 512);  resize_storage_bytes__2 = None
        as_strided_5: "bf16[16, 16]" = torch.ops.aten.as_strided.default(getitem_11, [16, 16], [16, 1], 0);  getitem_11 = None
        copy__2 = torch.ops.fsdp.copy_.default(arg7_1, as_strided_5);  as_strided_5 = copy__2 = None
        le_1: "b8[1, 16]" = torch.ops.aten.le.Scalar(arg4_1, 0);  arg4_1 = None
        where_2: "bf16[1, 16]" = torch.ops.aten.where.self(le_1, full_default, mm_3);  le_1 = full_default = mm_3 = None
        permute_6: "bf16[16, 1]" = torch.ops.aten.permute.default(where_2, [1, 0]);  where_2 = None
        mm_4: "bf16[16, 16]" = torch.ops.aten.mm.default(permute_6, arg1_1);  permute_6 = arg1_1 = None
        resize_storage_bytes__3 = torch.ops.inductor.resize_storage_bytes_.default(arg7_1, 0);  arg7_1 = resize_storage_bytes__3 = None
        empty_3: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        auto_functionalized_v2_3 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.fsdp.chunk_cat.default, tensors = [mm_4], dim = 0, num_chunks = 2, _out_base_index = 0, _out_size = (2, 128), _out_stride = (128, 1), _out_storage_offset = 0, _all_bases = [empty_3]);  mm_4 = empty_3 = None
        getitem_13: "bf16[256]" = auto_functionalized_v2_3[1];  auto_functionalized_v2_3 = None
        empty_4: "bf16[128]" = torch.ops.aten.empty.memory_format([128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False);  empty_4 = None
        reduce_scatter_tensor: "bf16[128]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(getitem_13, 'avg', 2, '0');  getitem_13 = None
        wait_tensor_3: "bf16[128]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor);  reduce_scatter_tensor = None
        as_strided_7: "bf16[8, 16]" = torch.ops.aten.as_strided.default(wait_tensor_3, [8, 16], [16, 1], 0);  wait_tensor_3 = None
        add: "bf16[8, 16]" = torch.ops.aten.add.Tensor(arg11_1, as_strided_7);  as_strided_7 = None
        resize_storage_bytes__4 = torch.ops.inductor.resize_storage_bytes_.default(arg2_1, 0);  arg2_1 = resize_storage_bytes__4 = None
        empty_5: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        auto_functionalized_v2_4 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.fsdp.chunk_cat.default, tensors = [mm_2], dim = 0, num_chunks = 2, _out_base_index = 0, _out_size = (2, 128), _out_stride = (128, 1), _out_storage_offset = 0, _all_bases = [empty_5]);  mm_2 = empty_5 = None
        getitem_15: "bf16[256]" = auto_functionalized_v2_4[1];  auto_functionalized_v2_4 = None
        empty_6: "bf16[128]" = torch.ops.aten.empty.memory_format([128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False);  empty_6 = None
        reduce_scatter_tensor_1: "bf16[128]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(getitem_15, 'avg', 2, '0');  getitem_15 = None
        wait_tensor_4: "bf16[128]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_1);  reduce_scatter_tensor_1 = None
        as_strided_9: "bf16[8, 16]" = torch.ops.aten.as_strided.default(wait_tensor_4, [8, 16], [16, 1], 0);  wait_tensor_4 = None
        add_1: "bf16[8, 16]" = torch.ops.aten.add.Tensor(arg12_1, as_strided_9);  as_strided_9 = None
        resize_storage_bytes__5 = torch.ops.inductor.resize_storage_bytes_.default(arg3_1, 0);  arg3_1 = resize_storage_bytes__5 = None
        empty_7: "bf16[256]" = torch.ops.aten.empty.memory_format([256], dtype = torch.bfloat16, device = device(type='cuda', index=1), pin_memory = False)
        auto_functionalized_v2_5 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.fsdp.chunk_cat.default, tensors = [mm], dim = 0, num_chunks = 2, _out_base_index = 0, _out_size = (2, 128), _out_stride = (128, 1), _out_storage_offset = 0, _all_bases = [empty_7]);  mm = empty_7 = None
        getitem_17: "bf16[256]" = auto_functionalized_v2_5[1];  auto_functionalized_v2_5 = None
        empty_8: "bf16[128]" = torch.ops.aten.empty.memory_format([128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False);  empty_8 = None
        reduce_scatter_tensor_2: "bf16[128]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(getitem_17, 'avg', 2, '0');  getitem_17 = None
        wait_tensor_5: "bf16[128]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_2);  reduce_scatter_tensor_2 = None
        as_strided_11: "bf16[8, 16]" = torch.ops.aten.as_strided.default(wait_tensor_5, [8, 16], [16, 1], 0);  wait_tensor_5 = None
        add_2: "bf16[8, 16]" = torch.ops.aten.add.Tensor(arg13_1, as_strided_11);  as_strided_11 = None
        copy__3: "bf16[8, 16]" = torch.ops.aten.copy_.default(arg11_1, add);  arg11_1 = add = None
        copy__4: "bf16[8, 16]" = torch.ops.aten.copy_.default(arg12_1, add_1);  arg12_1 = add_1 = None
        copy__5: "bf16[8, 16]" = torch.ops.aten.copy_.default(arg13_1, add_2);  arg13_1 = add_2 = None
        return (copy__3, copy__4, copy__5)
        