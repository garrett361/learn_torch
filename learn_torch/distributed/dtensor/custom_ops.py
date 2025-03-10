from copy import deepcopy

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor, ones
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSchema, OutputSharding
from torch.distributed.tensor._ops.utils import register_prop_rule

from dtest import DTest


# Op def and registration
@triton.jit
def square_kernel(
    x_ptr,
    d_model,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(axis=0)
    pid_d_model = tl.program_id(axis=1)

    d_model_idx = pid_d_model * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = d_model_idx < d_model
    offsets = d_model * pid_batch + d_model_idx
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * x
    tl.store(output_ptr + offsets, output, mask=mask)


# NOTE: Can also use torch.library.triton_op if we want torch to further trace into and optimize the
# kernel, but this should not be used if we want DTensor/general tensor subclass support. custom_op
# treats the wrapped op opaquely, as desired for DTensor.


# https://pytorch.org/docs/stable/library.html#torch.library.triton_op
@torch.library.custom_op("learn_torch::square", mutates_args=(), device_types=("cuda",))
def triton_square(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    assert x.ndim == 2
    output = torch.empty_like(x)

    grid = lambda meta: (
        x.shape[0],
        triton.cdiv(x.shape[-1], meta["BLOCK_SIZE"]),
    )
    square_kernel[grid](x, x.shape[-1], output, BLOCK_SIZE=1024)
    return output


@triton.jit
def mul_kernel(
    x_ptr,
    d_model,
    factor,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(axis=0)
    pid_d_model = tl.program_id(axis=1)

    d_model_idx = pid_d_model * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = d_model_idx < d_model
    offsets = d_model * pid_batch + d_model_idx
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * factor
    tl.store(output_ptr + offsets, output, mask=mask)


# Also need to register triton_mul even though it will never be explicitly user-called (only called
# in backward).
@torch.library.custom_op("learn_torch::mul", mutates_args=(), device_types=("cuda",))
def triton_mul(x: torch.Tensor, factor: float) -> torch.Tensor:
    x = x.contiguous()
    assert x.ndim == 2
    output = torch.empty_like(x)

    grid = lambda meta: (
        x.shape[0],
        triton.cdiv(x.shape[-1], meta["BLOCK_SIZE"]),
    )
    mul_kernel[grid](x, x.shape[-1], factor, output, BLOCK_SIZE=1024)
    return output


# Shape registration
@triton_square.register_fake
def _(x: torch.Tensor):
    return torch.empty_like(x)


@triton_mul.register_fake
def _(x: torch.Tensor, factor: float):
    return torch.empty_like(x)


# Autograd
def setup_context_triton_square(ctx, inputs, output):
    ctx.inputs = inputs[0]


def backward_triton_square(ctx, grad):
    return grad * triton_mul(ctx.inputs, 2)


triton_square.register_autograd(backward_triton_square, setup_context=setup_context_triton_square)


class Test:
    def test_kernel_fwd(self) -> None:
        x = torch.randn(3, 2345, device="cuda")
        x_square = triton_square(x)
        torch.testing.assert_close(x_square, x.pow(2))

    def test_kernel_bwd(self) -> None:
        x = torch.randn(3, 2345, device="cuda", requires_grad=True)
        x_copy = deepcopy(x)
        x.pow(2).sum().backward()
        triton_square(x_copy).sum().backward()
        torch.testing.assert_close(x.grad, x_copy.grad)

    def test_opcheck(self) -> None:
        inputs = (
            (torch.randn(1, 1024, device="cuda"),),
            (torch.randn(1, 1024, device="cuda", requires_grad=True),),
            (torch.randn(3, 2345, device="cuda"),),
            (torch.randn(3, 2345, device="cuda", requires_grad=True),),
        )
        for t in inputs:
            torch.library.opcheck(triton_square, t)


# Start DTensor-specific code and tests


@register_prop_rule(torch.ops.learn_torch.square.default)
def _(op_schema: OpSchema) -> OutputSharding:
    (input_spec,) = op_schema.args_schema

    assert isinstance(input_spec, DTensorSpec)
    assert input_spec.tensor_meta is not None

    tensor_meta = TensorMeta(
        torch.Size(input_spec.tensor_meta.shape),
        input_spec.tensor_meta.stride,
        input_spec.tensor_meta.dtype,
    )
    return OutputSharding(
        DTensorSpec.from_dim_map(
            input_spec.mesh,
            input_spec.dim_map,
            input_spec.sums,
            tensor_meta=tensor_meta,
        )
    )


@register_prop_rule(torch.ops.learn_torch.mul.default)
def _(op_schema: OpSchema) -> OutputSharding:
    (input_spec, _) = op_schema.args_schema

    assert isinstance(input_spec, DTensorSpec)
    assert input_spec.tensor_meta is not None

    tensor_meta = TensorMeta(
        torch.Size(input_spec.tensor_meta.shape),
        input_spec.tensor_meta.stride,
        input_spec.tensor_meta.dtype,
    )
    return OutputSharding(
        DTensorSpec.from_dim_map(
            input_spec.mesh,
            input_spec.dim_map,
            input_spec.sums,
            tensor_meta=tensor_meta,
        )
    )


@torch.library.register_torch_dispatch("learn_torch::mul", DTensor)
def _(cls, op_call, type, args, kwargs) -> DTensor:
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)

    # sharding propagation
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"

    # local propagation
    (inputs, factor) = args
    local_results = triton_mul(inputs.to_local(), factor)

    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


class TestDTensor(DTest):
    def test_ones_registration(self) -> None:
        """
        Dumb test where we overwrite learn_torch::square's DTensor dispatch to just act as producing
        ones instead of actually squaring the tensor when acting on DTensors.
        """

        @torch.library.register_torch_dispatch("learn_torch::square", DTensor)
        def _(cls, func, type, args, kwargs) -> DTensor:
            (inputs,) = args
            return ones(
                *inputs.shape,
                dtype=inputs.dtype,
                requires_grad=inputs.requires_grad,
                device_mesh=inputs.device_mesh,
                placements=inputs.placements,
            )

        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh(device_type="cuda", mesh_shape=(self.world_size,))
        x = torch.randn(3, 2345, device=self.device)
        x_dt = DTensor.from_local(x, mesh)
        out = triton_square(x)
        out_dt = triton_square(x_dt)
        torch.testing.assert_close(out, x**2)
        torch.testing.assert_close(out_dt.to_local(), torch.ones_like(x))

    def test_replicated_fwd(self) -> None:
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh(device_type="cuda", mesh_shape=(self.world_size,))
        _x = torch.randn(3, 2345, device=self.device, requires_grad=True)
        x = deepcopy(_x)
        shard_spec = (Replicate(),)
        x_dt = distribute_tensor(deepcopy(_x), mesh, shard_spec)
        out = triton_square(x)
        out_dt = triton_square(x_dt)
        torch.testing.assert_close(out, out_dt.to_local())

    def test_replicated_bwd(self) -> None:
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh(device_type="cuda", mesh_shape=(self.world_size,))
        _x = torch.randn(3, 2345, device=self.device, requires_grad=True)
        x = deepcopy(_x).requires_grad_()
        shard_spec = (Replicate(),)
        x_dt = distribute_tensor(deepcopy(_x), mesh, shard_spec).requires_grad_()
        out = triton_square(x)
        out_dt = triton_square(x_dt)
        out.sum().backward()
        out_dt.sum().backward()
        assert x_dt.grad is not None
        torch.testing.assert_close(x.grad, x_dt.grad.to_local())

    def test_sharded_fwd(self) -> None:
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh(device_type="cuda", mesh_shape=(self.world_size,))
        _x = torch.randn(3, 2345, device=self.device, requires_grad=True)
        x = deepcopy(_x)
        shard_spec = (Shard(-1),)
        x_dt = distribute_tensor(deepcopy(_x), mesh, shard_spec)
        out = triton_square(x)
        out_dt = triton_square(x_dt)
        torch.testing.assert_close(out.chunk(self.world_size, dim=-1)[self.rank], out_dt.to_local())

    def test_sharded_bwd(self) -> None:
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh(device_type="cuda", mesh_shape=(self.world_size,))
        _x = torch.randn(3, 2345, device=self.device, requires_grad=True)
        x = deepcopy(_x).requires_grad_()
        shard_spec = (Shard(-1),)
        x_dt = distribute_tensor(deepcopy(_x), mesh, shard_spec).requires_grad_()
        out = triton_square(x)
        out_dt = triton_square(x_dt)
        out.sum().backward()
        out_dt.sum().backward()
        assert x_dt.grad is not None
        torch.testing.assert_close(
            x.grad.chunk(self.world_size, dim=-1)[self.rank], x_dt.grad.to_local()
        )

    def test_slice(self) -> None:
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh(device_type="cuda", mesh_shape=(self.world_size,))
        _x = torch.randn(3, 2345, device=self.device, requires_grad=True)
        x = deepcopy(_x)
        shard_spec = (Shard(-1),)
        x_dt = distribute_tensor(deepcopy(_x), mesh, shard_spec)
        x_dt_slice = x_dt[..., :-2]
        x_dt_slice
