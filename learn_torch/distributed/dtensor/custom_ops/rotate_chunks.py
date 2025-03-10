from copy import deepcopy

import torch
import torch.distributed as dist
from einops import rearrange
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSchema, OutputSharding
from torch.distributed.tensor._ops.utils import register_prop_rule

from dtest import DTest

"""
Example of non-pointwise custom op registration for DTensor.
"""


@torch.library.custom_op("learn_torch::rotate_chunk", mutates_args=())
def _rotate_chunk(x: torch.Tensor, num_chunks: int) -> torch.Tensor:
    chunk_size, remainder = divmod(x.shape[-1], num_chunks)
    # Enforce divisibility for simplicity
    assert remainder == 0
    return x.roll(chunk_size, -1).contiguous()


# Create an explicit backward op so that we can override its DTensor impl.
@torch.library.custom_op("learn_torch::rotate_chunk_backward", mutates_args=())
def _rotate_chunk_backward(grad_outputs: torch.Tensor, num_chunks: int) -> torch.Tensor:
    chunk_size, remainder = divmod(grad_outputs.shape[-1], num_chunks)
    # Enforce divisibility for simplicity
    assert remainder == 0
    return grad_outputs.roll(-chunk_size, -1)


# Shape registration
@_rotate_chunk.register_fake
def _(x: torch.Tensor, num_chunks: int):
    return torch.empty_like(x)


@_rotate_chunk_backward.register_fake
def _(x: torch.Tensor, num_chunks: int):
    return torch.empty_like(x)


# Autograd
def setup_context_rotate_chunk(ctx, inputs, output):
    ctx.num_chunks = inputs[-1]


def backward_rotate_chunk(ctx, grad):
    return torch.ops.learn_torch.rotate_chunk_backward.default(grad, ctx.num_chunks), None


_rotate_chunk.register_autograd(backward_rotate_chunk, setup_context=setup_context_rotate_chunk)


class Test:
    batch_size = 3
    num_chunks = 4
    d_model = 32
    chunk_size = d_model // num_chunks

    def test_fwd(self) -> None:
        x = torch.randn(self.batch_size, self.d_model, device="cuda")
        out_custom = torch.ops.learn_torch.rotate_chunk.default(x, self.num_chunks)
        out_expected = x.roll(self.chunk_size, -1)

        torch.testing.assert_close(out_custom, out_expected)

    def test_bwd(self) -> None:
        x = torch.randn(self.batch_size, self.d_model, device="cuda", requires_grad=True)
        x_copy = deepcopy(x)
        out_custom = torch.ops.learn_torch.rotate_chunk.default(x, self.num_chunks)
        out_expected = x_copy.roll(self.chunk_size, -1)
        out_custom.pow(2).sum().backward()
        out_expected.pow(2).sum().backward()
        torch.testing.assert_close(x.grad, x_copy.grad)

    def test_opcheck(self) -> None:
        inputs = (
            (torch.randn(1, self.d_model * self.num_chunks, device="cuda"), self.num_chunks),
            (
                torch.randn(1, self.d_model * self.num_chunks, device="cuda", requires_grad=True),
                self.num_chunks,
            ),
            (torch.randn(3, 3 * self.d_model * self.num_chunks, device="cuda"), self.num_chunks),
            (
                torch.randn(
                    3, 3 * self.d_model * self.num_chunks, device="cuda", requires_grad=True
                ),
                self.num_chunks,
            ),
        )
        for args in inputs:
            torch.library.opcheck(torch.ops.learn_torch.rotate_chunk, args)


# Start DTensor-specific code and tests


@register_prop_rule(torch.ops.learn_torch.rotate_chunk.default)
def _(op_schema: OpSchema) -> OutputSharding:
    (input_spec,) = op_schema.args_spec

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


@register_prop_rule(torch.ops.learn_torch.rotate_chunk_backward.default)
def _(op_schema: OpSchema) -> OutputSharding:
    (input_spec,) = op_schema.args_spec

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


# Define DTensor specific impls for use with sharded tensors.
def _rotate_chunk_dtensor(x: DTensor, num_chunks: int) -> torch.Tensor:
    # Just cover the easy cases
    mesh = x.device_mesh
    if mesh.ndim != 1:
        raise ValueError("Only 1D meshes supported.")
    mesh_size = mesh.size()
    if mesh_size != num_chunks:
        raise ValueError(f"Only meshes of size {num_chunks} supported: {x.device_mesh.size()=}")

    # Really poor allgather impl:
    out = torch.empty(mesh_size, *x.to_local().shape, dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(out, x.to_local().contiguous(), group=mesh.get_group(0))
    out = rearrange(out, "w ... c -> ... (w c)")
    return out.chunk(mesh_size, -1)[(mesh.get_rank() + 1) % mesh_size]


@torch.library.register_torch_dispatch("learn_torch::rotate_chunk", DTensor)
def _(cls, op_call, type, args, kwargs) -> DTensor:
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    # sharding propagation
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"

    (inputs, num_chunks) = args
    inputs_spec = op_info.schema.args_spec[0]
    if not inputs_spec.is_sharded():
        local_results = torch.ops.learn_torch.rotate_chunk.default(inputs.to_local(), num_chunks)
    else:
        shard_dim = None
        for p in inputs_spec.placements:
            if isinstance(p, Shard):
                if shard_dim is not None:
                    raise ValueError(f"More than one sharded dim found: {inputs_spec.placements=}")
                shard_dim = p.dim % inputs_spec.ndim
                if shard_dim != inputs_spec.ndim - 1:
                    raise ValueError(f"Expected sharding on final dim, found {p}")
        local_results = _rotate_chunk_dtensor(inputs, num_chunks)

    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


class TestDTensor(DTest):
    batch_size = 3

    def test_replicated_fwd(self) -> None:
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh(device_type="cuda", mesh_shape=(self.world_size,))
        _x = torch.randn(
            self.batch_size, 32 * self.world_size, device=self.device, requires_grad=True
        )
        x = deepcopy(_x)
        shard_spec = (Replicate(),)
        x_dt = distribute_tensor(deepcopy(_x), mesh, shard_spec)
        out = torch.ops.learn_torch.rotate_chunk.default(x, self.world_size)
        out_dt = torch.ops.learn_torch.rotate_chunk.default(x_dt, self.world_size)
        torch.testing.assert_close(out, out_dt.to_local())

    def test_replicated_bwd(self) -> None:
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh(device_type="cuda", mesh_shape=(self.world_size,))
        _x = torch.randn(3, 2345, device=self.device, requires_grad=True)
        x = deepcopy(_x).requires_grad_()
        shard_spec = (Replicate(),)
        x_dt = distribute_tensor(deepcopy(_x), mesh, shard_spec).requires_grad_()
        out = torch.ops.learn_torch.rotate_chunk.default(x)
        out_dt = torch.ops.learn_torch.rotate_chunk.default(x_dt)
        out.sum().backward()
        out_dt.sum().backward()
        assert x_dt.grad is not None
        torch.testing.assert_close(x.grad, x_dt.grad.to_local())

    def test_sharded_fwd(self) -> None:
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh(device_type="cuda", mesh_shape=(self.world_size,))
        _x = torch.randn(
            self.batch_size, 32 * self.world_size, device=self.device, requires_grad=True
        )
        x = deepcopy(_x)
        shard_spec = (Shard(-1),)
        x_dt = distribute_tensor(deepcopy(_x), mesh, shard_spec)
        out = torch.ops.learn_torch.rotate_chunk.default(x, self.world_size)
        out_dt = torch.ops.learn_torch.rotate_chunk.default(x_dt, self.world_size)
        torch.testing.assert_close(out.chunk(self.world_size, -1)[self.rank], out_dt.to_local())

    def test_sharded_bwd(self) -> None:
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh(device_type="cuda", mesh_shape=(self.world_size,))
        _x = torch.randn(3, 2345, device=self.device, requires_grad=True)
        x = deepcopy(_x).requires_grad_()
        shard_spec = (Shard(-1),)
        x_dt = distribute_tensor(deepcopy(_x), mesh, shard_spec).requires_grad_()
        out = torch.ops.learn_torch.rotate_chunk.default(x)
        out_dt = torch.ops.learn_torch.rotate_chunk.default(x_dt)
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
