import torch
import torch.distributed as dist
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import OpSchema, OpStrategy, PlacementStrategy
from torch.distributed.tensor._ops.utils import (
    register_op_strategy,
)
from torch.distributed.tensor.device_mesh import DeviceMesh

from dtest import DTest


@torch.library.custom_op("learn_torch::replicated_add", mutates_args=())
def _add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


@_add.register_fake
def _(x: torch.Tensor, y: torch.Tensor):
    return torch.empty_like(x)


# Register an op strategy. The decorated function should have be a
# Callable[[DeviceMesh, OpSchema], StrategyType]


# Say we always want the tensor to be fully replicated:
@register_op_strategy(torch.ops.learn_torch.replicated_add.default)
def _replicated_add_strat(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # NOTE: @goon - The OpSchema has `OpStrategy` instances as its args_spec, not `DTensorSpec`s,
    # and so `op_schema.args_spec` is trivial. We also don't need to access it for this simple
    # start.
    return OpStrategy(
        [
            PlacementStrategy(
                DTensorSpec(mesh=mesh, placements=[Replicate() for _ in range(mesh.ndim)])
            )
        ]
    )


class TestStrategies(DTest):
    def test_replicated_add(self) -> None:
        torch.manual_seed(42)
        mesh = dist.device_mesh.init_device_mesh(device_type="cuda", mesh_shape=(self.world_size,))
        x = torch.randn(2, 32, device=self.device, requires_grad=True)
        y = torch.randn(2, 32, device=self.device, requires_grad=True)
        x_dt = distribute_tensor(x, mesh, (Shard(-1),))
        y_dt = distribute_tensor(y, mesh, (Replicate(),))
        out = x_dt + y_dt
        out_custom = torch.ops.learn_torch.replicated_add.default(x_dt, y_dt)
        assert not any(isinstance(p, Replicate) for p in out.placements)
        assert all(isinstance(p, Replicate) for p in out_custom.placements)
