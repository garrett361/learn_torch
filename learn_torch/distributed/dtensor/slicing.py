import torch
import torch.distributed as dist
from torch.distributed.tensor import Shard, distribute_tensor

from dtest import DTest


class TestSlicing(DTest):
    def test_slice_replication(self) -> None:
        mesh = dist.device_mesh.init_device_mesh(device_type="cuda", mesh_shape=(self.world_size,))

        shard_spec = (Shard(-1),)
        x = torch.arange(2 * self.world_size, device=self.device)
        x_dt = distribute_tensor(x, mesh, shard_spec)

        # Create a slice with only self.world_size elements
        x_dt_slice = x_dt[: self.world_size]
        x_dt_slice_alt = distribute_tensor(x[: self.world_size], mesh, shard_spec)
        # Expect these would have the same local tensors and sharding pattern, but nope.
        assert x_dt_slice.placements != x_dt_slice_alt.placements
        assert x_dt_slice.to_local().shape != x_dt_slice_alt.to_local().shape

    def test_slice_non_sharded_axis(self) -> None:
        mesh = dist.device_mesh.init_device_mesh(device_type="cuda", mesh_shape=(self.world_size,))

        shard_spec = (Shard(-1),)
        x = torch.randn(self.world_size, 2 * self.world_size, device=self.device)
        x_dt = distribute_tensor(x, mesh, shard_spec)

        # Create a slice with only self.world_size elements
        x_dt_slice = x_dt[: self.world_size]
        x_dt_slice_alt = distribute_tensor(x[: self.world_size], mesh, shard_spec)
        # Expect these would have the same local tensors and sharding pattern, but nope.
        x_dt_slice
