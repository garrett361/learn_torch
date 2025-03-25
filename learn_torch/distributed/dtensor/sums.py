import torch
import torch.distributed as dist
from torch.distributed.tensor import Shard, distribute_tensor

from dtest import DTest


class Test(DTest):
    def test_sum_sharded(self) -> None:
        """
        What happens if a sharded axis is summed over.
        """

        mesh = dist.device_mesh.init_device_mesh(device_type="cuda", mesh_shape=(self.world_size,))

        shard_spec = (Shard(-1),)
        x = torch.arange(2 * self.world_size, device=self.device).reshape(-1, self.world_size)
        x_dt = distribute_tensor(x, mesh, shard_spec)

        x_sum = x.sum(dim=-1)
        x_dt_sum = x_dt.sum(dim=-1)
        x_dt_sum_squared = x_dt_sum**2
        x_dt_sum_squared

    def test_mean_sharded(self) -> None:
        mesh = dist.device_mesh.init_device_mesh(device_type="cuda", mesh_shape=(self.world_size,))

        shard_spec = (Shard(-1),)
        x = torch.arange(2 * self.world_size, device=self.device, dtype=torch.bfloat16).reshape(
            -1, self.world_size
        )
        x_dt = distribute_tensor(x, mesh, shard_spec)

        x_dt_mean = x_dt.mean(dim=-1)
        x_dt_mean_squared = x_dt_mean**2
        x_dt_mean_squared
