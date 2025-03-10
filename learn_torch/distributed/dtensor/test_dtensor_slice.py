import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import Shard, distribute_tensor

if __name__ == "__main__":
    try:
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        mesh = dist.device_mesh.init_device_mesh(device_type="cuda", mesh_shape=(world_size,))

        # Create a DTensor with 2 * world_size logical elements, sharded over the world. Every rank
        # holds 2 tensor elements.
        shard_spec = (Shard(0),)
        x_dt = distribute_tensor(torch.arange(2 * world_size, device=device), mesh, shard_spec)

        # Create a slice with only world_size elements
        x_dt_slice = x_dt[..., :world_size]

        # Expectation: the DTensor slice should still be sharded, as in Shard(-1), and every rank
        # should hold a single local element.

        # Actual: the DTensor slice ends up with Replicate() sharding, with every rank holding
        # world_size elements.

        # For nicer printing:
        for rank in range(world_size):
            if rank == local_rank:
                print(f"\n{rank=}")

                print(f"\t{x_dt=}")
                print(f"\t{x_dt.placements=}")
                print(f"\t{x_dt.to_local().shape=}")

                print(f"\n\t{x_dt_slice=}")
                print(f"\t{x_dt_slice.placements=}")
                print(f"\t{x_dt_slice.to_local().shape=}")

            dist.barrier()

        assert x_dt.to_local().numel() == 2, f"{x_dt.to_local().numel=}"
        assert x_dt.placements == shard_spec, f"{x_dt.placements =}, {shard_spec=}"

        # These fail:
        assert x_dt_slice.to_local().numel() == 1
        assert x_dt_slice.placements == shard_spec, f"{x_dt_slice.placements =}, {shard_spec=}"
    finally:
        torch.distributed.destroy_process_group()
