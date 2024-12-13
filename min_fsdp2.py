import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
import os

d_model = 128

if __name__ == "__main__":
    RANK = int(os.environ["RANK"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{RANK}")
    torch.cuda.set_device(device)
    try:
        dist.init_process_group(
            backend="nccl", rank=RANK, world_size=WORLD_SIZE, device_id=device
        )
        model = nn.Sequential(
            *(nn.Linear(d_model, d_model, bias=False, device=device) for _ in range(3))
        )
        for lin in model.modules():
            if isinstance(lin, nn.Linear):
                nn.init.eye_(lin.weight)
                fully_shard(lin)
        fully_shard(model)

        compiled_model = torch.compile(model)
        inputs = torch.randn(1, d_model, device=device)
        outputs = compiled_model(inputs)
        torch.testing.assert_close(inputs, outputs)
        print(f"SUCCESS on {RANK=}")
    finally:
        dist.destroy_process_group()
