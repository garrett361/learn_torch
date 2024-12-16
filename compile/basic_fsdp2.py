import argparse
import os

import torch
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed._composable.fsdp import fully_shard

"""
Extremely minimal script for testing various torch compile options

torchrun --nproc-per-node 2 basic_fsdp2.py
"""


def get_simple_linear_model(d_model: int, device: torch.device) -> nn.Module:
    model = nn.Sequential(
        *(nn.Linear(d_model, d_model, bias=False, device=device) for _ in range(3))
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=128)
    args = parser.parse_args()

    try:
        local_rank = os.environ["LOCAL_RANK"]
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        init_process_group("nccl")

        model = get_simple_linear_model(args.d_model, device)
        for lin in model.modules():
            if isinstance(lin, nn.Linear):
                nn.init.eye_(lin.weight)
                fully_shard(lin)
        fully_shard(model)

        compiled_model = torch.compile(model)
        inputs = torch.randn(1, args.d_model, device=device)
        outputs = compiled_model(inputs)

        outputs = compiled_model(inputs)
        print(f"{outputs=}")
        print(f"{outputs.shape=}")
    finally:
        destroy_process_group()
