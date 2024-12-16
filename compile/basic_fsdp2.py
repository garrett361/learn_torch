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


class BasicModel(nn.Module):
    def __init__(self, d_model: int, device: torch.device) -> None:
        super().__init__()
        self.d_model = d_model
        self.lin0 = nn.Linear(d_model, 2 * d_model, bias=False, device=device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.lin0(inputs).relu()
        return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=128)
    args = parser.parse_args()

    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        init_process_group("nccl")

        model = BasicModel(args.d_model, device)
        for lin in model.modules():
            if isinstance(lin, nn.Linear):
                nn.init.eye_(lin.weight)
                fully_shard(lin)
        fully_shard(model)
        if not local_rank:
            print(f"{model=}")

        compiled_model = torch.compile(model)
        inputs = torch.randn(1, args.d_model, device=device)

        outputs = compiled_model(inputs)
        if not local_rank:
            print(f"{outputs=}")
            print(f"{outputs-inputs=}")
    finally:
        destroy_process_group()
