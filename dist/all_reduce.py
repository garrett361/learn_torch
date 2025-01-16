import argparse
import os

import torch
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group, all_reduce
from torch.distributed._composable.fsdp import fully_shard

"""
Extremely minimal script all-reduce script for testing.

torchrun --nproc-per-node 2 basic_fsdp2.py
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=128)
    args = parser.parse_args()

    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        init_process_group("nccl")

        t = torch.arange(world_size, device=device, dtype=torch.float32)
        if not local_rank:
            print(f"Before: {t=}")
        all_reduce(t)
        if not local_rank:
            print(f"After: {t=}")

    finally:
        destroy_process_group()
