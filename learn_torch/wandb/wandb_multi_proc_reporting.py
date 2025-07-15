import argparse
import os
import random

import torch
import torch.distributed as dist
import wandb


def main(project_name, run_id, rank, local_rank):
    x_primary = rank == 0
    x_label = f"rank_{rank}"
    # https://wandb.ai/dimaduev/simple-cnn-ddp/reports/Distributed-Training-with-Shared-Mode--VmlldzoxMTI0NTE1NA
    settings = wandb.Settings(
        x_label=x_label,
        mode="shared",
        x_stats_gpu_device_ids=[local_rank],
        x_primary=x_primary,
    )
    if rank != 0:
        # Do not upload wandb files except console logs.
        settings.x_primary = False
        # Do not change the state of the run on run.finish().
        settings.x_update_finish_state = False
    run = wandb.init(
        project=project_name,
        settings=settings,
        id=run_id,
    )

    print(f"[{rank=}]: {x_label=} {x_primary=} {run_id=} {run.id=}")

    # Log the rank as a wandb data point
    # Log some rank-specific stats
    for step in range(1, 11):
        stats = {f"stat_on_{rank}": random.random()}
        if not rank:
            stats["loss"] = random.random()

        # wandb.log(stats, step=step)
        wandb.log(stats)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-process wandb reporting")
    parser.add_argument("--project_name", required=True, help="Wandb project name")
    parser.add_argument("--run_id", required=True, help="Wandb run ID")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    RANK = int(os.environ["RANK"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{RANK}")
    torch.cuda.set_device(device)
    try:
        dist.init_process_group(backend="nccl", rank=RANK, world_size=WORLD_SIZE, device_id=device)
        main(args.project_name, args.run_id, RANK, LOCAL_RANK)
    finally:
        dist.destroy_process_group()
