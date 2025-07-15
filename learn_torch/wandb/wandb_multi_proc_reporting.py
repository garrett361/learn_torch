import argparse
import os

import torch
import torch.distributed as dist
import wandb


def main(project_name, run_id, rank):
    run = wandb.init(
        project=project_name,
        settings=wandb.Settings(
            x_label=f"rank_{rank}",
            mode="shared",
            x_primary=rank == 0,
        ),
        id=run_id,
    )
    print(f"[{rank=}]: {run_id=} {run.id=}")

    # Log the rank as a wandb data point
    wandb.log({"rank": rank})


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-process wandb reporting")
    parser.add_argument("--project-name", required=True, help="Wandb project name")
    parser.add_argument("--run-id", required=True, help="Wandb run ID")
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
        main(args.project_name, args.run_id, RANK)
    finally:
        dist.destroy_process_group()
