import argparse
import os

import torch
import torch.distributed as dist


def isend_irecv(send: torch.Tensor, recv: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    ops = []
    ops.append(dist.P2POp(dist.isend, send, (rank + 1) % world_size))
    ops.append(dist.P2POp(dist.irecv, recv, (rank - 1) % world_size))
    for op in dist.batch_isend_irecv(ops):
        op.wait()


def send_recv(send: torch.Tensor, recv: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    dist.send(send, (rank + 1) % world_size)
    dist.recv(recv, (rank - 1) % world_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--batched", action="store_true")

    args = parser.parse_args()
    print(f"{args=}")

    RANK = int(os.environ["RANK"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{LOCAL_RANK}")
    torch.cuda.set_device(device)
    send = torch.full((args.dim,), RANK, device=device)
    recv = torch.empty_like(send)
    try:
        dist.init_process_group(backend="nccl")
        if args.batched:
            isend_irecv(send, recv, RANK, WORLD_SIZE)
        else:
            send_recv(send, recv, RANK, WORLD_SIZE)
        dist.barrier()
        torch.testing.assert_close(recv, torch.full_like(recv, (RANK - 1) % WORLD_SIZE))
        print(f"SUCCESS on {RANK=}")
    finally:
        dist.destroy_process_group()
