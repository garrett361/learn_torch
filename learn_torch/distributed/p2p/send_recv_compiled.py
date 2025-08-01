import argparse
import os

import torch
import torch.distributed as dist

"""
Send simple tensors in a ring with send/recv or isend/irecv.
"""


@torch.compile(fullgraph=True)
def isend_irecv(send: torch.Tensor, recv: torch.Tensor, rank: int, world_size: int):
    ops = []
    ops.append(dist.P2POp(dist.isend, send, (rank + 1) % world_size))
    ops.append(dist.P2POp(dist.irecv, recv, (rank - 1) % world_size))
    for op in dist.batch_isend_irecv(ops):
        op.wait()


@torch.compile(fullgraph=True)
def send_recv(send: torch.Tensor, recv: torch.Tensor, rank: int, world_size: int):
    for send_rank in range(world_size):
        recv_rank = (send_rank + 1) % world_size
        if send_rank == rank:
            dist.send(send, recv_rank)
        elif recv_rank == rank:
            dist.recv(recv, send_rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--batched", action="store_true")

    args = parser.parse_args()
    print(f"{args=}")

    RANK = int(os.environ["RANK"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = torch.device(f"cuda:{LOCAL_RANK}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(f"cpu:{LOCAL_RANK}")
    send = torch.full((args.dim,), RANK, device=device)
    recv = torch.empty_like(send)
    try:
        dist.init_process_group(backend="nccl" if cuda_available else "gloo")
        dist.barrier()
        if args.batched:
            isend_irecv(send, recv, RANK, WORLD_SIZE)
        else:
            send_recv(send, recv, RANK, WORLD_SIZE)
        dist.barrier()
        torch.testing.assert_close(recv, torch.full_like(recv, (RANK - 1) % WORLD_SIZE))
        print(f"SUCCESS on {RANK=}")
    finally:
        dist.destroy_process_group()
