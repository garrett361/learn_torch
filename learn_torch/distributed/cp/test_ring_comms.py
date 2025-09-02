from argparse import ArgumentParser

import torch
import torch.distributed as dist
from torch_cp_inference import ring_send_recv

"""
torchrun --nproc-per_node 4 <path-to-this-file>
"""

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_kv_heads", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--d_head", type=int, default=64)
    args = parser.parse_args()
    dist.init_process_group("gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    try:
        torch.manual_seed(42)

        tensor = torch.randn(args.batch_size, args.n_heads, args.seqlen, args.d_head)
        # Ring impl
        send_chunk = tensor.tensor_split(world_size, dim=2)[rank]

        compiled_ring_send_recv = torch.compile(ring_send_recv, fullgraph=True)

        recv_chunk = ring_send_recv(send_chunk)
        compiled_recv_chunk = compiled_ring_send_recv(send_chunk)
        torch.testing.assert_close(compiled_recv_chunk, recv_chunk)
        torch.testing.assert_close(tensor.tensor_split(world_size, dim=2)[rank - 1], recv_chunk)
        print(f"Succeeded on {rank=}\n")
    finally:
        dist.destroy_process_group()
