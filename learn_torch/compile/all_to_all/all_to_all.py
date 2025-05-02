from argparse import ArgumentParser

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

"""
Compile + dynamic all-to-all compute

torchrun --nproc-per_node 2 <path-to-this-file>
"""


def fn(inputs: torch.Tensor, size_t: torch.Tensor) -> torch.Tensor:
    default_pg = dist.GroupMember.WORLD
    outputs = funcol.all_to_all_single_autograd(
        inputs,
        size_t.tolist(),
        size_t.tolist(),
        group=default_pg,
    )
    return outputs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    dist.init_process_group("gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    try:
        torch.manual_seed(42)
        if args.compile:
            fn = torch.compile(fn)
        for size in range(4, 8):
            size_t = torch.full((world_size,), size, dtype=torch.int64)
            inputs = torch.randn(world_size * size, args.dim, requires_grad=True)
            outputs = fn(inputs, size_t)
            outputs.sum().backward()

    finally:
        dist.destroy_process_group()
