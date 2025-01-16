import argparse
import os

import torch
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed._composable.fsdp import fully_shard

"""
Extremely minimal script for testing various torch compile options

torchrun --nproc-per-node 2 basic_fsdp2.py

Trying to get the full graph to compile.
"""


class BasicModel(nn.Module):
    def __init__(self, d_model: int, device: torch.device) -> None:
        super().__init__()
        self.d_model = d_model
        self.lin0 = nn.Linear(d_model, d_model, bias=False, device=device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.lin0(inputs).relu()
        return outputs


def set_torch_compile_flags() -> None:
    # From linsong, inherited from FSDP tests
    # https://github.com/pytorch/pytorch/blob/135a2d44830b2de1ed6714f52cc6a612406adb6d/test/distributed/_composable/fsdp/test_fully_shard_compile.py#L569-L570
    torch._dynamo.config.skip_fsdp_hooks = False
    torch._dynamo.config.compiled_autograd = True
    torch._dynamo.config.inline_inbuilt_nn_modules = True
    torch._functorch.config.enable_autograd_cache = False
    torch._functorch.config.recompute_views = True
    torch._inductor.config.force_disable_caches = True
    torch._inductor.config.reorder_for_compute_comm_overlap = True
    torch._inductor.config.reorder_for_compute_comm_overlap_passes = [
        "sink_waits",
        "raise_comms",
        "reorder_compute_for_overlap",
    ]


def main() -> None:
    set_torch_compile_flags()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    init_process_group("nccl")

    model = nn.Sequential(*(BasicModel(args.d_model, device) for _ in range(3)))
    for module in model.modules():
        if isinstance(module, BasicModel):
            fully_shard(module)
    fully_shard(model)
    if not local_rank:
        print(f"{model=}")
        for name, param in model.named_parameters():
            print(f"{name=}, {param=}")

    compiled_model = torch.compile(model)
    inputs = torch.randn(1, args.d_model, device=device)

    outputs = compiled_model(inputs)
    print(f"{local_rank=}, {outputs=}")
    outputs.pow(2).mean().backward()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=128)
    args = parser.parse_args()

    if not int(os.environ["RANK"]):
        os.environ["TORCH_COMPILE_DEBUG"] = "1"
        os.environ["TORCH_LOGS"] = "graph_breaks,aot_graphs"

    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        init_process_group("nccl")

        model = nn.Sequential(*(BasicModel(args.d_model, device) for _ in range(3)))
        for module in model.modules():
            if isinstance(module, BasicModel):
                fully_shard(module)
        fully_shard(model)
        if not local_rank:
            print(f"{model=}")
            for name, param in model.named_parameters():
                print(f"{name=}, {param=}")

        compiled_model = torch.compile(model)
        inputs = torch.randn(1, args.d_model, device=device)

        outputs = compiled_model(inputs)
        print(f"{local_rank=}, {outputs=}")
        outputs.pow(2).mean().backward()
    finally:
        destroy_process_group()
