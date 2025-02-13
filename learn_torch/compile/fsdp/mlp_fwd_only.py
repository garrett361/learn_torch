import argparse

import os

import torch
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed._composable.fsdp import fully_shard

"""
FWD compile only.
Based on the tests here: https://github.com/pytorch/pytorch/blob/6c81435f16ff90f0c0464ea61b958f7328bbddcc/test/distributed/_composable/fsdp/test_fully_shard_compile.py?plain=1#L464

torchrun --nproc-per-node 2 <filename.py>

Relevant env vars:
- export TORCH_COMPILE_DEBUG=1 to get graphs
- export TORCH_LOGS="graph_breaks,aot_graphs,compiled_autograd_verbose" for graph info
- export TORCH_LOGS_FORMAT="short" for shorter output
"""


def test_compiled_autograd_ctx(fsdp_model, inputs):
    with torch._dynamo.config.patch(
        skip_fsdp_hooks=False,
    ), torch._functorch.config.patch(
        recompute_views=True,
    ):
        fsdp_model_compiled = torch.compile(fsdp_model, backend="inductor")
        for i in range(10):
            torch.compiler.set_stance(
                "force_eager" if i < 1 else "default"
            )  # eager warmup for 1 iteration
            with torch._dynamo.compiled_autograd._enable(
                torch.compile(backend="inductor", fullgraph=True)
            ):
                out = fsdp_model_compiled(inputs)
                out.sum().backward()


class BasicModel(nn.Module):
    def __init__(self, d_model: int, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.d_model = d_model
        self.lin0 = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.lin0(inputs).relu()
        return outputs


def get_fsdp_model(device: torch.device, dtype: torch.dtype) -> nn.Module:
    model = nn.Sequential(*(BasicModel(args.d_model, device, dtype) for _ in range(3)))
    for module in model.modules():
        if isinstance(module, BasicModel):
            fully_shard(module)
    fully_shard(model)
    return model


def main(args: argparse.Namespace) -> None:
    if int(os.environ["RANK"]) == 0:
        print(f"{torch.__version__=}")

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16
    torch.cuda.set_device(device)
    init_process_group("nccl")

    fsdp_model = get_fsdp_model(device, dtype)
    if not local_rank:
        print(fsdp_model)

    inputs = torch.randn(1, args.d_model, device=device, dtype=dtype)
    test_compiled_autograd_ctx(fsdp_model, inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=16)
    args = parser.parse_args()
    try:
        main(args)
    finally:
        destroy_process_group()
