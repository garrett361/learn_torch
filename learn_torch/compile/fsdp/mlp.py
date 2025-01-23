import argparse
import warnings
from torch._dynamo import compiled_autograd
import contextlib

import os
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed._composable.fsdp import fully_shard

"""
Extremely minimal script for testing various torch compile options. Testing against 2.5.1

Based on the tests here: https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/test/distributed/_composable/fsdp/test_fully_shard_compile.py?plain=1#L324

torchrun --nproc-per-node 2 <filename.py>

Relevant env vars:
- export TORCH_COMPILE_DEBUG=1 to get graphs
- export TORCH_LOGS="graph_breaks,aot_graphs,compiled_autograd_verbose" for graph info
- export TORCH_LOGS_FORMAT="short" for shorter output
"""


def compiler_fn(backend):
    def _fn(gm):
        # fullgraph=True because graph-break in Compiled Autograd BWD graph is not supported by Traceable FSDP2 yet
        # (main difficulty comes from queue_callback not working well when BWD has graph break).
        return torch.compile(gm, backend=backend, fullgraph=True)

    return _fn


def fwd_bwd(
    model: nn.Module,
    inputs: torch.Tensor,
    optim: Optional[torch.optim.Optimizer] = None,
    backend: Optional[str] = None,
) -> torch.Tensor:
    if backend is not None:
        maybe_compiled_autograd_ctx = compiled_autograd.enable(compiler_fn(backend))
    else:
        maybe_compiled_autograd_ctx = contextlib.nullcontext()
    with maybe_compiled_autograd_ctx:
        out = model(inputs)
        loss = out.sum()
        loss.backward()
    if optim is not None:
        optim.step()
        optim.zero_grad(set_to_none=True)
    return loss


@torch._dynamo.config.patch(
    inline_inbuilt_nn_modules=True,
    skip_fsdp_hooks=False,
)
@torch._functorch.config.patch(recompute_views=True)
@torch._functorch.config.patch(cse=False)
@torch._inductor.config.patch(
    reorder_for_compute_comm_overlap=True,
    reorder_for_compute_comm_overlap_passes=[
        "sink_waits",
        "raise_comms",
        "reorder_compute_for_overlap",
    ],
)
def test_compiled(
    model: nn.Module,
    inputs: torch.Tensor,
    fwd_bwd: Callable,
    optim: Optional[torch.optim.Optimizer] = None,
    backend="inductor",
):
    # eager warmup for 1 iteration, so that all FSDP2 lazy-initialization is done in eager
    # Can use torch.compile.set_stance in newer torch versions instead.
    fwd_bwd(model, inputs, optim, None)
    print("DONE warmup")

    model_compiled = torch.compile(model, backend=backend, fullgraph=True)
    for step in range(10):
        print(f"Compiled {step=}")
        fwd_bwd(model_compiled, inputs, optim, backend)
    print("DONE compiled")


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

    model = get_fsdp_model(device, dtype)
    if not local_rank:
        print(model)
    optim = None if args.no_optim else torch.optim.SGD(model.parameters(), lr=1e-7)

    inputs = torch.randn(1, args.d_model, device=device, dtype=dtype)

    torch._dynamo.reset()
    torch._dynamo.compiled_autograd.reset()
    test_compiled(model, inputs, fwd_bwd, optim=optim, backend=args.backend)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument(
        "--backend",
        type=str,
        default="inductor",
        choices=["inductor", "aot_eager", "aot_eager_decomp_partition"],
    )
    parser.add_argument("--no_optim", action="store_true")
    args = parser.parse_args()
    if args.no_optim:
        warnings.warn("--no_optim currently erroring!")
    try:
        main(args)
    finally:
        destroy_process_group()
