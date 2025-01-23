import os
import contextlib
from torch.distributed import destroy_process_group, init_process_group

import torch
import torch._dynamo.testing
import torch.distributed._composable.fsdp._fsdp_param
from torch import nn
from torch._dynamo import compiled_autograd
from torch.distributed._composable.fsdp import fully_shard


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
def _test_traceable_fsdp(model_init_fn, input_creation_fn, backend, fullgraph):
    def compiler_fn(compiled_autograd_backend):
        def _fn(gm):
            # fullgraph=True because graph-break in Compiled Autograd BWD graph is not supported by Traceable FSDP2 yet
            # (main difficulty comes from queue_callback not working well when BWD has graph break).
            return torch.compile(gm, backend=compiled_autograd_backend, fullgraph=True)

        return _fn

    def run_iters(model, optim, n_iter=10, compiled_autograd_backend=None):
        torch.manual_seed(42)
        losses = []
        for i in range(n_iter):
            inp = input_creation_fn()
            if compiled_autograd_backend is not None:
                maybe_compiled_autograd_ctx = compiled_autograd.enable(
                    compiler_fn(compiled_autograd_backend)
                )
            else:
                maybe_compiled_autograd_ctx = contextlib.nullcontext()
            with maybe_compiled_autograd_ctx:
                out = model(inp)
                loss = out.sum()
                losses.append(loss.item())
                loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
        return losses

    def test_compiled():
        model, optim = model_init_fn()
        # FSDP2 does lazy init using 1st run, so run it once to init using eager mode
        run_iters(model, optim, n_iter=1)

        model_compiled = torch.compile(model, backend=backend, fullgraph=fullgraph)
        res = run_iters(model_compiled, optim, compiled_autograd_backend=backend)
        return res

    def test_eager():
        model, optim = model_init_fn()
        # FSDP2 does lazy init using 1st run, so run it once to init using eager mode
        run_iters(model, optim, n_iter=1)

        res = run_iters(model, optim)
        return res

    losses_compiled = test_compiled()
    losses_eager = test_eager()


def _create_simple_mlp_factory_fns():
    hidden_dim = 16

    local_rank = int(os.environ["LOCAL_RANK"])

    def model_init_fn():
        torch.manual_seed(local_rank)
        fsdp_config = {}
        model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, device="cuda"),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device="cuda"),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device="cuda"),
        )
        fully_shard(model, reshard_after_forward=True, **fsdp_config)
        optim = torch.optim.SGD(model.parameters(), lr=1e-4)
        return model, optim

    def input_creation_fn():
        torch.manual_seed(local_rank)
        inp = torch.randn((2, hidden_dim), device="cuda", requires_grad=False)
        return inp

    return model_init_fn, input_creation_fn


def test_simple_mlp_fullgraph_backend_aot_eager():
    _test_traceable_fsdp(*_create_simple_mlp_factory_fns(), "aot_eager", fullgraph=True)


def test_simple_mlp_fullgraph_backend_aot_eager_decomp_partition():
    _test_traceable_fsdp(
        *_create_simple_mlp_factory_fns(),
        "aot_eager_decomp_partition",
        fullgraph=True,
    )


def test_simple_mlp_fullgraph_backend_inductor():
    _test_traceable_fsdp(*_create_simple_mlp_factory_fns(), "inductor", fullgraph=True)


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    init_process_group("nccl")
    try:
        test_simple_mlp_fullgraph_backend_aot_eager()
    finally:
        destroy_process_group()
