import functools
from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, fully_shard
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.tensor import DTensor

from dtest import DTest


def _test_grads_fsdp2(model: nn.Module, model_fsdp: nn.Module, tol: float) -> None:
    with torch.no_grad():
        for n, m_fsdp in model_fsdp.named_modules():
            m = model.get_submodule(n)
            for (n, p), (_, p_fsdp) in zip(
                m.named_parameters(recurse=False),
                m_fsdp.named_parameters(recurse=False),
            ):
                if p.grad is None:
                    assert p_fsdp.grad is None
                    return
                grad = p.grad
                grad_fsdp = p_fsdp.grad
                if isinstance(grad_fsdp, DTensor):
                    grad_fsdp = grad_fsdp.full_tensor()
                try:
                    torch.testing.assert_close(grad, grad_fsdp, atol=tol, rtol=tol)
                except Exception as e:
                    raise RuntimeError(f"Failed on {n=}") from e


def _test_grads_fsdp1(model: nn.Module, model_fsdp: nn.Module, tol: float) -> None:
    with torch.no_grad(), FSDP.summon_full_params(model_fsdp, with_grads=True):
        grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        grads_fsdp = {}
        for n, p in model_fsdp.named_parameters():
            if p.grad is not None:
                g_cp = deepcopy(p.grad)
                grads_fsdp[n] = g_cp

        for n, g_fsdp in grads_fsdp.items():
            g = grads[n]
            try:
                torch.testing.assert_close(g, g_fsdp, atol=tol, rtol=tol)
            except Exception as e:
                raise RuntimeError(f"Failed on {n=}") from e


class TestGradClipFSDP(DTest):
    d_model = 128
    n_layers = 3
    batch_size = 2
    clip = 1.0

    def get_model(self) -> nn.Module:
        return nn.Sequential(
            *[
                nn.Linear(self.d_model, self.d_model, bias=False, device=self.device)
                for _ in range(self.n_layers)
            ]
        )

    def test_fsdp2(self) -> None:
        torch.manual_seed(42)
        model = self.get_model()

        model_fsdp = deepcopy(model)
        for m in model_fsdp.modules():
            if isinstance(m, nn.Linear):
                fully_shard(m)
        fully_shard(model_fsdp)

        inputs = torch.randn(self.batch_size * self.world_size, self.d_model, device=self.device)
        inputs_fsdp = inputs.chunk(self.world_size, dim=0)[self.rank]
        outputs = model(inputs)
        outputs_fsdp = model_fsdp(inputs_fsdp)

        torch.testing.assert_close(outputs.chunk(self.world_size, dim=0)[self.rank], outputs_fsdp)

        (100 * outputs.mean()).backward()
        (100 * outputs_fsdp.mean()).backward()

        # Grads agree:
        _test_grads_fsdp2(model, model_fsdp, 1e-2)

        # Also agree after clipping:
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), self.clip)
        grad_norm_fsdp = nn.utils.clip_grad_norm_(model_fsdp.parameters(), self.clip)
        assert grad_norm > self.clip, "clip not needed"
        _test_grads_fsdp2(model, model_fsdp, 1e-2)

        # Calling full_tensor() gives the proper norm
        torch.testing.assert_close(grad_norm, grad_norm_fsdp.full_tensor())

        # Calling item() only gives the norm of the local shard.
        g_item_t = torch.tensor(grad_norm_fsdp.item(), device=self.device)
        # Average across the world:
        dist.all_reduce(g_item_t)
        g_item_t /= self.world_size

        # Sum of norms != proper norm
        with pytest.raises(AssertionError):
            torch.testing.assert_close(grad_norm, g_item_t)
        # Off by a factor of sqrt(world_size)
        torch.testing.assert_close(
            grad_norm, (self.world_size) ** 0.5 * g_item_t, atol=1e-2, rtol=1e-2
        )

    def test_fsdp1(self) -> None:
        torch.manual_seed(42)
        model = self.get_model()

        model_fsdp = deepcopy(model)
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                nn.Linear,
            },
        )

        model_fsdp = FSDP(
            model_fsdp,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap_policy,
            use_orig_params=True,
        )

        inputs = torch.randn(self.batch_size * self.world_size, self.d_model, device=self.device)
        inputs_fsdp = inputs.chunk(self.world_size, dim=0)[self.rank]
        outputs = model(inputs)
        outputs_fsdp = model_fsdp(inputs_fsdp)

        torch.testing.assert_close(outputs.chunk(self.world_size, dim=0)[self.rank], outputs_fsdp)

        (100 * outputs.mean()).backward()
        (100 * outputs_fsdp.mean()).backward()

        _test_grads_fsdp1(model, model_fsdp, 1e-2)

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), self.clip)
        grad_norm_fsdp = model_fsdp.clip_grad_norm_(self.clip)
        assert grad_norm > self.clip, "clip not needed"
        _test_grads_fsdp1(model, model_fsdp, 1e-2)

        # Passes
        torch.testing.assert_close(grad_norm, grad_norm_fsdp)
        g_item_t = torch.tensor(grad_norm_fsdp.item(), device=self.device)
        dist.all_reduce(g_item_t)
        g_item_t /= self.world_size
        torch.testing.assert_close(grad_norm, g_item_t)
