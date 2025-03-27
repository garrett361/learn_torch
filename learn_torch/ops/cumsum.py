from copy import deepcopy

import pytest
import torch

"""
Rewriting `torch.cumsum` in terms of other ops.


The full signature is `torch.cumsum(input, dim, *, dtype=None, out=None) → Tensor`, but just
covering the first two args here.
"""


def cumsum(input: torch.Tensor, dim: int) -> torch.Tensor:
    dim_size = input.shape[dim]
    cs_matrix = torch.ones(dim_size, dim_size, dtype=input.dtype, device=input.device).tril()
    n_dims = len(input.shape)
    dim = dim % n_dims  # Normalize
    n_dims_to_insert = n_dims - 1 - dim
    if n_dims_to_insert:
        cs_matrix = cs_matrix.view(
            *(cs_matrix.shape + torch.Size(1 for _ in range(n_dims_to_insert)))
        )
    return (input.unsqueeze(dim) * cs_matrix).sum(dim=dim + 1)


class TestCumsum:
    @pytest.mark.parametrize("dim", [1, 2, -1, -2])
    def test_fwd(self, dim: int) -> None:
        t = torch.randn(2, 32, 64)
        t_cs = cumsum(t, dim=dim)
        t_cs_torch = t.cumsum(dim=dim)
        torch.testing.assert_close(t_cs, t_cs_torch)

    @pytest.mark.parametrize("dim", [1, 2, -1, -2])
    def test_bwd(self, dim: int) -> None:
        t = torch.randn(2, 32, 64, requires_grad=True)
        t_copy = deepcopy(t)

        cumsum(t, dim=dim).sum().backward()
        t_copy.cumsum(dim=dim).sum().backward()
        torch.testing.assert_close(t.grad, t_copy.grad)
