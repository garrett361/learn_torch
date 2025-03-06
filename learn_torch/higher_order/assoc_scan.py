from typing import Callable
import torch


def segprod(x: torch.Tensor, dim: int = -1, offset: int = 0) -> torch.Tensor:
    """
    For an arbitrary tensor of shape  x.shape=(..., D, ...) with the size D dimension as position
    dim, return a (..., D, D, ...)-shaped tensor out such that

    ```
    out[..., i, j, ...] = x[...,  i:j+1, ...].cumprod(dim=dim) if i <= j else 0.0
    ```

    offset > 0 fills the first offset diagonals with ones before performing the cumprod,
    so that we instead get:

    ```
    out[..., i, j, ...] = x[...,  i+offset:j+1, ...].cumprod(dim=dim) if i + offset <= j
    out[..., i, j, ...] = 1.0 if i <= j < i + offset
    out[..., i, j, ...] = 0.0 if j < i
    ```
    """
    dim = dim % x.ndim
    D = x.shape[dim]
    x_segprod = x.unsqueeze(dim).repeat_interleave(D, dim)
    ones_mask = torch.tril(
        torch.ones(D, D, device=x.device, dtype=bool),
        diagonal=offset - 1,
    )
    # Expand the mask so it can be broadcast
    ones_mask = ones_mask.expand(*x.shape[:dim], -1, -1, *x.shape[dim + 1 :])

    x_segprod.masked_fill_(ones_mask, 1.0)
    x_segprod = x_segprod.cumprod(dim=dim + 1)
    zeros_mask = torch.tril(
        torch.ones(D, D, device=x.device, dtype=bool),
        diagonal=-1,
    )
    zeros_mask = zeros_mask.expand(*x.shape[:dim], -1, -1, *x.shape[dim + 1 :])
    x_segprod.masked_fill_(zeros_mask, 0.0)
    return x_segprod


def dsum(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the pointwise derivative of the sum operation with respect to its inputs.
    """
    return torch.ones_like(x), torch.ones_like(y)


def dprod(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the pointwise derivative of the prod operation with respect to its inputs.
    """
    return y, x


def get_scan_derivative_pointwise(
    outputs_grad: torch.Tensor,
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    dop_pointwise: Callable[[torch.Tensor, torch.Tensor], [torch.Tensor, torch.Tensor]],
    scan_dim: int = -1,
):
    """
    Returns the derivative of the scan with respect to its inputs, given the upstream grad, scan
    inputs and outputs, and the derivative function for the associative op, assumed to be pointwise.

    Setup: {inputs, outputs} are {x_i, y_i}, and upstream grads are g_i = dL/dy_i . The pointwise
    associative scan op is A(x, y), such that y_i = A(x_i, y_{i-1}). The desired derivatives are
    (sums over the repeated j index implicit):

    ```
    dL/dx_i = g_j * dy_j/dx_i
            = g_j * (dA(x_j, y_{j-1})/dy_{j-1}) * dy_{j-1}/dx_i
            = g_j * (dA(x_j, y_{j-1})/dy_{j-1}) * ... * (dA(x_{i+1}, y_{i})/dy_{i}) * dA(x_i, y_i-1)/dx_i
            = g_j * cumprod(D[i+1:j+1]) * dA(x_i, y_i-1)/dx_i
    ````

    where `D_j = dA(x_j, y_j-1)/dy_{j-1})` and the cumprod of the trivial slice should be understood
    as giving 1 and invalid slices giving zero:

    ```
    cumprod(D[i+1:j+1]) = 1 if i == j
    cumprod(D[i+1:j+1]) = 0 if i > j
    ```

    So, cumprod(D[i+1:j+1]) is equal to a segprod of a shifted D with a diagonal filled with ones.
    """
    scan_dim = scan_dim % inputs.ndim
    # For the assoc. op A(x, y), get the derivatives dA(x_i, y_{i-1})/dy_{i-1} and
    # dA(x_i,y_{i-1})/dx_i for all i, with invalid index values giving derivatives equal to 1.
    dop_pointwise_vmap = torch.vmap(dop_pointwise, out_dims=(0, 0))
    dop_x, dop_y = dop_pointwise_vmap(inputs, outputs.roll(1, scan_dim))
    # Set the i = 0 elements of the x derivative to 1
    torch.select(dop_x, scan_dim, 0).fill_(1.0)
    D_segprod = segprod(dop_y, dim=scan_dim, offset=1)
    inputs_grad = torch.bmm(D_segprod, outputs_grad[..., None]).squeeze(-1) * dop_x

    return inputs_grad


class TestAssocScanDerivatives:
    batch_size = 2
    d_model = 16

    def test_segprod(self) -> None:
        torch.manual_seed(42)
        inputs = torch.randn(self.batch_size, self.d_model)
        inputs_segprod = segprod(inputs, dim=1)
        for d in range(self.d_model):
            expected_cumprod = inputs[:, d:].cumprod(dim=-1)
            actual_cumprod = inputs_segprod[:, d, d:]
            torch.testing.assert_close(expected_cumprod, actual_cumprod)
            # Other elements are zeros:
            zeros = inputs_segprod[:, d, :d]
            torch.testing.assert_close(zeros, torch.zeros_like(zeros))

    def test_segprod_with_offset(self) -> None:
        torch.manual_seed(42)
        inputs = torch.randn(self.batch_size, self.d_model)
        for offset in range(1, self.d_model):
            inputs_segprod = segprod(inputs, dim=1, offset=offset)
            for d in range(self.d_model):
                expected_cumprod = inputs[:, d + offset :].cumprod(dim=-1)
                actual_cumprod = inputs_segprod[:, d, d + offset :]
                torch.testing.assert_close(expected_cumprod, actual_cumprod)
                # Other elements are zeros or ones:
                zeros = inputs_segprod[:, d, :d]
                torch.testing.assert_close(zeros, torch.zeros_like(zeros))
                ones = inputs_segprod[:, d, d : d + offset - 1]
                torch.testing.assert_close(ones, torch.ones_like(ones))

    def test_cumsum(self) -> None:
        torch.manual_seed(42)
        inputs = torch.randn(self.batch_size, self.d_model, requires_grad=True)
        outputs = inputs.cumsum(dim=-1)
        grad_out = torch.randn_like(outputs)
        # Populate grad on inputs
        outputs.backward(grad_out)

        # Compute the same grads with our helper
        inputs_grad = get_scan_derivative_pointwise(grad_out, outputs, inputs, dsum)
        torch.testing.assert_close(inputs.grad, inputs_grad)

    def test_cumprod(self) -> None:
        torch.manual_seed(42)
        inputs = torch.randn(self.batch_size, self.d_model, requires_grad=True)
        outputs = inputs.cumprod(dim=-1)
        grad_out = torch.randn_like(outputs)
        # Populate grad on inputs

        outputs.backward(grad_out)
        # Compute the same grads with our helper
        inputs_grad = get_scan_derivative_pointwise(grad_out, outputs, inputs, dprod)
        torch.testing.assert_close(inputs.grad, inputs_grad)
