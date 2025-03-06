from typing import Callable
import torch


def segprod(x: torch.Tensor, dim: int = -1, offset: int = 0) -> torch.Tensor:
def segprod(x: torch.Tensor, dim: int = 1, offset: int = 0) -> torch.Tensor:
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
    # Broadcast the mask to the same shape as x. Can probably do this more simply?
    leading_dims, trailing_dims = x.shape[:dim], x.shape[dim + 1 :]
    if leading_dims:
        ones_mask = ones_mask[*(None for _ in leading_dims)]
    if trailing_dims:
        ones_mask = ones_mask[..., *(None for _ in trailing_dims)]
    ones_mask = ones_mask.expand(*leading_dims, -1, -1, *trailing_dims)

    x_segprod.masked_fill_(ones_mask, 1.0)
    x_segprod = x_segprod.cumprod(dim=dim + 1)

    # Similar broadcast with the zeros mask.
    zeros_mask = torch.tril(
        torch.ones(D, D, device=x.device, dtype=bool),
        diagonal=-1,
    )
    if leading_dims:
        zeros_mask = zeros_mask[*(None for _ in leading_dims)]
    if trailing_dims:
        zeros_mask = zeros_mask[..., *(None for _ in trailing_dims)]
    zeros_mask = zeros_mask.expand(*leading_dims, -1, -1, *trailing_dims)

    x_segprod.masked_fill_(zeros_mask, 0.0)
    return x_segprod


def sum_bwd(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the pointwise derivative of the sum (torch.add) operation with respect to its inputs.
    """
    return torch.ones_like(x), torch.ones_like(y)


def prod_bwd(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the pointwise derivative of the prod (torch.mul) operation with respect to its inputs.
    """
    return y, x


def get_scan_derivative_pointwise(
def get_scan_derivative(
    outputs_grad: torch.Tensor,
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    op_bwd: Callable[[torch.Tensor, torch.Tensor], [torch.Tensor, torch.Tensor]],
    dim: int = 1,
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
    dim = dim % inputs.ndim
    # For the assoc. op A(x, y), get the derivatives dA(x_i, y_{i-1})/dy_{i-1} and
    # dA(x_i,y_{i-1})/dx_i for all i, with invalid index values giving derivatives equal to 1.
    op_bwd_vmap = torch.vmap(op_bwd, out_dims=(0, 0))
    op_bwd_x, op_bwd_y = op_bwd_vmap(inputs, outputs.roll(1, dim))
    # Set the i = 0 elements of the x derivative to 1
    torch.select(op_bwd_x, dim, 0).fill_(1.0)
    D_segprod = segprod(op_bwd_y, dim=dim, offset=1)
    # TODO: @goon - write this as a batched matmul
    inputs_grad = (D_segprod * outputs_grad.unsqueeze(dim)).sum(dim + 1) * op_bwd_x

    return inputs_grad


class TestAssocScanDerivatives:
    batch_size = 2
    d_model = 16
    dim = 1

    def test_segprod(self) -> None:
        torch.manual_seed(42)
        inputs = torch.randn(self.batch_size, self.d_model)
        inputs_segprod = segprod(inputs, dim=self.dim)
        for d in range(self.d_model):
            expected_cumprod = inputs[:, d:].cumprod(dim=self.dim)
            actual_cumprod = inputs_segprod[:, d, d:]
            torch.testing.assert_close(expected_cumprod, actual_cumprod)
            # Other elements are zeros:
            zeros = inputs_segprod[:, d, :d]
            torch.testing.assert_close(zeros, torch.zeros_like(zeros))

    def test_segprod_with_offset(self) -> None:
        torch.manual_seed(42)
        inputs = torch.randn(self.batch_size, self.d_model)
        for offset in range(1, self.d_model):
            inputs_segprod = segprod(inputs, dim=self.dim, offset=offset)
            for d in range(self.d_model):
                expected_cumprod = inputs[:, d + offset :].cumprod(dim=self.dim)
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
        outputs = inputs.cumsum(dim=self.dim)
        grad_out = torch.randn_like(outputs)

        # Populate grad on inputs
        grad_out = torch.randn_like(outputs)
        outputs.backward(grad_out)

        # Compute the same grads with our helper
        inputs_grad = get_scan_derivative(grad_out, outputs, inputs, sum_bwd, dim=self.dim)
        torch.testing.assert_close(inputs.grad, inputs_grad)

    def test_cumprod(self) -> None:
        torch.manual_seed(42)
        inputs = torch.randn(self.batch_size, self.d_model, requires_grad=True)
        outputs = inputs.cumprod(dim=self.dim)

        # Populate grad on inputs
        grad_out = torch.randn_like(outputs)
        outputs.backward(grad_out)

        outputs.backward(grad_out)
        # Compute the same grads with our helper
        inputs_grad = get_scan_derivative(grad_out, outputs, inputs, prod_bwd, dim=self.dim)
        torch.testing.assert_close(inputs.grad, inputs_grad)

    def test_cumsum_multi_dim(self) -> None:
        """
        Same as test_cumsumbut with additional dimensions.
        """
        torch.manual_seed(42)
        inputs = torch.randn(self.batch_size, self.d_model, self.d_model, requires_grad=True)
        outputs = inputs.cumsum(dim=self.dim)

        # Populate grad on inputs
        grad_out = torch.randn_like(outputs)
        outputs.backward(grad_out)

        # Compute the same grads with our helper
        inputs_grad = get_scan_derivative(grad_out, outputs, inputs, sum_bwd, dim=self.dim)
        torch.testing.assert_close(inputs.grad, inputs_grad)

    def test_cumprod_multi_dim(self) -> None:
        """
        Same as test_cumprod but with additional dimensions.
        """
        torch.manual_seed(42)
        inputs = torch.randn(self.batch_size, self.d_model, self.d_model, requires_grad=True)
        outputs = inputs.cumprod(dim=self.dim)

        # Populate grad on inputs
        grad_out = torch.randn_like(outputs)
        outputs.backward(grad_out)

        # Compute the same grads with our helper
        inputs_grad = get_scan_derivative(grad_out, outputs, inputs, prod_bwd, dim=self.dim)
        torch.testing.assert_close(inputs.grad, inputs_grad)
