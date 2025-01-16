import torch
import numpy as np
import pytest
from torch._dynamo.exc import TorchRuntimeError
from torch.autograd import Function


def test_numpy_op() -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    d_model = 16

    # Register simple numpy_sin op
    @torch.library.custom_op("test_custom_op::numpy_sin", mutates_args=())
    def numpy_sin(t: torch.Tensor) -> torch.Tensor:
        t_np = np.sin(t.numpy(force=True))
        return torch.from_numpy(t_np).to(t)

    @numpy_sin.register_fake
    def _(t) -> torch.Tensor:
        return torch.empty_like(t)

    # Both setup_context and backward need to be traceable!

    # arg names are meaningful! inputs is a tuple
    def setup_context(ctx, inputs, output) -> None:
        ctx.t = inputs[0]

    def backward(ctx, grad) -> torch.Tensor:
        return t.cos() * grad

    numpy_sin.register_autograd(backward, setup_context=setup_context)

    t = torch.randn(d_model, device=device, requires_grad=True)
    t_clone = t.detach().clone()
    t_clone.requires_grad_()

    t_sin = numpy_sin(t)
    t_sin_clone = t_clone.sin()
    with torch.no_grad():
        torch.testing.assert_close(t_sin, t_sin_clone)

    t_sin.sum().backward()
    t_sin_clone.sum().backward()
    with torch.no_grad():
        torch.testing.assert_close(t.grad, t_clone.grad)


def test_numpy_op_compiled() -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    d_model = 16

    # Register simple numpy_sin op
    @torch.library.custom_op("test_custom_op::numpy_sin", mutates_args=())
    def numpy_sin(t: torch.Tensor) -> torch.Tensor:
        t_np = np.sin(t.numpy(force=True))
        return torch.from_numpy(t_np).to(t)

    @numpy_sin.register_fake
    def _(t) -> torch.Tensor:
        return torch.empty_like(t)

    # Both setup_context and backward need to be traceable!

    # arg names are meaningful! inputs is a tuple
    def setup_context(ctx, inputs, output) -> None:
        ctx.t = inputs[0]

    def backward(ctx, grad) -> torch.Tensor:
        return ctx.t.cos() * grad

    numpy_sin.register_autograd(backward, setup_context=setup_context)

    t = torch.randn(d_model, device=device, requires_grad=True)
    t_clone = t.detach().clone()
    t_clone.requires_grad_()

    t_sin = torch.compile(numpy_sin)(t)
    t_sin_clone = t_clone.sin()

    t_sin.sum().backward()
    t_sin_clone.sum().backward()

    with torch.no_grad():
        torch.testing.assert_close(t.grad, t_clone.grad)
        torch.testing.assert_close(t_sin, t_sin_clone)


def test_numpy_op_non_traceable_bwd_ctx() -> None:
    """
    What happens if we perform non-traceable np ops in the bwd/ctx?
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    d_model = 16

    # Register simple numpy_sin op
    @torch.library.custom_op("test_custom_op::numpy_sin", mutates_args=())
    def numpy_sin(t: torch.Tensor) -> torch.Tensor:
        t_np = np.sin(t.numpy(force=True))
        return torch.from_numpy(t_np).to(t)

    @numpy_sin.register_fake
    def _(t) -> torch.Tensor:
        return torch.empty_like(t)

    # Both setup_context and backward need to be traceable!

    # arg names are meaningful! inputs is a tuple
    def setup_context(ctx, inputs, output) -> None:
        ctx.t_np = inputs[0].numpy(force=True)

    def backward(ctx, grad) -> torch.Tensor:
        return torch.from_numpy(ctx.t_np).to(grad).cos() * grad

    numpy_sin.register_autograd(backward, setup_context=setup_context)

    t = torch.randn(d_model, device=device, requires_grad=True)
    t_clone = t.detach().clone()
    t_clone.requires_grad_()

    t_sin = numpy_sin(t)
    t_sin_clone = t_clone.sin()

    t_sin.sum().backward()
    t_sin_clone.sum().backward()

    with torch.no_grad():
        torch.testing.assert_close(t_sin, t_sin_clone)
        torch.testing.assert_close(t.grad, t_clone.grad)


def test_numpy_op_non_traceable_bwd_ctx_compiled() -> None:
    """
    What happens if we perform non-traceable np ops in the bwd/ctx?
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    d_model = 16

    # Register simple numpy_sin op
    @torch.library.custom_op("test_custom_op::numpy_sin", mutates_args=())
    def numpy_sin(t: torch.Tensor) -> torch.Tensor:
        t_np = np.sin(t.numpy(force=True))
        return torch.from_numpy(t_np).to(t)

    @numpy_sin.register_fake
    def _(t) -> torch.Tensor:
        return torch.empty_like(t)

    # Both setup_context and backward need to be traceable!

    # arg names are meaningful! inputs is a tuple
    def setup_context(ctx, inputs, output) -> None:
        ctx.t_np = inputs[0].numpy(force=True)

    def backward(ctx, grad) -> torch.Tensor:
        return torch.from_numpy(ctx.t_np).to(grad).cos() * grad

    numpy_sin.register_autograd(backward, setup_context=setup_context)

    t = torch.randn(d_model, device=device, requires_grad=True)
    t_clone = t.detach().clone()
    t_clone.requires_grad_()

    """
    Raises:

    torch._dynamo.exc.TorchRuntimeError: Failed running call_function
    test_custom_op.numpy_sin.default(*(FakeTensor( ..., device='cuda:0', size=(16,), requir es_grad=True),), **{}):
    .numpy() is not supported for tensor subclasses.
    """
    with pytest.raises(TorchRuntimeError):
        t_sin = torch.compile(numpy_sin)(t)
        t_sin_clone = t_clone.sin()

        t_sin.sum().backward()
        t_sin_clone.sum().backward()

        with torch.no_grad():
            torch.testing.assert_close(t_sin, t_sin_clone)
            torch.testing.assert_close(t.grad, t_clone.grad)


def test_register_Function_wrapper() -> None:
    """
    Can register a wrapper around an autograd Function?
    """
    device = torch.device(
        "meta"
    )  # torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    d_model = 16

    class NoOp(Function):
        @staticmethod
        def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
            # ctx not used, just testing
            ctx.save_for_backward(*inputs)
            # Need a clone, or non-trivial mutates_args
            return inputs.clone()

        @staticmethod
        def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
            # ctx not used, just testing
            inputs = ctx.saved_tensors
            return grad

    # Register simple numpy_sin op
    @torch.library.custom_op("test_custom_op::no_op", mutates_args=())
    def no_op(t: torch.Tensor) -> torch.Tensor:
        return NoOp.apply(t)

    @no_op.register_fake
    def _(t) -> torch.Tensor:
        return torch.empty_like(t)

    def setup_context(ctx, inputs, output) -> None:
        pass

    def backward(ctx, grad) -> torch.Tensor:
        return NoOp.backward(ctx, grad)

    no_op.register_autograd(backward, setup_context=setup_context)
    t = torch.randn(d_model, device=device, requires_grad=True)
    t_no_op = torch.compile(no_op)(t)
    t_no_op.sum().backward()
