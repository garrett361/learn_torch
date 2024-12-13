import torch
import utils
from torch._dynamo.exc import Unsupported
import pytest


class TestBreaks:
    def test_no_break(self) -> None:
        def op(t: torch.Tensor) -> torch.Tensor:
            return t**2

        compiled_op = torch.compile(op, fullgraph=True)
        inputs = torch.randn(1, device=utils.get_device())
        out = compiled_op(inputs)
        torch.testing.assert_close(out, inputs**2)

    def test_break(self) -> None:
        def square(t: torch.Tensor) -> torch.Tensor:
            out = t**2
            print(out)
            return out

        compiled = torch.compile(square, fullgraph=True)
        inputs = torch.randn(1, device=utils.get_device())
        with pytest.raises(Unsupported):
            out = compiled(inputs)
