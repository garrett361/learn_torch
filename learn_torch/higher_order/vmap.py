from functools import partial
import torch


class TestSimpleVmap:
    d_model = 16
    batch_size = 2

    def test_pow(self) -> None:
        vmap = torch.vmap(partial(torch.pow, exponent=2))

        t = torch.randn(self.batch_size, self.d_model)
        out = vmap(t)
        torch.testing.assert_close(out, t.pow(2))

    def test_multi_output(self) -> None:
        def func(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return t, t**2

        vmap = torch.vmap(func, out_dims=(0, 0))

        t = torch.randn(self.batch_size, self.d_model)
        out, out_squared = vmap(t)
        torch.testing.assert_close(out, t)
        torch.testing.assert_close(out_squared, t.pow(2))
