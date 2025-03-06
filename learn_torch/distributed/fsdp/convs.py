from dtest import DTest
import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard


class Conv1dStack(nn.Module):
    def __init__(self, n_layers: int, *args, **kwargs) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv1d(*args, **kwargs) for _ in range(n_layers)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = inputs
        for conv in self.layers:
            out = conv(out)
        return out


class TestConv1DFSD2(DTest):
    C_out = C_in = 4
    kernel_size = 5
    padding = kernel_size // 2
    assert kernel_size - 2 * padding == 1, (
        "kernel size and padding width do not preserve input shape"
    )
    n_layers = 3
    batch_size = 2

    @property
    def d_model(self) -> int:
        return 4 * self.world_size * self.kernel_size

    def test_basic(self) -> None:
        model = Conv1dStack(
            self.n_layers, self.C_in, self.C_out, self.kernel_size, self.padding, device=self.device
        )
        for m in model.modules():
            if isinstance(m, nn.Conv1d):
                fully_shard(m)
        fully_shard(model)
        inputs = torch.randn(self.batch_size, self.C_in, self.d_model, device=self.device)
        outputs = model(inputs)
        outputs = model(inputs)
        outputs
