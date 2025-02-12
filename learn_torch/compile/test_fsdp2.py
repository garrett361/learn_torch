from dtest import DTest
import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard


def get_simple_linear_model(d_model: int, device: torch.device) -> nn.Module:
    model = nn.Sequential(
        *(nn.Linear(d_model, d_model, bias=False, device=device) for _ in range(3))
    )
    return model


class TestFSDP2(DTest):
    d_model = 128

    def test_basic(self) -> None:
        model = get_simple_linear_model(self.d_model, self.device)
        for lin in model.modules():
            if isinstance(lin, nn.Linear):
                nn.init.eye_(lin.weight)
                fully_shard(lin)
        fully_shard(model)
        inputs = torch.randn(1, self.d_model, device=self.device)
        outputs = model(inputs)
        self.print_rank(f"{outputs=}")
        torch.testing.assert_close(inputs, outputs)

    def test_basic_compile(self) -> None:
        model = get_simple_linear_model(self.d_model, self.device)
        for lin in model.modules():
            if isinstance(lin, nn.Linear):
                nn.init.eye_(lin.weight)
                fully_shard(lin)
        fully_shard(model)

        compiled_model = torch.compile(model)
        inputs = torch.randn(1, self.d_model, device=self.device)
        outputs = compiled_model(inputs)
        self.print_rank(f"{outputs=}")
        torch.testing.assert_close(inputs, outputs)
