from dtest import DTest
import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard


class TestFSDP2(DTest):
    d_model = 128

    def test_basic(self) -> None:
        model = nn.Sequential(
            *(
                nn.Linear(
                    self.d_model, self.d_model, bias=False, device=self.get_device()
                )
                for _ in range(3)
            )
        )
        for lin in model.modules():
            if isinstance(lin, nn.Linear):
                nn.init.eye_(lin.weight)
                fully_shard(lin)
        fully_shard(model)
        inputs = torch.randn(1, self.d_model, device=self.get_device())
        outputs = model(inputs)
        self.print_rank(f"{outputs=}")
        torch.testing.assert_close(inputs, outputs)
