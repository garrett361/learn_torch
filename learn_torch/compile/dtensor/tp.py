import pytest
import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

from dtest import DTest


class MLP(nn.Module):
    def __init__(self, device, d_model: int = 32) -> None:
        super().__init__()
        self.d_model = d_model
        self.lin0 = nn.Linear(self.d_model, self.d_model, bias=False, device=device)
        self.lin1 = nn.Linear(self.d_model, self.d_model, bias=False, device=device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.lin1(self.lin0(inputs).relu())


class TestTP(DTest):
    def test(self) -> None:
        torch.manual_seed(42)
        model = MLP(self.device)
        tp_mesh = init_device_mesh("cuda", (self.world_size,))
        model = parallelize_module(
            model, tp_mesh, {"lin0": ColwiseParallel(), "lin1": RowwiseParallel()}
        )
        inputs = torch.randn(1, model.d_model)
        outputs = model(inputs)

    @pytest.mark.world_size(2)
    @pytest.mark.cpu
    def test_compiled(self, world_size: int) -> None:
        torch.manual_seed(42)
        model = MLP(self.device)
        tp_mesh = init_device_mesh(self.device_type, (self.world_size,))
        model = parallelize_module(
            model, tp_mesh, {"lin0": ColwiseParallel(), "lin1": RowwiseParallel()}
        )
        model_compiled = torch.compile(model, backend="inductor", fullgraph=True)
        inputs = torch.randn(1, model.d_model, device=self.device)
        outputs = model_compiled(inputs)
