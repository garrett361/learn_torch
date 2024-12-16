import torch
import argparse
import torch.nn as nn

"""
Extremely minimal script for testing various torch compile options
"""


# class BasicModel(nn.Module):
#     def __init__(self, d_model: int, device: torch.device) -> None:
#         super().__init__()
#         self.d_model = d_model
#         self.lin0 = nn.Linear(d_model, d_model, bias=False, device=device)
#         self.lin1 = nn.Linear(d_model, d_model, bias=False, device=device)
#
#     def forward(self, inputs: torch.Tensor) -> torch.Tensor:
#         outputs = self.lin1(self.lin0(inputs).relu())
#         return outputs


class BasicModel(nn.Module):
    def __init__(self, d_model: int, device: torch.device) -> None:
        super().__init__()
        self.d_model = d_model
        self.lin0 = nn.Linear(d_model, 2 * d_model, bias=False, device=device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.lin0(inputs).relu()
        return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(f"{'cuda' if torch.cuda.is_available() else 'cpu'}:0")
    model = BasicModel(d_model=args.d_model, device=device)
    compiled_model = torch.compile(model)
    inputs = torch.randn(1, args.d_model, device=device)

    outputs = compiled_model(inputs)
    print(f"{outputs=}")
    print(f"{outputs.shape=}")
