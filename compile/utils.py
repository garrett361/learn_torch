from typing import Any, Callable

from torch.fx import GraphModule
from torch._functorch.aot_autograd import aot_module_simplified


class PrintGMBackend:
    """
    Simple backend for printing generated graph modules and tracking the number of them created.
    Based on: https://colab.research.google.com/drive/1Zh-Uo3TcTHD_MODELyYJF-LLo5rjlHVMtqvMdf?usp=sharing#scrollTo=UklMVs56u9j7

    """

    def __init__(self) -> None:
        self._gm_dict: dict[int, GraphModule] = {}

    def __call__(self, gm: GraphModule, sample_inputs: Any) -> Callable:
        print("Printing graph:")
        gm.print_readable()
        print(f"{sample_inputs=}")
        self._gm_dict[self.n_gms] = gm
        return gm.forward

    def get_aot_compiler(self) -> Callable:
        return lambda gm, sample_inputs: aot_module_simplified(gm, sample_inputs, self)

    def __getitem__(self, idx: int) -> GraphModule:
        return self._gm_dict[idx]

    @property
    def n_gms(self) -> int:
        return len(self._gm_dict)
