import torch
from typing import Any, Union


class CUDAMemContext:
    def __init__(self, use_gib: bool = True) -> None:
        # Ensure CUDA libraries are loaded:
        torch.cuda.current_blas_handle()

        self.use_gib = use_gib
        self.before: dict[str, Union[int, float]] = {}
        self.after: dict[str, Union[int, float]] = {}
        self.delta: dict[str, Union[int, float]] = {}

    def _get_mem_dict(self) -> dict[str, Union[int, float]]:
        return {
            k: v if not self.use_gib else v / 2**30 for k, v in torch.cuda.memory_stats().items()
        }

    def __enter__(self) -> "CUDAMemContext":
        self.before = self._get_mem_dict()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.after = self._get_mem_dict()
        self.delta = {k: v - self.before[k] for k, v in self.after.items()}
