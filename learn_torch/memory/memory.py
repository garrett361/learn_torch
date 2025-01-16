import torch
from typing import Any, Union


class CUDAMemContext:
    def __init__(
        self,
        use_GiB: bool = True,
        filter_patterns: tuple[str, ...] = ("allocated_bytes.all", "reserved_bytes.all"),
        mem_only: bool = True,
    ) -> None:
        # Ensure CUDA libraries are loaded:
        torch.cuda.current_blas_handle()

        self.use_GiB = use_GiB
        self.filter_patterns = filter_patterns
        self.mem_only = mem_only
        self.before: dict[str, Union[int, float]] = {}
        self.after: dict[str, Union[int, float]] = {}
        self.delta: dict[str, Union[int, float]] = {}

    def _get_mem_dict(self) -> dict[str, Union[int, float]]:
        mem_dict = torch.cuda.memory_stats()
        if self.filter_patterns:
            mem_dict = {
                k: v for k, v in mem_dict.items() if any(p in k for p in self.filter_patterns)
            }
        if self.mem_only:
            mem_dict = {k: v for k, v in mem_dict.items() if "bytes" in k}
        if self.use_GiB:
            mem_dict = {
                k.replace("bytes", "GiB"): v / 2**30 if "bytes" in k else v
                for k, v in mem_dict.items()
            }
        return mem_dict

    def __enter__(self) -> "CUDAMemContext":
        self.before = self._get_mem_dict()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        torch.cuda.synchronize()
        self.after = self._get_mem_dict()
        self.delta = {k: v - self.before[k] for k, v in self.after.items()}
