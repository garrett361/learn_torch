import os
from functools import cache

import torch


@cache
def get_rank() -> int:
    return int(os.getenv("RANK", 0))


@cache
def get_local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", 0))


@cache
def get_world_size() -> int:
    return int(os.getenv("WORLD_SIZE", 1))


@cache
def get_device_type() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@cache
def get_device() -> torch.device:
    return torch.device(f"{get_device_type()}:{get_local_rank()}")


@cache
def get_comms_backend() -> str:
    if torch.cuda.is_available():
        return "nccl"
    return "gloo"
