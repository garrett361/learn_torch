import torch


class CUDATimer:
    def __init__(self) -> None:
        self._start_events: list[torch.cuda.Event] = []
        self._end_events: list[torch.cuda.Event] = []

    def __enter__(self) -> "CUDATimer":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self._start_events.append(start)
        self._end_events.append(stop)
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self._end_events[-1].record()

    def __len__(self) -> int:
        return len(self._start_events)

    def get_time_list_s(self) -> list[float]:
        if not self._start_events:
            return [0.0]
        torch.cuda.synchronize()
        time_list_s = [
            start.elapsed_time(end) / 1e3
            for start, end in zip(self._start_events, self._end_events)
        ]
        return time_list_s

    def get_total_time_s(self) -> float:
        return sum(self.get_time_list_s())

    def get_mean_time_s(self) -> float:
        time_list_s = self.get_time_list_s()
        return sum(time_list_s) / len(time_list_s)

    def reset(self) -> None:
        self._start_events.clear()
        self._end_events.clear()
