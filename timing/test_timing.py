import torch
from timing import CudaTimer
from collections import defaultdict


class TestCudaTimer:
    def test_single_time(self) -> None:
        timer = CudaTimer()
        t = torch.randn(128, 128, device="cuda")

        with timer:
            for _ in range(100):
                t @ t
        print(f"{timer.get_time_list_s()=}")
        print(f"{timer.get_total_time_s()=}")
        print(f"{timer.get_mean_time_s()=}")

    def test_multiple_time(self) -> None:
        timer = CudaTimer()
        t = torch.randn(128, 128, device="cuda")

        steps = 100
        for _ in range(steps):
            with timer:
                t @ t
        assert len(timer.get_time_list_s()) == steps
        print(f"{timer.get_time_list_s()=}")
        print(f"{timer.get_total_time_s()=}")
        print(f"{timer.get_mean_time_s()=}")

    def test_reset(self) -> None:
        timer = CudaTimer()
        t = torch.randn(128, 128, device="cuda")

        with timer:
            t @ t

        timer.get_time_list_s()
        timer.get_total_time_s()
        timer.get_mean_time_s()

        timer.reset()
        assert timer.get_total_time_s() == 0.0
        assert timer.get_mean_time_s() == 0.0

        with timer:
            t @ t

        assert timer.get_total_time_s() >= 0.0
        assert timer.get_mean_time_s() >= 0.0

    def test_defaultdict_timer(self) -> None:
        timer_dict = defaultdict(lambda: CudaTimer())
        t = torch.randn(128, 128, device="cuda")

        for step in range(1, 10):
            for _ in range(step):
                with timer_dict[step]:
                    t @ t
        for k, v in timer_dict.items():
            assert k == len(v)

    def test_timing(self) -> None:
        timer = CudaTimer()
        # warmup
        t = torch.randn(128, 128, device="cuda")
        for step in range(1, 10):
            t @ t

        import time

        with timer:
            time.sleep(1)
        assert 0.99 <= timer.get_mean_time_s() <= 1.01
