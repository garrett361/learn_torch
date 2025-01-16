from dtest import DTest
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as func_colls


class TestFuncColls(DTest):
    def test_all_reduce(self) -> None:
        inputs = torch.tensor(1, device=self.get_device())
        outputs = func_colls.all_reduce(inputs, reduceOp="sum", group=dist.group.WORLD)
        self.print_rank(f"{outputs=}")
        assert outputs is not None
        torch.testing.assert_close(
            outputs, self.get_world_size() * torch.ones_like(outputs)
        )


class TestCompile(DTest):
    def test_all_reduce(self) -> None:
        inputs = torch.tensor(1, device=self.get_device())

        def op(t):
            dist.all_reduce(t)
            return t

        compiled_op = torch.compile(op)
        outputs = compiled_op(inputs)
        self.print_rank(f"{outputs=}")

        torch.testing.assert_close(
            outputs, self.get_world_size() * torch.ones_like(outputs)
        )
