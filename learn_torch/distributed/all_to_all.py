import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

from dtest import DTest


class TestAllToAll(DTest):
    dtype = torch.bfloat16

    def test_basic(self) -> None:
        t = torch.arange(
            self.rank * self.world_size, (self.rank + 1) * self.world_size, device=self.device
        )
        out = torch.empty_like(t)
        dist.all_to_all_single(out, t)
        torch.testing.assert_close(
            out, torch.arange(self.rank, self.world_size**2, self.world_size, device=self.device)
        )

    def test_permute(self) -> None:
        """
        Use the {input,output}_split_sizes args to pass tensors around.
        """
        t = torch.tensor([self.rank], device=self.device)
        out = torch.empty_like(t)
        # input_split_sizes determines where chunk boundaries on the inputs are and how many
        # elements to send to each rank: send_numel_to_rank[r] = input_split_sizes[r],
        # schematically, with
        # Must have sum(input_split_sizes) == inputs.shape[0].
        input_split_sizes = [
            t.shape[0] if ((rank + 1) % self.world_size) == self.rank else 0
            for rank in range(self.world_size)
        ]
        # Seems like output_split_sizes is the conjugate, determining how many elements should be
        # received from which ranks: numel_from_rank[r] = output_split_sizes[r], schematically.
        # Must have sum(output_split_sizes) == outputs.shape[0].
        output_split_sizes = [
            t.shape[0] if ((rank - 1) % self.world_size) == self.rank else 0
            for rank in range(self.world_size)
        ]
        dist.all_to_all_single(out, t, output_split_sizes, input_split_sizes)
        torch.testing.assert_close(
            out, torch.tensor([(self.rank + 1) % self.world_size], device=self.device)
        )

    def test_gather(self) -> None:
        """
        Build a gather to rank zero.
        """
        t = torch.tensor([self.rank], device=self.device, dtype=self.dtype)
        input_split_sizes = [t.shape[0]] + [0 for _ in range(self.world_size - 1)]
        if not self.rank:
            out = torch.empty(self.world_size, device=self.device, dtype=self.dtype)
            output_split_sizes = [t.shape[0] for _ in range(self.world_size)]
        else:
            out = torch.empty(0, device=self.device, dtype=self.dtype)
            output_split_sizes = [0 for _ in range(self.world_size)]
        dist.all_to_all_single(out, t, output_split_sizes, input_split_sizes)
        if not self.rank:
            torch.testing.assert_close(
                out, torch.arange(self.world_size, device=self.device, dtype=self.dtype)
            )
        else:
            assert not out.shape[0]

    def test_permute_autograd(self) -> None:
        """
        Use the {input,output}_split_sizes args to pass tensors around. Passes from higher rank to
        lower rank.
        """
        t = torch.tensor([self.rank], device=self.device)
        # input_split_sizes determines where chunk boundaries on the inputs are and how many
        # elements to send to each rank: send_numel_to_rank[r] = input_split_sizes[r],
        # schematically, with
        # Must have sum(input_split_sizes) == inputs.shape[0].
        input_split_sizes = [
            t.shape[0] if ((rank + 1) % self.world_size) == self.rank else 0
            for rank in range(self.world_size)
        ]
        # Seems like output_split_sizes is the conjugate, determining how many elements should be
        # received from which ranks: numel_from_rank[r] = output_split_sizes[r], schematically.
        # Must have sum(output_split_sizes) == outputs.shape[0].
        output_split_sizes = [
            t.shape[0] if ((rank - 1) % self.world_size) == self.rank else 0
            for rank in range(self.world_size)
        ]
        out = funcol.all_to_all_single_autograd(
            t, output_split_sizes, input_split_sizes, group=dist.group.WORLD
        )
        torch.testing.assert_close(
            out, torch.tensor([(self.rank + 1) % self.world_size], device=self.device)
        )
