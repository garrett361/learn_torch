import torch


class TestGather:
    """
    torch.gather is basically just fancy slicing.
    """

    batch_size = 2
    dim = 4

    def test_identity(self) -> None:
        t = torch.stack([torch.arange(self.dim) for _ in range(self.batch_size)], dim=0)
        idx = torch.randint(self.dim, size=(self.batch_size, 2))
        out = t.gather(dim=1, index=idx)
        torch.testing.assert_close(out, idx)
