import torch.utils._pytree as pytree
import torch


class TestPyTree:
    def test(self) -> None:
        tree = {idx: [torch.randn(16) for _ in range(4)] for idx in range(3)}
        leaves = pytree.tree_leaves(tree)
        assert len(leaves) == 12
        assert all(torch.is_tensor(t) for t in leaves)
        t_flat, spec = pytree.tree_flatten(tree)
        assert t_flat == leaves
        tree_1 = pytree.tree_unflatten(t_flat, spec)
        assert tree == tree_1
        leaves_1 = list(pytree.tree_iter(tree))
        assert leaves == leaves_1

        tree_zeros = pytree.tree_map(lambda t: torch.zeros_like(t), tree)
        tree_zeros_leaves = pytree.tree_leaves(tree_zeros)
        assert all(torch.allclose(t, torch.zeros_like(t)) for t in tree_zeros_leaves)
