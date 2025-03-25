from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.functional import scaled_dot_product_attention


def torch_attn_primitives(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Returns the softmax numerator, denominator, and max scale. These are primitive values which can
    be used to build normal softmax attention and ring attention.
    """
    _, n_q_heads, seqlen, d_head = q.shape
    n_k_heads = k.shape[1]
    gqa_ratio, remainder = divmod(n_q_heads, n_k_heads)
    if remainder:
        raise ValueError(
            f"The number of q-heads must be divisible by the number of k-heads: {q.shape=}, {k.shape=}"
        )
    if gqa_ratio != 1:
        k = k.unsqueeze(2).expand(-1, -1, gqa_ratio, -1, -1).flatten(1, 2)
        v = v.unsqueeze(2).expand(-1, -1, gqa_ratio, -1, -1).flatten(1, 2)

    scale = scale or d_head ** (0.5)
    scores = (q @ k.transpose(-1, -2)) / scale
    if is_causal:
        mask = torch.ones(seqlen, seqlen, dtype=torch.bool, device=scores.device).triu_(1)
        scores.masked_fill_(mask[None], float("-inf"))
    max_score = scores.max(dim=-1, keepdim=True).values
    scores = (scores - max_score).exp()
    numerator = scores @ v
    denominator = scores.sum(dim=-1, keepdim=True)
    return numerator, denominator, max_score


def torch_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Torch-native attention on a single device.
    """
    numerator, denominator, _ = torch_attn_primitives(q, k, v, scale, is_causal)
    return numerator / denominator


class TestSDPAEquality:
    def test_causal(self) -> None:
        torch.manual_seed(42)
        batch_size = 2
        seqlen = 128
        n_heads = 4
        d_head = 64
        q, k, v = torch.randn(batch_size, n_heads, seqlen, 3 * d_head).chunk(3, dim=-1)
        out = torch_attn(q, k, v, is_causal=True)
        out_sdpa = scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.testing.assert_close(out, out_sdpa)

    def test_non_causal(self) -> None:
        torch.manual_seed(42)
        batch_size = 2
        seqlen = 128
        n_heads = 4
        d_head = 64
        q, k, v = torch.randn(batch_size, n_heads, seqlen, 3 * d_head).chunk(3, dim=-1)
        out = torch_attn(q, k, v, is_causal=False)
        out_sdpa = scaled_dot_product_attention(q, k, v, is_causal=False)
        torch.testing.assert_close(out, out_sdpa)

    def test_causal_gqa(self) -> None:
        torch.manual_seed(42)
        batch_size = 2
        seqlen = 128
        n_heads = 4
        n_kv_heads = 4
        d_head = 64
        q = torch.randn(batch_size, n_heads, seqlen, d_head)
        k, v = torch.randn(batch_size, n_kv_heads, seqlen, 2 * d_head).chunk(2, dim=-1)
        out = torch_attn(q, k, v, is_causal=True)
        out_sdpa = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        torch.testing.assert_close(out, out_sdpa)


def ring_send_recv(buffer: torch.Tensor) -> torch.Tensor:
    """
    Utility for passing the given buffer to the next rank and receiving from the previous one, in
    ring order.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    recv_buffer = torch.empty_like(buffer)
    dist.send(buffer.contiguous(), dst=(rank + 1) % world_size)
    dist.recv(recv_buffer, src=(rank - 1) % world_size)
    return recv_buffer


def torch_ring_attn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Naive ring attention over the global process group.  A better implementation would use a zig-zag
    sharding pattern. See https://github.com/zhuzilin/ring-flash-attention or this comment
    specifically: https://github.com/zhuzilin/ring-flash-attention/issues/2#issuecomment-2236746166

    k, v are communicated, as this is less communication than passing around q in the typical use
    case where the gqa factor > 2.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    for send_rank in ((r % world_size) for r in range(rank + 1, rank + world_size)):
        pass
