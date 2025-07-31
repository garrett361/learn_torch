from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_to_all_single


def torch_attn_primitives(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns the softmax numerator, denominator, and max scale. These are primitive values which can
    be used to build normal softmax attention and ring attention.

    It is redundant to return both the denominator and max_score. They always appear in a particular
    combination together and so just one tensor can be returned instead of two. But, this is good
    enough for now. TODO: @goon - optimize.

    NOTE: if any of the below aten ops are supported, we can also build ring attention using them,
    but we assume they are not generally available:
    * aten._scaled_dot_product_flash_attention
    * aten._scaled_dot_product_efficient_attention
    * aten._scaled_dot_product_cudnn_attention
    See the native torch CP attn implementation: https://github.com/pytorch/pytorch/blob/e7cc42df58a86bee05944f6e80c535aa1d099443/torch/distributed/tensor/experimental/_attention.py?plain=1#L1

    NOTE: if the model uses RoPE, some care must also be taken that RoPE is properly applied.
    Namely, different CP ranks need to offset their seq idx positions appropriately.
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


def ring_send_recv(send_buffer: torch.Tensor) -> torch.Tensor:
    """
    Utility for passing the given buffer to the next rank and receiving from the previous one, in
    ring order.
    """
    # See also dist._functional_collectives.permute_tensor
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    input_split_sizes = [
        send_buffer.shape[0] if ((other_rank - 1) % world_size) == rank else 0
        for other_rank in range(world_size)
    ]
    output_split_sizes = [
        send_buffer.shape[0] if ((other_rank + 1) % world_size) == rank else 0
        for other_rank in range(world_size)
    ]
    return all_to_all_single(
        send_buffer.contiguous(),
        output_split_sizes,
        input_split_sizes,
        group=list(range(world_size)),
    )


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
    numerator, denominator, max_score = torch_attn_primitives(q, k, v, scale, is_causal)

    for idx in range(1, world_size):
        k = ring_send_recv(k)
        v = ring_send_recv(v)
        if is_causal and idx > rank:
            # TODO: @goon - torch compile complains that we didn't do anything with the k, v tensors
            continue

        # Update local results
        ring_numerator, ring_denominator, ring_max_score = torch_attn_primitives(
            q, k, v, scale, is_causal=False
        )
        new_max_score = torch.maximum(max_score, ring_max_score)
        numerator = (ring_max_score - new_max_score).exp() * ring_numerator + (
            max_score - new_max_score
        ).exp() * numerator
        denominator = (ring_max_score - new_max_score).exp() * ring_denominator + (
            max_score - new_max_score
        ).exp() * denominator

        max_score = new_max_score

    return numerator / denominator
