from argparse import ArgumentParser

import torch
import torch.distributed as dist
from torch.nn.functional import scaled_dot_product_attention
from torch_cp_inference import torch_ring_attn_prefill

"""
torchrun --nproc-per_node 4 <path-to-this-file>
"""

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_kv_heads", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--d_head", type=int, default=64)
    args = parser.parse_args()
    dist.init_process_group("gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    try:
        torch.manual_seed(42)

        q = torch.randn(args.batch_size, args.n_heads, args.seqlen, args.d_head)
        k, v = torch.randn(args.batch_size, args.n_kv_heads, args.seqlen, 2 * args.d_head).chunk(
            2, dim=-1
        )

        # SDPA Baseline
        out_sdpa = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

        # Ring impl
        q_chunk = q.tensor_split(world_size, dim=2)[rank]
        k_chunk = k.tensor_split(world_size, dim=2)[rank]
        v_chunk = v.tensor_split(world_size, dim=2)[rank]

        compiled_ring_attn = torch.compile(torch_ring_attn_prefill, fullgraph=True)
        out_ring = compiled_ring_attn(q_chunk, k_chunk, v_chunk, is_causal=True)

        # Compare corresponding chunks
        out_sdpa_chunk = out_sdpa.chunk(world_size, dim=2)[rank]
        torch.testing.assert_close(out_sdpa_chunk, out_ring)
        print(f"Succeeded on {rank=}\n")
    finally:
        dist.destroy_process_group()
