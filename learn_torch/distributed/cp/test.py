import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention
from torch_cp_inference import torch_attn, torch_ring_attn_prefill

from dtest._dtest import DTest


class TestSDPAEquality:
    @pytest.mark.parametrize("is_causal", [True, False])
    def test_no_gqa(self, is_causal: bool) -> None:
        torch.manual_seed(42)
        batch_size = 2
        seqlen = 128
        n_heads = 4
        d_head = 64
        q, k, v = torch.randn(batch_size, n_heads, seqlen, 3 * d_head).chunk(3, dim=-1)
        out = torch_attn(q, k, v, is_causal=is_causal)
        out_sdpa = scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        torch.testing.assert_close(out, out_sdpa)

    @pytest.mark.parametrize("is_causal", [True, False])
    def test_gqa(self, is_causal: bool) -> None:
        torch.manual_seed(42)
        batch_size = 2
        seqlen = 128
        n_heads = 4
        n_kv_heads = 2
        d_head = 64
        q = torch.randn(batch_size, n_heads, seqlen, d_head)
        k, v = torch.randn(batch_size, n_kv_heads, seqlen, 2 * d_head).chunk(2, dim=-1)
        out = torch_attn(q, k, v, is_causal=is_causal)
        out_sdpa = scaled_dot_product_attention(q, k, v, is_causal=is_causal, enable_gqa=True)
        torch.testing.assert_close(out, out_sdpa)


class TestRingAttn(DTest):
    batch_size = 2
    seqlen = 128
    n_heads = 4
    n_kv_heads = 2
    d_head = 64

    @pytest.mark.cpu
    @pytest.mark.world_size(4)
    def test(self) -> None:
        torch.manual_seed(42)
        q = torch.randn(self.batch_size, self.n_heads, self.seqlen, self.d_head, device=self.device)
        k, v = torch.randn(
            self.batch_size, self.n_kv_heads, self.seqlen, 2 * self.d_head, device=self.device
        ).chunk(2, dim=-1)

        # SDPA Basedine
        out_sdpa = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

        # Ring impl
        q_chunk = q.tensor_split(self.world_size, dim=2)[self.rank]
        k_chunk = k.tensor_split(self.world_size, dim=2)[self.rank]
        v_chunk = v.tensor_split(self.world_size, dim=2)[self.rank]

        out_ring = torch_ring_attn_prefill(q_chunk, k_chunk, v_chunk, is_causal=True)

        # Compare corresponding chunks
        out_sdpa_chunk = out_sdpa.chunk(self.world_size, dim=2)[self.rank]
        torch.testing.assert_close(out_sdpa_chunk, out_ring)

    @pytest.mark.cpu
    @pytest.mark.world_size(4)
    def test_compiled(self) -> None:
        torch.manual_seed(42)
        q = torch.randn(self.batch_size, self.n_heads, self.seqlen, self.d_head, device=self.device)
        k, v = torch.randn(
            self.batch_size, self.n_kv_heads, self.seqlen, 2 * self.d_head, device=self.device
        ).chunk(2, dim=-1)

        # SDPA Basedine
        out_sdpa = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

        # Ring impl
        q_chunk = q.tensor_split(self.world_size, dim=2)[self.rank]
        k_chunk = k.tensor_split(self.world_size, dim=2)[self.rank]
        v_chunk = v.tensor_split(self.world_size, dim=2)[self.rank]

        compiled_ring_attn = torch.compile(torch_ring_attn_prefill, fullgraph=True)
        out_ring = compiled_ring_attn(q_chunk, k_chunk, v_chunk, is_causal=True)

        # Compare corresponding chunks
        out_sdpa_chunk = out_sdpa.chunk(self.world_size, dim=2)[self.rank]
        torch.testing.assert_close(out_sdpa_chunk, out_ring)
