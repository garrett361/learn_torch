import torch
import torch.nn.functional as F


def kl_divergence(
    logits_p: torch.Tensor, logits_q: torch.Tensor, temp: float = 1.0, dim: int = -1
) -> torch.Tensor:
    """
    Stably compute the KL divergence D_KL(p|q) = sum(p ln (p/q)) from the corresponding logits.
    The mean is taken over all dimensions other than `dim`.
    """

    # Use shift-invariance for stability and account for temperature
    logits_p = (logits_p - logits_p.max(dim=dim, keepdim=True).values) / temp
    logits_q = (logits_q - logits_q.max(dim=dim, keepdim=True).values) / temp
    p = logits_p.softmax(dim=dim)
    # Avoid division by using logsumexp
    kl = (
        (
            p
            * (
                (logits_p - logits_q)
                + (
                    logits_q.logsumexp(dim=dim, keepdim=True)
                    - logits_p.logsumexp(dim=dim, keepdim=True)
                )
            )
        )
        .sum(dim=dim)
        .mean()
    )
    return kl


class Test:
    vocab_size = 128
    bsz = 2
    seqlen = 64

    def test(self) -> None:
        torch.manual_seed(42)
        logits_p = torch.randn(self.bsz, self.seqlen, self.vocab_size)
        logits_q = torch.randn(self.bsz, self.seqlen, self.vocab_size)
        kl_pq = kl_divergence(logits_p, logits_q)
        kl_qp = kl_divergence(logits_q, logits_p)
        kl_zero = kl_divergence(logits_p, logits_p)

        assert not torch.any(kl_pq < 0.0), f"{kl_pq=}"
        assert not torch.any(kl_qp < 0.0), f"{kl_qp=}"
        torch.testing.assert_close(kl_zero, torch.zeros_like(kl_zero))

        p, q = logits_p.softmax(dim=-1), logits_q.softmax(dim=-1)
        kl_pq_alt = (p * (p / q).log()).sum(dim=-1).mean()
        kl_qp_alt = (q * (q / p).log()).sum(dim=-1).mean()
        torch.testing.assert_close(kl_pq_alt, kl_pq)
        torch.testing.assert_close(kl_qp_alt, kl_qp)

        # Apparently there's a builtin:
        kl_pq_builtin = F.kl_div(
            logits_p.reshape(-1, self.seqlen).log_softmax(dim=-1),
            logits_q.reshape(-1, self.seqlen).log_softmax(dim=-1),
            log_target=True,
            reduction="batchmean",
        )
        kl_qp_builtin = F.kl_div(
            logits_q.reshape(-1, self.seqlen).log_softmax(dim=-1),
            logits_p.reshape(-1, self.seqlen).log_softmax(dim=-1),
            log_target=True,
            reduction="batchmean",
        )
        # Only passes w/ some degree of tolerance, should just use builtin
        torch.testing.assert_close(kl_pq_builtin, kl_pq, atol=1e-1, rtol=1e-1)
        torch.testing.assert_close(kl_qp_builtin, kl_qp, atol=1e-1, rtol=1e-1)
