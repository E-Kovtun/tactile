"""Local per-rank, single-view collapse diagnostics; natural-log entropies."""
import torch


@torch.no_grad()
def collapse_metrics(probs, embeddings):
    probs = probs.detach().float()
    probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-30)
    marginal = probs.mean(0)
    logp = probs.clamp_min(1e-30).log()
    entropy = -(probs * logp).sum(-1).mean()
    marginal_entropy = -(marginal * marginal.clamp_min(1e-30).log()).sum()
    # Mean KL(p_i || marginal), computed directly to avoid entropy cancellation.
    js = (probs * (logp - marginal.clamp_min(1e-30).log())).sum(-1).mean()
    cls = embeddings.detach().float()
    centered = cls - cls.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    mass = singular.sum()
    weights = singular / mass.clamp_min(1e-30)
    effective_rank = torch.where(
        mass > 1e-12,
        (-(weights * weights.clamp_min(1e-30).log()).sum()).exp(),
        mass.new_zeros(()),
    )
    return {
        "sample_entropy": entropy,
        "marginal_entropy": marginal_entropy,
        "effective_prototypes": marginal_entropy.exp(),
        "prediction_js": js.clamp_min(0),
        "prediction_variance": probs.var(0, unbiased=False).mean(),
        "cls_variance": centered.square().mean(),
        "cls_effective_rank": effective_rank,
    }
