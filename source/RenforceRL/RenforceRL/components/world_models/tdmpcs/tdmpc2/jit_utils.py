from typing import Tuple
import torch

@torch.jit.script
def _update_mean_std(
    seq: torch.Tensor,        # (B,P,H,A)
    scores: torch.Tensor,     # (B,P)
    beta: float,
    eps: float,
    min_std: float,
    max_std: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s_max = scores.max(dim=1, keepdim=True).values
    logits = beta * (scores - s_max)
    w = torch.exp(logits)
    w = w / (w.sum(dim=1, keepdim=True) + 1e-12)  # (B,P)
    B, P, H, A = seq.shape
    w4 = w.view(B, P, 1, 1)
    new_mean = (w4 * seq).sum(dim=1)
    new_var = (w4 * (seq * seq)).sum(dim=1) - new_mean * new_mean
    new_std = torch.sqrt(torch.clamp(new_var, 0.0) + eps)
    new_std = new_std.clamp(min_std, max_std)
    return new_mean, new_std, w