import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    """Per-dimension running mean/std with Welford update."""
    
    def __init__(self, shape, eps: float = 1e-8, device: str = None):
        super().__init__()
        self.eps = eps
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        dev = device if device is not None else "cpu"
        self.register_buffer("mean", torch.zeros(*shape, dtype=torch.float32, device=dev))
        self.register_buffer("var", torch.ones(*shape, dtype=torch.float32, device=dev))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float32, device=dev))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """Update running statistics with new batch of data."""
        # x: (..., D) -> flatten batch to (N, D)
        x_f32 = x.detach().to(dtype=torch.float32)
        x_2d = x_f32.view(-1, x_f32.shape[-1])
        batch_count = x_2d.shape[0]
        if batch_count == 0:
            return
        
        batch_mean = x_2d.mean(dim=0)
        batch_var = x_2d.var(dim=0, unbiased=False)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total_count)

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * (self.count * batch_count / total_count)
        new_var = M2 / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var.clamp_min(self.eps))
        self.count.copy_(total_count)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using running statistics."""
        std = torch.sqrt(self.var + self.eps)
        return (x - self.mean) / std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize input using running statistics."""
        std = torch.sqrt(self.var + self.eps)
        return x * std + self.mean