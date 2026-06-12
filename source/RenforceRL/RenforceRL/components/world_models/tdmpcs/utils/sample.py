import torch

def _sample_action_sequences(self, B: int, popsize: int, horizon: int, device: torch.device, mean=None, std=None):
    """
    Return shape: (B, popsize, horizon, action_dim)
    mean, std optional: (B, action_dim) broadcasted as initial proposal mean/std
    """
    if mean is None:
        mean = torch.zeros((B, self.action_dim), device=device)
    if std is None:
        std = torch.ones((B, self.action_dim), device=device) * 0.5

    # sample gaussian noises and form sequences
    # shape -> (B, popsize, horizon, action_dim)
    eps = torch.randn((B, popsize, horizon, self.action_dim), device=device)
    mean = mean.view(B, 1, 1, self.action_dim)
    std = std.view(B, 1, 1, self.action_dim)
    return mean + eps * std