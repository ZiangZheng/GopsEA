import torch
import torch.nn as nn
import torch.nn.functional as F

from RenforceRL.networks.transformers.attention_blocks import \
    get_subsequent_mask_with_batch_length

def get_valid_mask_from_termination(termination: torch.Tensor) -> torch.Tensor:
    """
    Generates a sequence mask (is_valid) based on termination flags using vectorization.
    Steps after the termination point in a sequence are marked as invalid (False).

    Args:
        termination (torch.Tensor): Termination flags, shape (B, T, 1) or (B, T).
    
    Returns:
        torch.Tensor: A boolean mask tensor, shape (B, T, 1).
    """
    # Squeeze the last dimension if shape is (B, T, 1)
    if termination.dim() == 3 and termination.shape[-1] == 1:
        termination = termination.squeeze(-1)
    # Ensure termination is boolean
    if termination.dtype != torch.bool:
        termination = (termination > 0.5)
    # 1. Accumulate termination status: True from the first termination point onwards. (B, T)
    is_terminated_accumulated = torch.cumsum(termination.float(), dim=1) > 0
    # 2. Shift the accumulated status one step to the right. 
    #    (B, T) -> (B, T+1). Value=0 ensures the first step is always valid.
    #    This tensor indicates if the *previous* step was terminal or already past termination.
    terminated_prev_step = F.pad(is_terminated_accumulated[:, :-1], (1, 0), value=0)
    
    # 3. The mask is the logical NOT of the shifted status.
    #    If the previous step was terminated, the current step is invalid (False).
    is_valid_mask = ~terminated_prev_step
    # 4. Return with the channel dimension (B, T, 1)
    return is_valid_mask.unsqueeze(-1)