import torch

class SimpleSystemReplayBuffer:
    """A minimal replay buffer for system dynamics training.

    Stores flattened (state, action, extension, contact, termination) tuples and
    returns contiguous windows for training the dynamics model.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        extension_dim: int,
        contact_dim: int,
        termination_dim: int,
        capacity: int,
        device: str = "cpu",
    ):
        self.device = device
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.extension_dim = extension_dim
        self.contact_dim = contact_dim
        self.termination_dim = termination_dim

        self.states = torch.zeros(capacity, state_dim, device=device)
        self.actions = torch.zeros(capacity, action_dim, device=device)
        self.extensions = (
            torch.zeros(capacity, extension_dim, device=device)
            if extension_dim > 0
            else None
        )
        self.contacts = (
            torch.zeros(capacity, contact_dim, device=device)
            if contact_dim > 0
            else None
        )
        self.terminations = (
            torch.zeros(capacity, termination_dim, device=device)
            if termination_dim > 0
            else None
        )

        self.idx = 0
        self.full = False

    @torch.no_grad()
    def insert(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        extension: torch.Tensor | None,
        contact: torch.Tensor | None,
        termination: torch.Tensor | None,
    ):
        """Insert a single transition (batched over envs is flattened outside)."""
        n = state.shape[0]
        assert state.shape[1] == self.state_dim
        assert action.shape[1] == self.action_dim

        for i in range(n):
            self.states[self.idx].copy_(state[i])
            self.actions[self.idx].copy_(action[i])
            if self.extensions is not None and extension is not None:
                self.extensions[self.idx].copy_(extension[i])
            if self.contacts is not None and contact is not None:
                self.contacts[self.idx].copy_(contact[i])
            if self.terminations is not None and termination is not None:
                self.terminations[self.idx].copy_(termination[i])

            self.idx = (self.idx + 1) % self.capacity
            if self.idx == 0:
                self.full = True

    def _num_valid(self) -> int:
        return self.capacity if self.full else self.idx

    @torch.no_grad()
    def mini_batch_generator(
        self,
        sequence_length: int,
        num_mini_batches: int,
        mini_batch_size: int,
    ):
        """Yield mini-batches of shape [B, T, *] for dynamics training."""
        num_valid = self._num_valid()
        if num_valid <= sequence_length:
            return

        # We sample start indices of contiguous windows.
        max_start = num_valid - sequence_length
        total_windows = max_start
        if total_windows <= 0:
            return

        indices = torch.randint(
            low=0,
            high=max_start,
            size=(num_mini_batches * mini_batch_size,),
            device=self.device,
        )

        for i in range(num_mini_batches):
            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size
            batch_idx = indices[start:end]

            # [B, T, *]
            state_batch = torch.stack(
                [self.states[j : j + sequence_length] for j in batch_idx], dim=0
            )
            action_batch = torch.stack(
                [self.actions[j : j + sequence_length] for j in batch_idx], dim=0
            )
            if self.extensions is not None:
                extension_batch = torch.stack(
                    [self.extensions[j : j + sequence_length] for j in batch_idx], dim=0
                )
            else:
                extension_batch = None
            if self.contacts is not None:
                contact_batch = torch.stack(
                    [self.contacts[j : j + sequence_length] for j in batch_idx], dim=0
                )
            else:
                contact_batch = None
            if self.terminations is not None:
                termination_batch = torch.stack(
                    [self.terminations[j : j + sequence_length] for j in batch_idx],
                    dim=0,
                )
            else:
                termination_batch = None

            yield (
                state_batch,
                action_batch,
                extension_batch,
                contact_batch,
                termination_batch,
            )