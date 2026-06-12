import torch
from .vecenv_wrapper import RenforceRLLabEnvWrapper
from .dynamic_env_wrapper import RFDynamicEnvWrapper
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from RenforceRL.components.world_models.system_dynamics.system_dynamics_mlp import SystemDynamicsMLP

class RFImagineEnvWrapper(RFDynamicEnvWrapper):
    """Wrapper for imagination rollouts using a learned system dynamics model.
    
    This wrapper extends RFDynamicEnvWrapper to support:
    - Extracting system observations (state, action, etc.) from the environment
    - Converting state/action history into policy observations for imagination
    - Stepping the imagination environment using the system dynamics model
    """
    
    system_dynamic_model: Optional["SystemDynamicsMLP"] = None
    _last_action: Optional[torch.Tensor] = None
    
    def __init__(self, env, clip_actions=None):
        super().__init__(env, clip_actions)
        self._last_action = None
    
    def set_system_dynamics(self, dynamic_model: "SystemDynamicsMLP"):
        """Set the system dynamics model for imagination rollouts.
        
        Args:
            dynamic_model: The learned system dynamics model (SystemDynamicsMLP)
        """
        self.system_dynamic_model = dynamic_model
        if dynamic_model is not None:
            self.system_dynamic_model.eval()
    
    def get_system_observation(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract system observations from the current environment state.
        
        Policy observation structure: [Commands, Dynamics(with noise), Last Action]
        We extract the Dynamics part for system dynamics prediction.
        
        Returns:
            system_state: [num_envs, dynamic_dim] - Dynamic state for system dynamics (extracted from policy obs)
            system_action: [num_envs, action_dim] - Last action taken
            system_extension: Optional[torch.Tensor] - Extension signals (if available)
            system_contact: Optional[torch.Tensor] - Contact signals (if available)
            system_termination: Optional[torch.Tensor] - Termination signals (if available)
        """
        # Get current policy observation
        obs_dict = {}
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        
        policy_obs = obs_dict["policy"]  # [num_envs, policy_dim]
        
        # Calculate dimensions
        command_dim = self.commad_shape  # Note: typo in parent class (commad_shape)
        dynamic_dim = self.dim_params["dynamic_dim"]
        action_dim = self.dim_params["action_dim"]
        
        # Extract Dynamics part from policy observation: [Commands, Dynamics, Last Action]
        # Dynamics starts at command_dim and has length dynamic_dim
        system_state = policy_obs[:, command_dim:command_dim + dynamic_dim]
        
        # Get last action from policy observation (last action_dim elements)
        # Or use stored action if available
        if self._last_action is not None:
            system_action = self._last_action
        else:
            # Extract from policy observation (last action_dim elements)
            system_action = policy_obs[:, -action_dim:]
        
        # Extract optional auxiliary signals
        system_extension = None
        system_contact = None
        system_termination = None
        
        return system_state, system_action, system_extension, system_contact, system_termination
    
    def get_imagination_observation(
        self,
        state_history: torch.Tensor,
        action_history: torch.Tensor,
        command_history: torch.Tensor = None
    ) -> torch.Tensor:
        """Convert state/action history into policy observations for imagination.
        
        Policy observation structure: [Commands, Dynamics(with noise), Last Action]
        We construct this from:
        - Commands: from command_history or current commands
        - Dynamics: from state_history (last state)
        - Last Action: from action_history (last action)
        
        Args:
            state_history: [num_envs, history_horizon, dynamic_dim] - History of system states (Dynamics part)
            action_history: [num_envs, history_horizon, action_dim] - History of actions
            command_history: Optional[torch.Tensor] - History of commands [num_envs, history_horizon, command_dim]
        
        Returns:
            imagination_obs: [num_envs, policy_dim] - Policy observations for imagination
        """
        # Calculate dimensions
        command_dim = self.commad_shape
        dynamic_dim = self.dim_params["dynamic_dim"]
        action_dim = self.dim_params["action_dim"]
        policy_dim = self.dim_params["policy_dim"]
        
        # Extract current Dynamics from state history (last state)
        current_dynamics = state_history[:, -1]  # [num_envs, dynamic_dim]
        
        # Extract last action from action history
        last_action = action_history[:, -1]  # [num_envs, action_dim]
        
        # Get current commands
        if command_history is not None:
            current_commands = command_history[:, -1]  # [num_envs, command_dim]
        else:
            # Get current commands from environment
            current_commands = self.get_commands()  # [num_envs, command_dim]
        
        # Construct policy observation: [Commands, Dynamics, Last Action]
        imagination_obs = torch.cat([current_commands, current_dynamics, last_action], dim=-1)
        
        # Verify dimension matches
        assert imagination_obs.shape[-1] == policy_dim, (
            f"Constructed observation dimension {imagination_obs.shape[-1]} "
            f"does not match policy_dim {policy_dim}. "
            f"command_dim={command_dim}, dynamic_dim={dynamic_dim}, action_dim={action_dim}"
        )
        
        return imagination_obs
    
    def imagination_step(
        self,
        actions: torch.Tensor,
        state_history: torch.Tensor,
        action_history: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict, torch.Tensor, torch.Tensor]:
        """Step the imagination environment using the system dynamics model.
        
        Args:
            actions: [num_envs, action_dim] - Actions to take in imagination
            state_history: [num_envs, history_horizon, dynamic_dim] - Current state history (Dynamics part)
            action_history: [num_envs, history_horizon, action_dim] - Current action history
        
        Returns:
            imagination_obs_next: [num_envs, policy_dim] - Next policy observations
            imagination_rewards: [num_envs, 1] - Predicted rewards
            imagination_dones: [num_envs, 1] - Predicted termination signals
            imagination_extras: dict - Additional information (observations dict)
            updated_state_history: [num_envs, history_horizon, dynamic_dim] - Updated state history
            updated_action_history: [num_envs, history_horizon, action_dim] - Updated action history
        """
        assert self.system_dynamic_model is not None, "System dynamics model not set. Call set_system_dynamics() first."
        history_horizon = state_history.shape[1]

        # Prepare input sequences for the system dynamics model
        # The model expects [B, T, dynamic_dim] and [B, T, action_dim]
        # Get current state (last state in history) - this is the Dynamics part
        current_state = state_history[:, -1:]  # [num_envs, 1, dynamic_dim]
        current_action = actions.unsqueeze(1)  # [num_envs, 1, action_dim]
        
        # Concatenate to form sequences
        state_seq = torch.cat([state_history, current_state], dim=1)  # [num_envs, history_horizon+1, dynamic_dim]
        action_seq = torch.cat([action_history, current_action], dim=1)  # [num_envs, history_horizon+1, action_dim]
        
        # Predict next Dynamics state using system dynamics model
        with torch.no_grad():
            next_dynamics_pred, extension_pred, contact_pred, termination_pred = self.system_dynamic_model(
                state_seq, action_seq
            )
        
        # Update state and action histories
        # Keep only the last history_horizon steps
        updated_state_history = torch.cat([state_history[:, 1:], next_dynamics_pred.unsqueeze(1)], dim=1)
        updated_action_history = torch.cat([action_history[:, 1:], current_action], dim=1)
        
        # Convert next Dynamics state to policy observation
        # Note: We need commands for the next observation, but in imagination we might keep the same commands
        # or sample new ones. For now, we'll use the current commands (they don't change in imagination)
        imagination_obs_next = self.get_imagination_observation(
            updated_state_history, updated_action_history
        )
        
        # Compute rewards from predicted states
        # Parse predicted states and compute reward terms
        parsed_states = self._parse_imagination_states(next_dynamics_pred)
        parsed_extensions = self._parse_extensions(extension_pred) if extension_pred is not None else None
        parsed_contacts = self._parse_contacts(contact_pred) if contact_pred is not None else None
        
        # Compute reward terms (implemented by subclasses or use default)
        self._compute_imagination_reward_terms(parsed_states, actions, parsed_extensions, parsed_contacts)
        
        # Get final rewards (implemented by subclasses)
        imagination_rewards = self._post_imagination_reward_step()
        
        # Ensure rewards have correct shape [num_envs, 1]
        if imagination_rewards.dim() == 1:
            imagination_rewards = imagination_rewards.unsqueeze(-1)
        
        # Predict termination
        if termination_pred is not None:
            # Convert termination prediction to done signal
            # termination_pred might be logits, so apply sigmoid and threshold
            if termination_pred.dim() > 1 and termination_pred.shape[-1] > 1:
                # Multi-dimensional termination, take mean or max
                termination_prob = torch.sigmoid(termination_pred).mean(dim=-1, keepdim=True)
            else:
                termination_prob = torch.sigmoid(termination_pred)
            imagination_dones = (termination_prob > 0.5).to(dtype=torch.long)
        else:
            # No termination prediction, assume no termination
            imagination_dones = torch.zeros(
                (actions.shape[0], 1),
                device=next_dynamics_pred.device,
                dtype=torch.long
            )
        
        # Prepare extras dict (similar to real environment step)
        imagination_extras = {
            "observations": {
                "policy": imagination_obs_next,
                "critic": imagination_obs_next,  # Assume same as policy for now
            },
            "termination": imagination_dones.squeeze(-1) if imagination_dones.shape[-1] == 1 else imagination_dones,
        }
        
        return (
            imagination_obs_next,
            imagination_rewards,
            imagination_dones,
            imagination_extras,
            updated_state_history,
            updated_action_history,
        )
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Override step to track last action for system observation extraction."""
        # Store action before stepping
        self._last_action = actions.clone()
        
        # Call parent step
        obs, reward, done, extras = super().step(actions)
        return obs, reward, done, extras
    
    def _parse_imagination_states(self, imagination_states: torch.Tensor) -> dict:
        """Parse predicted states into named components.
        
        This is a placeholder that should be overridden by subclasses.
        The default implementation assumes the state is already in the correct format.
        
        Args:
            imagination_states: [num_envs, dynamic_dim] - Predicted dynamics state
        
        Returns:
            dict: Parsed state components (e.g., {"base_lin_vel": ..., "joint_pos": ...})
        """
        # Default: return as-is (subclasses should override)
        return {"dynamics": imagination_states}
    
    def _parse_extensions(self, extensions: torch.Tensor) -> Optional[dict]:
        """Parse extension signals.
        
        Args:
            extensions: [num_envs, extension_dim] or None
        
        Returns:
            dict or None: Parsed extension components
        """
        if extensions is None:
            return None
        return {"extensions": extensions}
    
    def _parse_contacts(self, contacts: torch.Tensor) -> Optional[dict]:
        """Parse contact signals.
        
        Args:
            contacts: [num_envs, contact_dim] or None
        
        Returns:
            dict or None: Parsed contact components (e.g., {"foot_contact": ..., "thigh_contact": ...})
        """
        if contacts is None:
            return None
        # Default: apply sigmoid and round to get binary contacts
        contacts_binary = torch.sigmoid(contacts).round()
        return {"contacts": contacts_binary}
    
    def _compute_imagination_reward_terms(
        self,
        parsed_states: dict,
        actions: torch.Tensor,
        parsed_extensions: Optional[dict] = None,
        parsed_contacts: Optional[dict] = None,
    ):
        """Compute reward terms from predicted states.
        
        This method should be overridden by subclasses to implement environment-specific
        reward computation. The default implementation returns zero rewards.
        
        Args:
            parsed_states: dict - Parsed state components from _parse_imagination_states
            actions: [num_envs, action_dim] - Actions taken
            parsed_extensions: Optional dict - Parsed extension signals
            parsed_contacts: Optional dict - Parsed contact signals
        
        Note:
            Subclasses should store reward terms in self.imagination_reward_per_step
            for use in _post_imagination_reward_step()
        """
        # Default: zero rewards (subclasses should override)
        num_envs = actions.shape[0]
        self.imagination_reward_per_step = {
            "reward": torch.zeros(num_envs, device=actions.device)
        }
    
    def _post_imagination_reward_step(self) -> torch.Tensor:
        """Combine reward terms into final reward.
        
        This method combines the reward terms computed in _compute_imagination_reward_terms
        using the environment's reward manager weights (if available).
        
        Returns:
            rewards: [num_envs, 1] - Final reward values
        """
        # Default: sum all reward terms (subclasses should override to use reward_manager)
        if hasattr(self, "imagination_reward_per_step") and self.imagination_reward_per_step:
            # Sum all reward terms
            reward_values = list(self.imagination_reward_per_step.values())
            if reward_values and isinstance(reward_values[0], torch.Tensor):
                total_reward = sum(reward_values)
                # Ensure correct shape [num_envs, 1]
                if total_reward.dim() == 1:
                    return total_reward.unsqueeze(-1)
                elif total_reward.dim() == 0:
                    # Scalar, need to know num_envs
                    num_envs = self.num_envs if hasattr(self, "num_envs") else 1
                    return total_reward.unsqueeze(0).unsqueeze(-1).expand(num_envs, 1)
                else:
                    return total_reward
        
        # Fallback: zero rewards
        num_envs = self.num_envs if hasattr(self, "num_envs") else 1
        device = next(self.system_dynamic_model.parameters()).device if self.system_dynamic_model is not None else "cpu"
        return torch.zeros((num_envs, 1), device=device)
    
    """
    We assume that the policy observation is contructed as [Commands, Dynamics(with noise), Last Action]
    """
    
    @property
    def dim_params(self):
        dim_params = {
            "policy_dim": self.observation_space["policy"].shape[-1],
            "critic_dim": self.observation_space.get("critic", self.observation_space["policy"]).shape[-1],
            "dynamic_dim": self.observation_space.get("dynamic", self.observation_space["policy"]).shape[-1],
            "action_dim": self.action_space.shape[-1],
            "rewards_dim": self.rewards_shape,
        }
        return dim_params