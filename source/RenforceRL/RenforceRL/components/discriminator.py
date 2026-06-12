import torch
import torch.nn as nn

class DynamicsDiscriminator(nn.Module):
    def __init__(self, state_dim, action_dim, lr=1e-4):
        # Initialize the parent class (nn.Module)
        super(DynamicsDiscriminator, self).__init__()
        
        # Input dimension is the concatenation of two states and the action dimension
        input_dim = state_dim * 2 + action_dim
        
        # State Encoder: Encodes the state into a hidden representation
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Action Encoder: Encodes the action into a hidden representation
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Dynamics Processor: Processes the concatenated encodings of state, action, and next state
        self.dynamics_processor = nn.Sequential(
            nn.Linear(256 + 128 + 256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            
            nn.Linear(64, 2),
            nn.Softmax(dim=1)  # Output probabilities for real or fake
        )
        
        # Initialize the optimizer and loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        
    def forward(self, state, action, next_state):
        # Encode state, action, and next state
        state_encoded = self.state_encoder(state)
        action_encoded = self.action_encoder(action)
        next_state_encoded = self.state_encoder(next_state)
        
        # Concatenate the encodings and process them through the dynamics processor
        combined = torch.cat([state_encoded, action_encoded, next_state_encoded], dim=1)
        return self.dynamics_processor(combined)
    
    def update_step(self, real_batch, fake_batch):
        # Zero the gradients from the previous step
        self.optimizer.zero_grad()
        
        # Unpack the real and fake data batches
        real_state, real_action, real_next_state = real_batch
        fake_state, fake_action, fake_next_state = fake_batch
        
        # Get the batch size and device for processing
        batch_size = real_state.shape[0]
        device = real_state.device
        
        # Combine the real and fake data for training
        mixed_states = torch.cat([real_state, fake_state], dim=0)
        mixed_actions = torch.cat([real_action, fake_action], dim=0)
        mixed_next_states = torch.cat([real_next_state, fake_next_state], dim=0)
        
        # Create labels for real and fake data (one-hot encoded)
        real_labels = torch.zeros((batch_size, 2), device=device)
        real_labels[:, 0] = 1  # [1,0] for real
        fake_labels = torch.zeros((batch_size, 2), device=device)
        fake_labels[:, 1] = 1  # [0,1] for fake
        mixed_labels = torch.cat([real_labels, fake_labels], dim=0)
        
        # Generate random permutation for shuffling
        perm = torch.randperm(2 * batch_size, device=device)
        
        # Shuffle the mixed data and labels
        mixed_states = mixed_states[perm]
        mixed_actions = mixed_actions[perm]
        mixed_next_states = mixed_next_states[perm]
        mixed_labels = mixed_labels[perm]
        
        # Forward pass through the model
        mixed_pred = self.forward(mixed_states, mixed_actions, mixed_next_states)
        
        # Compute loss
        loss = self.criterion(mixed_pred, mixed_labels)
        
        # Backpropagation and optimizer step
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy for real and fake samples
        real_mask = mixed_labels[:, 0] == 1  # Real samples
        fake_mask = mixed_labels[:, 1] == 1  # Fake samples
        
        real_acc = (mixed_pred[real_mask].argmax(dim=1) == 0).float().mean()
        fake_acc = (mixed_pred[fake_mask].argmax(dim=1) == 1).float().mean()
        
        # Return loss and accuracies
        return {
            'DynamicsDiscriminator/total_loss': loss.item(),
            'DynamicsDiscriminator/real_accuracy': real_acc.item(),
            'DynamicsDiscriminator/fake_accuracy': fake_acc.item(),
            'DynamicsDiscriminator/avg_accuracy': (real_acc + fake_acc).item() / 2
        }

class StateDiscriminator(nn.Module):
    def __init__(self, state_dim):
        # Initialize the state discriminator class
        super(StateDiscriminator, self).__init__()
        
        # State Encoder: Encodes state into a hidden representation
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Discriminator: Classifies the state as real or fake
        self.discriminator = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 2),  # Output 2 classes: real or fake
            nn.Softmax(dim=1)
        )

    def forward(self, state):
        # Encode the state and classify it
        encoded = self.state_encoder(state)
        return self.discriminator(encoded)
