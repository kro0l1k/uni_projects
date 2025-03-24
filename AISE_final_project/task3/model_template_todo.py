import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
    """
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # x.shape == [batch_size, in_channels, number of grid points]

        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        return torch.fft.irfft(out_ft, n=x.size(-1))



class FNOBlock(nn.Module):
    def __init__(self, modes, width):
        super().__init__()
        # TODO: Initialize the FNO block components
        # Should include:
        # - Spectral convolution
        # - Pointwise convolution
        # - Normalization (if needed)
        self.modes = modes
        self.width = width
        self.spectral_layer = SpectralConv1d(self.width, self.width, self.modes)
        self.conv_layer = nn.Conv1d(self.width, self.width, 1)

        self.activation = nn.ReLU()
    

    def forward(self, x):
        # TODO: Implement the FNO block forward pass
        # Remember to include skip connections
        x = self.spectral_layer(x)
        x = x + self.conv_layer(x)
        return self.activation(x)


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        # TODO: Initialize time embedding
        # Consider using positional encoding or a learnable embedding

        # NOTE: using a learnable time embedding: 
        # time as input is of size (batch_size, 1)
        self.linear1 = nn.Linear(1, 4 * embedding_dim)  # first linear layer: (batch_size, 1) --> (batch_size, 4 * embedding_dim)
        self.linear2 = nn.Linear(4 * embedding_dim, 2 * embedding_dim)  # second linear layer: (batch_size, 4 * embedding_dim) -> (batch_size, 2 * embedding_dim)
        self.linear3 = nn.Linear(2 * embedding_dim, embedding_dim) # third linear layer: (batch_size, 2 * embedding_dim) -> (batch_size, 1 * embedding_dim)
        # Add activation functions between layers
        self.activation = nn.SiLU()  # Also known as Swish, works well for embeddings

    def forward(self, t):
        # TODO: Implement time embedding
        # t shape: (batch_size, 1)
        # return shape: (batch_size, embedding_dim)
        x = self.activation(self.linear1(t))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        # return shape: (batch_size, embedding_dim)
        return x


class AllenCahnFNO(nn.Module):
    def __init__(self, modes=16, width=64, n_layers = 3):
        super().__init__()
        self.modes = modes
        self.width = width
        
        # TODO: Initialize model components
        # Consider:
        # - Epsilon embedding
        # - Time embedding
        # - Input/output layers
        # - FNO blocks

        # Time and epsilon embeddings
        self.time_embed = TimeEmbedding(embedding_dim=width)
        self.eps_embed = nn.Linear(1, width)
        
        # Initial processing of spatial input
        self.fc0 = nn.Linear(1, width)  # lift spatial dimension to width
        
        # FNO blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock(modes=modes, width=width) 
            for _ in range(n_layers)
        ])
        
        # Final output layer
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, eps, t):
        """
        Args:
            x: Initial condition (batch_size, x_size)
            eps: Epsilon values (batch_size, 1)
            t: Time points (batch_size, n_steps)
        Returns:
            Predictions at requested timepoints (batch_size, n_steps, x_size)
        """
        # TODO: Implement the full model forward pass
        # 1. Embed epsilon and time
        # 2. Process spatial information with FNO blocks
        # 3. Generate predictions for each timestep
        
        batch_size, x_size = x.shape
        n_steps = t.shape[1]
        
        # Embed epsilon
        eps_embedded = self.eps_embed(eps)  # (batch_size, width)
        
        # Process initial condition
        x = x.unsqueeze(-1)     # (batch_size, x_size, 1)
        x = self.fc0(x)         # (batch_size, x_size, width)
        x = x.permute(0, 2, 1)  # (batch_size, width, x_size)
        
        # Initialize output tensor
        out = torch.zeros(batch_size, n_steps, x_size, device=x.device)
        
        # Process each timestep
        for i in range(n_steps):
            # Embed current timestep
            t_i = t[:, i:i+1]  # (batch_size, 1)
            t_embedded = self.time_embed(t_i)  # (batch_size, width)
            
            # Add time and epsilon embeddings to spatial features
            x_t = x + t_embedded.unsqueeze(-1) + eps_embedded.unsqueeze(-1)
            
            # Apply FNO blocks
            for block in self.fno_blocks:
                x_t = block(x_t)
            
            # Project to output space
            x_t = x_t.permute(0, 2, 1)  # (batch_size, x_size, width)
            x_t = F.gelu(self.fc1(x_t))
            x_t = self.fc2(x_t)
            out[:, i] = x_t.squeeze(-1)
        
        return out

def get_loss_func():
    """
    TODO: Define custom loss function(s) for training
    Consider:
    - L2 loss on predictions
    - Physical constraints (energy, boundaries)
    - Gradient-based penalties
    """
    def loss_function(pred, target, eps):
            # L2 loss on predictions
            mse_loss = F.mse_loss(pred, target)
            
            # Add physical constraints
            # Gradient-based penalty for smoothness
            eps = eps.unsqueeze(1).unsqueeze(1)
            print("eps shape: ", eps.shape)
            print(" shape of the gradients: ", torch.gradient(pred, dim=-1)[0].shape)
            print("shape of predictions: ", pred.shape)
            spatial_gradients = torch.gradient(pred, dim=-1)[0]
            smoothness_loss = torch.mean(spatial_gradients**2)
            
            # Energy constraint for Allen-Cahn
            # The energy functional is ∫ (ε²/2)|∇u|² + (1/4)(u² - 1)² dx
            energy_density = (eps**2/2) * spatial_gradients**2 + 0.25 * (pred**2 - 1)**2
            energy_loss = torch.mean(energy_density)
            
            # Combine losses with weights
            total_loss = mse_loss + 0.1 * smoothness_loss + 0.01 * energy_loss
            return total_loss
    
    return loss_function

def get_optimizer(model):
    """
    TODO: Configure optimizer and learning rate schedule
    Consider:
    - Adam with appropriate learning rate
    - Learning rate schedule for curriculum
    """
    # Adam optimizer with learning rate schedule
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Learning rate scheduler with warm-up and cosine decay
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=100,  # adjust based on your training setup
        steps_per_epoch=100,  # adjust based on your dataset size
        pct_start=0.1,  # warm-up period
        anneal_strategy='cos'
    )
    
    return optimizer, scheduler

def train_step(model, batch, optimizer, loss_func):
    """
    TODO: Implement single training step
    1. Forward pass
    2. Loss computation
    3. Backward pass
    4. Optimizer step
    Return loss value
    """

    model.train()
    optimizer.zero_grad()
    
    # Unpack batch
    x, eps, t, target = batch
    
    # Forward pass
    pred = model(x, eps, t)
    
    # Compute loss
    loss = loss_func(pred, target, eps)
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping to prevent explosions
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Optimizer step
    optimizer.step()
    
    return loss.item()

def validation_step(model, batch, loss_func):
    """
    TODO: Implement single validation step
    Similar to train_step but without gradient updates
    Return loss value
    """
    model.eval()
    
    with torch.no_grad():
        # Unpack batch
        x, eps, t, target = batch
        
        # Forward pass
        pred = model(x, eps, t)
        
        # Compute loss
        loss = loss_func(pred, target, eps)
    
    return loss.item()

# Example usage:
if __name__ == "__main__":
     # Model initialization
    model = AllenCahnFNO(modes=16, width=64)
    
    # Get loss function and optimizer
    loss_func = get_loss_func()
    optimizer, scheduler = get_optimizer(model)
    
    # Sample data
    batch_size, x_size = 32, 128
    x = torch.randn(batch_size, x_size)
    eps = torch.randn(batch_size, 1)
    t = torch.linspace(0, 1, 4)[None].expand(batch_size, -1)
    target = torch.randn(batch_size, 4, x_size)  # Example target data
    
    # Create batch
    batch = (x, eps, t, target)
    
    # Training step
    train_loss = train_step(model, batch, optimizer, loss_func)
    print(f"Training loss: {train_loss}")
    
    # Validation step
    val_loss = validation_step(model, batch, loss_func)
    print(f"Validation loss: {val_loss}")
