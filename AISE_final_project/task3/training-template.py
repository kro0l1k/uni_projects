import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class AllenCahnDataset(Dataset):
    def __init__(self, data, epsilon_values, time_points):
        """
        data: dictionary mapping epsilon values to numpy arrays of shape (n_samples, n_timesteps, n_points)
        epsilon_values: list of epsilon values
        time_points: numpy array of time points
        """
        self.data = data
        self.epsilon_values = epsilon_values
        self.time_points = time_points
        
        # Create index mapping
        self.indices = []
        for eps in epsilon_values:
            n_samples = len(data[eps])
            self.indices.extend([(eps, i) for i in range(n_samples)])
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        eps, sample_idx = self.indices[idx]
        trajectory = self.data[eps][sample_idx]
        
        return {
            'initial': torch.FloatTensor(trajectory[0]),
            'target': torch.FloatTensor(trajectory[1:]),
            'epsilon': torch.FloatTensor([eps]),
            'times': torch.FloatTensor(self.time_points[1:])
        }

def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """Run one epoch of training."""
    model.train()
    total_loss = 0
    n_batches = len(train_loader)
    
    for batch in train_loader:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(batch['initial'], batch['epsilon'], batch['times'])
        
        # Compute loss with physical constraints
        loss = loss_fn(pred, batch['target'], batch['epsilon'])
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent explosions
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / n_batches

def validate(model, val_loader, loss_fn, device):
    """Run validation."""
    model.eval()
    total_loss = 0
    n_batches = len(val_loader)
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(batch['initial'], batch['epsilon'], batch['times'])
            loss = loss_fn(pred, batch['target'], batch['epsilon'])
            total_loss += loss.item()
    
    return total_loss / n_batches

def test_model(model, test_loader, loss_fn, device):
    """Evaluate model on test set and compute metrics."""
    model.eval()
    total_loss = 0
    n_batches = len(test_loader)
    
    # Additional metrics
    mse_by_epsilon = {}
    relative_error_by_epsilon = {}
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(batch['initial'], batch['epsilon'], batch['times'])
            
            # Compute standard loss
            loss = loss_fn(pred, batch['target'], batch['epsilon'])
            total_loss += loss.item()
            
            # Compute MSE and relative error for each epsilon
            eps = batch['epsilon'].item()
            if eps not in mse_by_epsilon:
                mse_by_epsilon[eps] = []
                relative_error_by_epsilon[eps] = []
            
            mse = torch.mean((pred - batch['target'])**2).item()
            rel_error = torch.mean(torch.abs(pred - batch['target']) / 
                                 (torch.abs(batch['target']) + 1e-6)).item()
            
            mse_by_epsilon[eps].append(mse)
            relative_error_by_epsilon[eps].append(rel_error)
    
    # Average metrics
    metrics = {
        'test_loss': total_loss / n_batches,
        'mse_by_epsilon': {eps: np.mean(values) for eps, values in mse_by_epsilon.items()},
        'relative_error_by_epsilon': {eps: np.mean(values) 
                                    for eps, values in relative_error_by_epsilon.items()}
    }
    
    return metrics

def train_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn,
                n_epochs, device, save_path='best_model.pt', curriculum_steps=None):
    """Complete training loop with curriculum learning."""
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # Update curriculum if needed
        if curriculum_steps:
            for step_epoch, eps_subset in curriculum_steps:
                if epoch == step_epoch:
                    print(f"Curriculum update: now training on epsilon values {eps_subset}")
                    # Update dataloaders here if needed
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        train_losses.append(train_loss)
        
        # Validation
        val_loss = validate(model, val_loader, loss_fn, device)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss
            }, save_path)
        
        # Print progress
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print("-"*50)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }

def load_and_test(model, test_loader, loss_fn, device, model_path='best_model.pt'):
    """Load best model and evaluate on test set."""
    # Load best model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test model
    metrics = test_model(model, test_loader, loss_fn, device)
    
    # Print results
    print("\nTest Results:")
    print(f"Overall Test Loss: {metrics['test_loss']:.6f}")
    print("\nMSE by epsilon:")
    for eps, mse in metrics['mse_by_epsilon'].items():
        print(f"ε = {eps:.3f}: {mse:.6f}")
    print("\nRelative Error by epsilon:")
    for eps, rel_error in metrics['relative_error_by_epsilon'].items():
        print(f"ε = {eps:.3f}: {rel_error:.6f}")
    
    return metrics

# Example curriculum steps
curriculum_steps = [
    (0, [0.1]),           # Start with largest epsilon
    (20, [0.1, 0.05]),    # Add medium epsilon
    (40, [0.1, 0.05, 0.02])  # Add smallest epsilon
]

