import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn.functional as F



def generate_fourier_ic(x, n_modes=5, seed=None):
    """TODO: Generate random Fourier series initial condition.
    Hints:
    1. Use random coefficients for sin and cos terms
    2. Ensure the result is normalized to [-1, 1]
    3. Consider using np.random.normal for coefficients
    """
    if seed is not None:
        np.random.seed(seed)
    
    # TODO: Generate coefficients for Fourier series
    sin_coeff = np.random.normal(size=n_modes)
    cos_coeff = np.random.normal(size=n_modes)
    
    # TODO: Compute the Fourier series
    u0 = np.zeros_like(x)
    for i in range(n_modes):
        u0 += sin_coeff[i] * np.sin((i+1) * np.pi * x)
        u0 += cos_coeff[i] * np.cos((i+1) * np.pi * x)
    # TODO: Normalize to [-1, 1]
    u0 = (u0 - u0.min()) / (u0.max() - u0.min())
    u0 = 2 * u0 - 1
    return u0
    

def generate_gmm_ic(x, n_components=None, seed=None):
    """TODO: Generate Gaussian mixture model initial condition.
    Hints:
    1. Random number of components if n_components is None
    2. Use random means, variances, and weights
    3. Ensure result is normalized to [-1, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    if n_components is None:
        n_components = np.random.randint(2, 6)
    
    # TODO: Generate means, variances, and weights
    means = np.random.uniform(-1, 1, size=n_components)
    variances = np.random.uniform(0.01, 0.1, size=n_components)
    weights = np.random.dirichlet(np.ones(n_components))
    
    # TODO: Compute GMM
    u0 = np.zeros_like(x)
    for i in range(n_components):
        u0 += weights[i] * np.exp(-(x - means[i])**2 / (2 * variances[i]))
        
    # TODO: Normalize to [-1, 1]
    u0 = (u0 - u0.min()) / (u0.max() - u0.min())
    u0 = 2 * u0 - 1
    return u0



def generate_piecewise_ic(x, n_pieces=None, seed=None):
    """
    Generate a piecewise linear initial condition.
    1. Generate random breakpoints and their function values.
    2. For x between breakpoints[i] and breakpoints[i+1],
       interpolate linearly between values[i] and values[i+1].
    3. Extend constant values to the left and right of the outer breakpoints.
    """
    if seed is not None:
        np.random.seed(seed)
    
    if n_pieces is None:
        n_pieces = np.random.randint(3, 7)
    
    # Generate breakpoints, sorted in ascending order
    breakpoints = np.sort(np.random.uniform(-1, 1, size=n_pieces))
    
    # Generate the function values at these breakpoints
    values = np.random.uniform(-1, 1, size=n_pieces)
    
    # Precompute slopes for each interval [breakpoints[i], breakpoints[i+1]]
    # slopes[i] = (values[i+1] - values[i]) / (breakpoints[i+1] - breakpoints[i])
    slopes = (values[1:] - values[:-1]) / (breakpoints[1:] - breakpoints[:-1])
    
    # 1) Find which interval each x belongs to via searchsorted
    #    - 'side="right"' means an x == breakpoint[i] will map to interval i (not i-1).
    #    - subtract 1 so that idx = i means breakpoints[i] <= x < breakpoints[i+1]
    idx = np.searchsorted(breakpoints, x, side='right') - 1
    
    # 2) Clip idx to ensure it's within [0, n_pieces-2] for valid interpolation
    #    We'll handle left-of-first and right-of-last breakpoints separately.
    idx_clipped = np.clip(idx, 0, n_pieces - 2)
    
    # 3) Piecewise linear interpolation
    #    For x in the interval [breakpoints[i], breakpoints[i+1]]:
    #    u0(x) = values[i] + slopes[i] * (x - breakpoints[i])
    #
    #    We'll fill in everything, then override left and right boundaries.
    u0 = values[idx_clipped] + slopes[idx_clipped] * (x - breakpoints[idx_clipped])
    
    # 4) Handle x < breakpoints[0] => constant = values[0]
    left_mask = x < breakpoints[0]
    u0[left_mask] = values[0]
    
    # 5) Handle x >= breakpoints[-1] => constant = values[-1]
    right_mask = x >= breakpoints[-1]
    u0[right_mask] = values[-1]
    
    return u0


def allen_cahn_rhs(t, u, epsilon, x_grid):
    """TODO: Implement Allen-Cahn equation RHS:
        ∂u/∂t = Δu - (1/ε²)(u³ - u)
    """
    dx = x_grid[1] - x_grid[0]
    
    # TODO: Compute Laplacian (Δu) with periodic boundary conditions
    laplacian = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / dx**2
    # TODO: Compute nonlinear term -(1/ε²)(u³ - u)
    nonlinear = -1 / epsilon**2 * (u**3 - u)
    
    # TODO: Return full RHS
    return laplacian + nonlinear

def generate_dataset(n_samples, epsilon, x_grid, t_eval, ic_type='fourier', seed=None):
    """Generate dataset for Allen-Cahn equation."""
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize dataset array
    dataset = np.zeros((n_samples, len(t_eval), len(x_grid)))
    
    # Generate samples
    for i in range(n_samples):
        # Generate initial condition based on type
        if ic_type == 'fourier':
            u0 = generate_fourier_ic(x_grid, seed=seed+i if seed else None)
        elif ic_type == 'gmm':
            u0 = generate_gmm_ic(x_grid, seed=seed+i if seed else None)
        elif ic_type == 'piecewise':
            u0 = generate_piecewise_ic(x_grid, seed=seed+i if seed else None)
        else:
            raise ValueError(f"Unknown IC type: {ic_type}")
        
        # Solve PDE using solve_ivp
        sol = solve_ivp(
            allen_cahn_rhs,
            t_span=(t_eval[0], t_eval[-1]),
            y0=u0,
            t_eval=t_eval,
            args=(epsilon, x_grid),
            method='RK45',
            rtol=1e-6,
            atol=1e-6
        )
        
        dataset[i] = sol.y.T
    
    return dataset

def main():
    """Generate all datasets."""
    # Set up spatial grid
    nx = 128
    x_grid = np.linspace(-1, 1, nx)
    
    # Set up temporal grid
    t_eval = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    
    # Parameters for datasets
    ic_types = ['fourier', 'gmm', 'piecewise']  # Different initial condition types
    epsilons = [0.1, 0.05, 0.02, 0.04]  # Different epsilon values
    n_train = 2   # Number of training samples per configuration
    n_test = 1     # Number of test samples
    base_seed = 42  # For reproducibility
    additional_val_epsilons = [0.03, 0.06]  # Additional epsilons for validation
    
    train_data = []
    train_epsilons = []
    val_data = []
    val_epsilons = []
    
    # Generate training datasets. want to end up with a mapping from eps to [n_samples * |ic_types|, |t_eval|, x_grid]
    for epsilon in epsilons:
        train_data_eps = []
        for ic_type in ic_types:
            train_data_eps.append(generate_dataset(n_train, epsilon, x_grid, t_eval, ic_type, seed=base_seed))
        train_epsilons.append([epsilon])
        train_data.append(np.concatenate(train_data_eps, axis=0))
        
        val_data_eps = []
        for ic in ic_types:
            val_data_eps.append(generate_dataset(n_test, epsilon, x_grid, t_eval, ic, seed=base_seed))
        val_epsilons.append([epsilon])
        val_data.append(np.concatenate(val_data_eps, axis=0))
            
    # Generate additional validation datasets
    for epsilon in additional_val_epsilons:
        val_data_eps = []
        for ic in ic_types:
            val_data_eps.append(generate_dataset(n_test, epsilon, x_grid, t_eval, ic, seed=base_seed))
        val_epsilons.append([epsilon])
        val_data.append(np.concatenate(val_data_eps, axis=0)) 

    print("train data shape: ", np.array(train_data).shape)
    print("train epsilon shape: ", np.array(train_epsilons).shape)
    
    # assert that there is one-to-one correspondence between data and epsilon
    assert len(train_data) == len(train_epsilons)
    assert len(val_data) == len(val_epsilons)
    
    # Save datasets as numpy files
    train_data = np.array(train_data)
    val_data = np.array(val_data)
    train_epsilons = np.array(train_epsilons)
    val_epsilons = np.array(val_epsilons)
    
    # Save datasets as numpy files
    np.save('train_data.npy', train_data, allow_pickle=True)
    np.save('train_epsilons.npy', train_epsilons, allow_pickle=True)
    print("train data shape: ", train_data.shape)
    print("train epsilon shape: ", train_epsilons.shape)

    
    np.save('val_data.npy', val_data, allow_pickle=True)
    np.save('val_epsilons.npy', val_epsilons, allow_pickle=True)
    print("val data shape: ", val_data.shape)
    print("val epsilon shape: ", val_epsilons.shape)

if __name__ == "__main__":
    main()