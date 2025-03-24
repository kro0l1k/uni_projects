import numpy as np

import numpy as np
import torch


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

def generate_piecewise_ic_old(x, n_pieces=None, seed=None):
    """TODO: Generate piecewise linear initial condition.
    Hints:
    1. Generate random breakpoints
    2. Create piecewise linear function
    3. Add occasional discontinuities
    """
    if seed is not None:
        np.random.seed(seed)
    
    if n_pieces is None:
        n_pieces = np.random.randint(3, 7)
    
    # TODO: Generate breakpoints
    breakpoints = np.sort(np.random.uniform(-1, 1, size=n_pieces))
    # TODO: Generate values at breakpoints
    values = np.random.uniform(-1, 1, size=n_pieces)
    
    # TODO: Create piecewise linear function
    u0 = np.zeros_like(x)
    for i in range(n_pieces):
        u0 += values[i] * (x >= breakpoints[i])
        
    return u0

# compare the two functions

# seed = 42   
# n_modes = 5
# x = np.linspace(0, 1, 100)
# u0_old = generate_piecewise_ic_old(x, n_modes, seed)
# u0 = generate_piecewise_ic(x, n_modes, seed)


# # plot the old and new with different colors
# import matplotlib.pyplot as plt

# plt.plot(x, u0_old, label='Old Fourier IC', marker='o')
# # use x to mark the new function
# plt.plot(x, u0, label='New Fourier IC', marker='x')
# plt.legend()
# plt.show()
    
    
    
# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("mps")

    # Create a Tensor directly on the mps device
    x = torch.ones(5, device=mps_device)
    # Or
    x = torch.ones(5, device="mps")

    # Any operation happens on the GPU
    y = x * 2

    