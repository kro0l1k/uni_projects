import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters for the Allen-Cahn equation
epsilon = 2  # Interface width parameter
dt = 0.1       # Time step
dx = 1.0       # Spatial step
T = 20         # Total simulation time
L = 10        # Domain size

# Initialize the 1D grid
x = np.linspace(-L/2, L/2, L)

# Initial condition: random perturbation around 0
np.random.seed(42)
#u = 0.1 * np.random.randn(L)

# initial condition: sin(2πx) function

u = np.sin(1*np.pi*x/L)

# Function to compute one time step of Allen-Cahn equation in 1D
def allen_cahn_step(u):
    # Compute Laplacian using finite differences
    laplacian = (np.roll(u, 1) + np.roll(u, -1) - 2*u) / (dx**2)
    
    # Compute the reaction term (u - u^3)
    reaction = u - u**3
    
    # Update u using forward Euler method
    u_new = u + dt * (epsilon**2 * laplacian + reaction)
    return u_new

# Set up the figure for animation
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(x, u, 'b-', lw=2)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_title('Allen-Cahn Equation Evolution (1D)')
ax.grid(True)

# Add time text
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Animation update function
def update(frame):
    global u
    u = allen_cahn_step(u)
    line.set_ydata(u)
    current_time = frame * dt
    time_text.set_text(f't = {current_time:.1f}')
    return [line, time_text]

# Create animation
anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
plt.show()

