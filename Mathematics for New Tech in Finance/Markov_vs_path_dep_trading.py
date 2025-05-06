import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import iisignature


# Set default tensor type to float32 to run on a macbook
torch.set_default_dtype(torch.float32)
# Check for available devices
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
# Set the random seed for reproducibility
torch.manual_seed(42)

#### definition of models ####
class BlackScholesModel:
    """
    Black–Scholes / Geometric Brownian Motion model:
        dS_t = mu * S_t * dt + sigma * S_t * dW_t
    """
    def __init__(self, mu: float,
                 sigma: float,
                 D: int = 1):
        """
        Initialize the Black–Scholes model parameters.

        Parameters:
        ----------
        mu : float
            Drift coefficient.
        sigma : float
            Volatility coefficient.
        D : int, optional
            Number of assets (default: 1).
            The first asset is the risk-free asset, and the rest are risky assets.
        """
        self.mu = mu
        self.sigma = sigma

    def simulate(self,
                 S0: float,
                 finalTime: float,
                 T: int,
                 D: int,
                 S: int,
                 random_seed: int = None
                 ) -> np.ndarray:
        """
        Simulate sample paths using the Euler–Maruyama scheme.

        Parameters:
        ----------
        S0 : float
            Initial asset price.
        finalTime : float
            Total time horizon.
        T : int
            Number of time steps.
        D : int
            Number of assets: 0 is the risk-free asset, 1:D is the risky asset.
        S : int
            Number of trajectories to simulate.
        random_seed : int, optional
            Seed for reproducibility (default: None).

        Returns:
        -------
        paths : np.ndarray
            Simulated trajectories array of shape (S, N+1).
        """
        dt = finalTime / T
        if random_seed is not None:
            np.random.seed(random_seed)

        paths_STD = np.zeros((S, T + 1, D))
        paths_STD[:, 0, :] = S0

        for timestep in range(T):
            # Risk-free asset
            paths_STD[:, timestep + 1, 0] = (
                paths_STD[:, timestep, 0]
                * np.exp(RISK_FREE_RATE * dt)
            )
            # Risky assets

            for asset_nr in range(1, D):
                dW = np.sqrt(dt) * np.random.randn(S)
                S_prev = paths_STD[:, timestep, :]
                paths_STD[:, timestep + 1, asset_nr] = (
                    S_prev[:, asset_nr]
                    + self.mu * S_prev[:, asset_nr] * dt
                    + self.sigma * S_prev[:, asset_nr]  * dW
                )

        return paths_STD


class CEVModel:
    """
    Constant Elasticity of Variance (CEV) model:
        dS_t = mu * S_t * dt + sigma * S_t^gamma * dW_t
    """
    def __init__(self, mu: float, sigma: float, gamma: float, D: int = 1):
        """
        Initialize the CEV model parameters.

        Parameters:
        ----------
        mu : float
            Drift coefficient.
        sigma : float
            Volatility coefficient.
        gamma : float
            Elasticity parameter.
        D : int, optional
            Number of assets (default: 1).
            The first asset is the risk-free asset, and the rest are risky assets.
        """
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.D = D

    def simulate(self,
                 S0: float,
                 finalTime: float,
                 T: int,
                 D: int,
                 S: int,
                 random_seed: int = None
                 ) -> np.ndarray:
        """
        Simulate sample paths using the Euler–Maruyama scheme.

        Parameters:
        ----------
        S0 : float
            Initial asset price.
        finalTime : float
            Total time horizon.
        T : int
            Number of time steps.
        D : int
            Number of assets: 0 is the risk-free asset, 1:D is the risky asset.
        S : int
            Number of trajectories to simulate.
        random_seed : int, optional
            Seed for reproducibility (default: None).

        Returns:
        -------
        paths : np.ndarray
            Simulated trajectories array of shape (S, N+1).
        """
        dt = finalTime / T
        if random_seed is not None:
            np.random.seed(random_seed)

        paths_STD = np.zeros((S, T + 1, D))
        paths_STD[:, 0, :] = S0

        for time_index in range(T):
            # Risk-free asset
            paths_STD[:, time_index + 1, 0] = (
                paths_STD[:, time_index, 0]
                * np.exp(RISK_FREE_RATE * dt)
            )

            # Risky assets
            for asset_nr in range(1, D):
                dW = np.sqrt(dt) * np.random.randn(S)
                S_prev = paths_STD[:, time_index, :]

                paths_STD[:, time_index + 1, asset_nr] = (
                    S_prev[:, asset_nr]
                    + self.mu * S_prev[:, asset_nr] * dt
                    + self.sigma * (S_prev[:, asset_nr] ** self.gamma) * dW
                )
        return paths_STD


class HestonModel:
    """
    Heston stochastic volatility model:
        dS_t = mu * S_t * dt + sqrt(v_t) * S_t * dW_t^1
        dv_t = kappa*(theta - v_t) * dt + xi * sqrt(v_t) * dW_t^2
        Corr(dW^1, dW^2) = rho
    """
    def __init__(self,
                 mu: float,
                 kappa: float,
                 theta: float,
                 xi: float,
                 rho: float,
                 ):
        """
        Initialize the Heston model parameters.

        Parameters:
        ----------
        mu : float
            Drift coefficient for the asset.
        kappa : float
            Rate of mean reversion of variance.
        theta : float
            Long-term variance.
        xi : float
            Volatility of volatility.
        rho : float
            Correlation between asset and variance Brownian motions.
        """
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

    def simulate(self,
                 S0: float,
                 finalTime: float,
                 T: int,
                 D: int,
                 S: int,
                 random_seed: int = None
                 ) -> np.ndarray:
        """
        Simulate sample paths using the Euler–Maruyama scheme.

        Parameters:
        ----------
        S0 : float
            Initial asset price.
        finalTime : float
            Total time horizon.
        T : int
            Number of time steps.
        D : int
            Number of assets: 0 is the risk-free asset, 1:D is the risky asset.
        S : int
            Number of trajectories to simulate.
        random_seed : int, optional
            Seed for reproducibility (default: None).

        Returns:
        -------
        S_paths_STD : np.ndarray
            Simulated asset paths of shape (n_samples, N+1).
        v_paths_STD : np.ndarray
            Simulated variance paths of shape (n_samples, N+1).
        """
        dt = finalTime / T
        if random_seed is not None:
            np.random.seed(random_seed)

        paths_STD = np.zeros((S, T + 1, D))
        paths_STD[:, 0, :] = S0
        v0 = 0.04  # Initial variance
        v_paths_STD = np.zeros((S, T + 1, D))
        paths_STD[:, 0, :] = S0
        v_paths_STD[:, 0, : ] = v0

        for timestep in range(T):
            z1 = np.random.randn(S)
            z2 = np.random.randn(S)

            # Risk-free asset
            paths_STD[:, timestep + 1, 0] = (
                paths_STD[:, timestep, 0]
                * np.exp(RISK_FREE_RATE * dt)
            )
            # Risky assets
            for asset_nr in range(1, D):
                dW1 = np.sqrt(dt) * z1
                dW2 = np.sqrt(dt) * (self.rho * z1 + np.sqrt(1 - self.rho**2) * z2)

                v_prev = v_paths_STD[:, timestep, asset_nr]
                v_prev_clipped = np.maximum(v_prev, 0)
                v_paths_STD[:, timestep + 1, asset_nr] = (
                    v_prev
                    + self.kappa * (self.theta - v_prev) * dt
                    + self.xi * np.sqrt(v_prev_clipped) * dW2
                )

                S_prev = paths_STD[:, timestep, asset_nr]
                paths_STD[:, timestep + 1, asset_nr] = (
                    S_prev
                    + self.mu * S_prev * dt
                    + np.sqrt(np.maximum(v_paths_STD[:, timestep + 1, asset_nr], 0)) * S_prev * dW1
                )

        return paths_STD



# plot 5 sample paths from each model
def plot_sample_paths(paths_STD, title):
    plt.figure(figsize=(10, 6))
    print("examples of 5 samples from the model")
    # plot the 0th asset (risk-free asset)
    plt.plot(paths_STD[0, : , 0], lw=1.5, label='Risk-free Asset')
    for i in range(5):
        plt.plot(paths_STD[i, : , 1], lw=1.5)

    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Asset Price')
    plt.grid()
    plt.legend()
    plt.show()

### dataset generation ###

# Example usage:
T = 10 # nr of time steps
finalTime = 1.0
S0 = 100
RISK_FREE_RATE = 0.05
nr_of_stocks = 1 # only one stock for now
D = nr_of_stocks + 1 # risk-free asset + risky asset


S = 100


# TODO: maybe add a few out-of-distribution values to test generalization?
mu_values = np.random.uniform(0.03, 0.2, 10)
sigma_values = np.random.uniform(0.3, 0.5, 10)
gamma_values = np.random.uniform(0.3, 0.5, 10)

print(" the datasets are generated with the following parameters: ")
print(mu_values)
print(sigma_values)
print(gamma_values)
print("\n\n\n")

train_paths_BlackScholes = []
val_paths_BlackScholes = []

train_paths_CEV = []
val_paths_CEV = []

train_paths_Heston = []
val_paths_Heston = []

for mu, sigma, gamma in zip(mu_values, sigma_values, gamma_values):
    # model = HestonModel(mu, sigma, 0.04, sigma, 0.5)
    model_BlackScholes = BlackScholesModel(mu, sigma, D)
    paths = model_BlackScholes.simulate(S0, finalTime, T, D, S)

    # split into train, val and test
    train_paths_BlackScholes.append(paths[:int(0.8*S)])
    val_paths_BlackScholes.append(paths[int(0.8*S):])

    # CEV model
    model_CEV = CEVModel(mu, sigma, gamma, D)
    paths = model_CEV.simulate(S0, finalTime, T, D, S)
    # split into train, and val
    train_paths_CEV.append(paths[:int(0.8*S)])
    val_paths_CEV.append(paths[int(0.8*S):])

    # Heston model
    model_Heston = HestonModel(mu, kappa=sigma, theta=0.04, xi=gamma, rho=0.5)
    paths = model_Heston.simulate(S0, finalTime, T, D, S)
    # split into train, and val
    train_paths_Heston.append(paths[:int(0.8*S)])
    val_paths_Heston.append(paths[int(0.8*S):])

# convert to numpy arrays
train_paths_BlackScholes = np.concatenate(train_paths_BlackScholes, axis=0)
val_paths_BlackScholes = np.concatenate(val_paths_BlackScholes, axis=0)
train_paths_CEV = np.concatenate(train_paths_CEV, axis=0)
val_paths_CEV = np.concatenate(val_paths_CEV, axis=0)
train_paths_Heston = np.concatenate(train_paths_Heston, axis=0)
val_paths_Heston = np.concatenate(val_paths_Heston, axis=0)

# # plot sample paths (to verify if the greek parameters are sensible)
plot_sample_paths(train_paths_BlackScholes, 'training Sample Paths BlackScholes')
plot_sample_paths(train_paths_CEV, 'Training Sample Paths CEV')
plot_sample_paths(train_paths_Heston, 'Training Sample Paths Heston')
# plot_sample_paths(test_paths, 'Test Sample Paths')



###### Utility Function ######
def util(w, epsilon=1e-8):
    """
    Logarithmic utility function.

    Parameters:
        w (float): Wealth
        epsilon (float): Small value to avoid log(0)

    Returns:
        float: log(w + epsilon)
    """
    return torch.log(w + epsilon)

##### Trading Agents #####
class MarkovianAgent(nn.Module):
    def __init__(self, D ):
        super(MarkovianAgent, self).__init__()

        self.D = D
        self.wealth_embedding = nn.Linear(1, self.D)
        self.time_embedding = nn.Linear(1, self.D)


        # Simple network for demonstration
        self.fc1 = nn.Linear(self.D * 3,  self.D * 20)
        self.fc2 = nn.Linear(self.D * 20, self.D * 20)
        self.fc3 = nn.Linear(self.D * 20, self.D * 20)
        self.fc4 = nn.Linear(self.D * 20, self.D * 20)
        self.out = nn.Linear(self.D * 20, self.D)

    def forward(self, X_B1, S_BD, t_B1):
        """ network should take as input the last wealth (1 dim),
        the last stock prices (D + 1 dim),
        and the time (1 dim) and output
        a probability distribution over (d+1) available actions.

        Args:
            X_B1 (torch tensor B1):  wealth at time t
            S_BD (torch tensor BD):  stock prices at time t
                                    1 dim for the risk-free asset
                                    D dim for the risky assets
            t_B1 (torch tensor B1):  time at time t

        Returns:
            torch tensor: allocations for the next time step. they must sum to 1.
        """

        # embed the state and time
        X_B1 = self.wealth_embedding(X_B1)
        t_B1 = self.time_embedding(t_B1)

        # Concatenate the state and time embeddings
        x = torch.cat((X_B1, S_BD, t_B1), dim=1)
        # print(" after concatenation x shape: ", x.shape)
        x = torch.sigmoid(self.fc1(x))
        # print(" after fc1 x shape: ", x.shape)
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))


        x = self.out(x)


        # get the sum of the actions for every sample in the batch
        x = x / torch.sum(x, dim=1, keepdim=True)
        return x

class PathDependentAgent(nn.Module):
    """
    Path Dependent Agent

    This agent uses the entire history of wealth and stock prices up to time t
    to make a decision at time t. It interpolates both to a fixed length
    (10 * T), takes the last 10% as “recent history,” re‐interpolates that,
    then concatenates everything (including a time embedding) and flattens into
    an MLP.

    Args:
        D (int): Number of assets (1 risk-free + D-1 risky). Defaults to 1.
        T (int): Number of original time steps. Defaults to 100.
    """
    def __init__(self, D=1, T=100):
        super(PathDependentAgent, self).__init__()
        self.D = D
        self.T = T


        self.in_dim = 265
        h = D * 50               # hidden size
        self.fc1 = nn.Linear(self.in_dim, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, h)
        self.fc4 = nn.Linear(h, h)

        self.fc_out = nn.Linear(h, D)

    def forward(self, preprocessed_features):
        """
        preprocessed_features: [B, 265]
        """
        x = torch.sigmoid( self.fc1(preprocessed_features) )
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))

        x = self.fc_out(x)

        # normalize to sum to 1
        out = x / x.sum(dim=1, keepdim=True)
        return out

#### features extraction  for path dependant agent ####
def time_series_interpolation(
    X_BDt: torch.Tensor,
    target_length: int = 1000,
    method: str = 'linear'
    ) -> torch.Tensor:
    """
    Interpolates a time series to a fixed length using the specified method.
    Args:
        X_BDt (torch.Tensor): Input time series of shape (batch_size, dimension, length).
        target_length (int): Desired length of the output time series.
        method (str): Interpolation method ('linear', other can be implemented).
    Returns:
        torch.Tensor [B, D, target_length]:  Interpolated time series of shape (batch_size, dimension, target_length).
    """

    batch_size, dimension, length = X_BDt.shape
    interpolated = torch.zeros(batch_size, dimension, target_length)


    for d in range(dimension):
        for b in range(batch_size):

            interpolated[b, d, :] = torch.nn.functional.interpolate(
                X_BDt[b, d, :].unsqueeze(0).unsqueeze(0),
                size=target_length,
                mode=method
            ).squeeze(0).squeeze(0)

    # print(" shape of interpolated: ", interpolated.shape)
    return interpolated

def test_interpolation():
    # test the time series interpolation function
    X_BDt = torch.randn(10, 5, 100) # 10 samples, 5 dimensions, 100 time steps
    target_length = 1000
    interpolated = time_series_interpolation(X_BDt, target_length)
    print(" shape of interpolated: ", interpolated.shape)


    tenpercent = interpolated[:, :, 9 * T:]
    print(" shape of tenpercent: ", tenpercent.shape)
    interpolated_tenpercent = time_series_interpolation(tenpercent, target_length=10 * T)

    # plot the first sample and the first interpolated sample
    plt.figure(figsize=(10, 6))
    # plt.plot(X_BDt[0, 0, :], label='Original Sample')
    plt.plot(interpolated[0, 0, :], label='Interpolated Sample')
    plt.plot(tenpercent[0, 0, :], label='Ten Percent Sample')
    plt.plot(interpolated_tenpercent[0, 0, :], label='Interpolated Ten Percent Sample')
    plt.title('Time Series Interpolation')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()

# test_interpolation()

#  Training the Agents
def train_markov_agent(agent, train_scenarios, val_scenarios, utility_func, lr=0.05, batch_size=256, n_epochs=100, initial_wealth=100.0):
    """
    Trains an agent using scenario data.

    Parameters:
        agent (nn.Module): Agent to train. either Markovian or PathDependent
        train_scenarios (numpy array): Training scenarios. contains the (d+1)-dimensional time series. realisations of the stochastic process
        val_scenarios (numpy array): Validation scenarios
        utility_func (callable): Utility function of the agent. log utility
        lr (float): Learning rate
        batch_size (int): Batch size for training
        n_epochs (int): Number of training epochs
        initial_wealth (float): Initial wealth of the agent

    Returns:
        trained_agent: Trained agent
    """

    # Move agent to the appropriate device
    agent = agent.to(device)
    print(f"Model moved to {device}")

    train_scenarios_torch = torch.FloatTensor(train_scenarios).to(device)
    val_scenarios_torch = torch.FloatTensor(val_scenarios).to(device)

    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(train_scenarios_torch)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # we turn off shuffling for now, because we want to see the performance of the agents
    val_dataset = torch.utils.data.TensorDataset(val_scenarios_torch)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(agent.parameters(), lr=lr)

    training_logging_frequency = 10
    train_losses = []
    val_performances = []

    for epoch in range(n_epochs):
        agent.train()
        epoch_losses = []

        # iterate over batches
        for batch_index, batch in enumerate(train_loader):

            optimizer.zero_grad()  # Zero gradients before computing loss

            final_wealth = propagate_markov_trading(agent, batch, initial_wealth, plot_wealth_history=False)

            batch_performance_B1 = utility_func(final_wealth)
            loss = -1 * torch.mean(batch_performance_B1)  # Negative because we want to maximize utility
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_epoch_loss)

        # Evaluate on validation set every logging frequency epochs
        if epoch % training_logging_frequency == 0:
            agent.eval()
            val_perfs = []
            with torch.no_grad():
                for val_batch in val_loader:
                    final_wealth = propagate_markov_trading(agent, val_batch, initial_wealth, False)
                    val_perf = torch.mean(utility_func(final_wealth)).item()
                    val_perfs.append(val_perf)
            avg_val_perf = sum(val_perfs) / len(val_perfs)
            val_performances.append(avg_val_perf)
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_epoch_loss:.4f}, Val Performance: {avg_val_perf:.4f}")

    # Plot training progress
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(0, n_epochs, training_logging_frequency), val_performances)
    plt.title('Validation Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Utility')
    plt.tight_layout()
    plt.show()

    return agent

def propagate_markov_trading(agent, batch, initial_wealth=100.0, plot_wealth_history=False):
    """
    Propagates the agent's trading decisions through the batch of scenarios,
    **including** t0 in the plot.
    """
    scenarios = batch[0]  # shape [B, N, D]
    B, N, D = scenarios.shape
    device = next(agent.parameters()).device

    # 1) Initialize wealth history with t0
    current_wealth = torch.ones(B, 1, device=device) * initial_wealth
    wealth_history = [current_wealth]

    # 2) Step through time
    for t in range(N - 1):
        S_t      = scenarios[:, t    , :]  # [B, D]
        S_next   = scenarios[:, t + 1, :]
        time_t   = torch.full((B,1), t / (N - 1), device=device)
        alloc    = agent(current_wealth, S_t, time_t)               # [B, D]
        rel_ret  = S_next / S_t                                     # [B, D]
        current_wealth = current_wealth * (alloc * rel_ret).sum(1, keepdim=True)

        wealth_history.append(current_wealth)

    final_wealth = current_wealth

    if plot_wealth_history:
        # Build an (B × N) array, and a time axis 0…N-1
        wealth_np = torch.cat(wealth_history, dim=1).detach().cpu().numpy()
        t_axis    = np.arange(N)

        plt.figure(figsize=(10,6))
        for i in range(min(5, B)):
            plt.plot(t_axis, wealth_np[i], lw=1.5, label=f'Agent {i}')
        plt.title('Wealth History (including $t_0$)')
        plt.xlabel('Time Step')
        plt.ylabel('Wealth')
        plt.grid(True)
        plt.legend()
        plt.show()

    return final_wealth

def get_path_dependent_inputs(X_B1t, S_BDt, t_B1):
    ## this is the feature extraction function for the path dependent agent
    # wealth history: [B, 1, T]
    # stock history: [B, D, T]
    # time tensor: [B, 1]

    # signature depth
    sig_dept = 6 # changing this changes the dimension of the network layers!
    target_length = 3 * T

    # increase the resolutions with interpations;
    X_B1T = time_series_interpolation(X_B1t, target_length=target_length)   # [B, D, 5T]
    S_BDT = time_series_interpolation(S_BDt, target_length=target_length)   # [B, D, 5T]
    # print(" after resolution enhancement X_B1T shape: ", X_B1T.shape)
    # print(" after resolution enhancement S_BDT shape: ", S_BDT.shape)

    # 2) recent 10% history (length T) → re‐interpolate back to 10T
    X_rec = X_B1T[:, :, int(0.9 * target_length):]  # [B, D, T]
    S_rec = S_BDT[:, :, int(0.9 * target_length):]  # [B, D, T]

    # this is the CPU heavy part. using signax or signatory would be a lifesaver
    # Initialize lists to store signatures
    X_B1T_signatures = []
    S_BDT_signatures = []
    X_rec_signatures = []
    S_rec_signatures = []

    B = X_B1T.size(0)
    for b in range(B):
        # Process X_B1T
        x_single = X_B1T[b, :, :].detach().cpu().numpy().astype(np.float32)
        x_single_T = x_single.transpose()  # Shape: [T, channels]
        x_full_sig = iisignature.sig(x_single_T, sig_dept)
        X_B1T_signatures.append(torch.tensor(x_full_sig, dtype=torch.float32, device=device))

        # Process S_BDT
        s_single = S_BDT[b, :, :].detach().cpu().numpy().astype(np.float32)
        s_single_T = s_single.transpose()
        s_full_sig = iisignature.sig(s_single_T, sig_dept)
        S_BDT_signatures.append(torch.tensor(s_full_sig, dtype=torch.float32, device=device))

        # Process X_rec (recent history)
        x_rec_single = X_rec[b, :, :].detach().cpu().numpy().astype(np.float32)
        x_rec_single_T = x_rec_single.transpose()
        x_rec_full_sig = iisignature.sig(x_rec_single_T, sig_dept)
        X_rec_signatures.append(torch.tensor(x_rec_full_sig, dtype=torch.float32, device=device))

        # Process S_rec (recent stock history)
        s_rec_single = S_rec[b, :, :].detach().cpu().numpy().astype(np.float32)
        s_rec_single_T = s_rec_single.transpose()
        s_rec_full_sig = iisignature.sig(s_rec_single_T, sig_dept)
        S_rec_signatures.append(torch.tensor(s_rec_full_sig, dtype=torch.float32, device=device))

    # Stack the signatures to create batch tensors
    X_B1T_signatures = torch.stack(X_B1T_signatures, dim=0)
    S_BDT_signatures = torch.stack(S_BDT_signatures, dim=0)
    X_rec_signatures = torch.stack(X_rec_signatures, dim=0)
    S_rec_signatures = torch.stack(S_rec_signatures, dim=0)

    # Return combined signatures for network input
    combined_features = torch.cat([
        X_B1T_signatures,
        S_BDT_signatures,
        X_rec_signatures,
        S_rec_signatures,
        t_B1.to(torch.float32)  # Ensure time is float32 as well
    ], dim=1)
    # print(" shape of combined features: ", combined_features.shape)

    return combined_features

def train_path_dependent_agent(agent, train_paths, val_paths,  utility_func, lr=0.05, batch_size=256, n_epochs=100, initial_wealth=100.0):
    """
    Trains an agent using scenario data.

    Parameters:
        agent (nn.Module): Agent to train. either Markovian or PathDependent
        train_scenarios (numpy array): Training scenarios. contains the (d+1)-dimensional time series. realisations of the stochastic process
        val_scenarios (numpy array): Validation scenarios
        utility_func (callable): Utility function of the agent. log utility
        lr (float): Learning rate
        batch_size (int): Batch size for training
        n_epochs (int): Number of training epochs
        initial_wealth (float): Initial wealth of the agent

    Returns:
        trained_agent: Trained agent
    """

    # Move agent to the appropriate device
    agent = agent.to(device)
    print(f"Model moved to {device}")

    # Convert scenarios to tensor and move to device
    train_scenarios_torch = torch.FloatTensor(train_paths).to(device)
    val_scenarios_torch = torch.FloatTensor(val_paths).to(device)

    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(train_scenarios_torch)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # we turn off shuffling for now, because we want to see the performance of the agents
    val_dataset = torch.utils.data.TensorDataset(val_scenarios_torch)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(agent.parameters(), lr=lr)

    training_logging_frequency = 10
    train_losses = []
    val_performances = []

    for epoch in range(n_epochs):
        agent.train()
        epoch_losses = []

        # iterate over batches
        for batch_index, batch in enumerate(train_loader):

            optimizer.zero_grad()  # Zero gradients before computing loss

            final_wealth = propagate_path_dep_agent(agent, batch, initial_wealth, plot_wealth_history= False)

            batch_performance_B1 = utility_func(final_wealth)
            loss = -1 * torch.mean(batch_performance_B1)  # Negative because we want to maximize utility
            loss.backward()
            optimizer.step()

            # break
            epoch_losses.append(loss.item())

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_epoch_loss)

        # Evaluate on validation set every logging frequency epochs
        if epoch % training_logging_frequency == 0:
            agent.eval()
            val_perfs = []
            with torch.no_grad():
                for val_batch in val_loader:
                    final_wealth = propagate_path_dep_agent(agent, val_batch, initial_wealth, plot_wealth_history=False)
                    val_perf = torch.mean(utility_func(final_wealth)).item()
                    val_perfs.append(val_perf)
            avg_val_perf = sum(val_perfs) / len(val_perfs)
            val_performances.append(avg_val_perf)
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_epoch_loss:.4f}, Val Performance: {avg_val_perf:.4f}")

    # Plot training progress
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(0, n_epochs, training_logging_frequency), val_performances)
    plt.title('Validation Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Utility')
    plt.tight_layout()
    plt.show()

    return agent



def propagate_path_dep_agent(agent, batch, initial_wealth=100.0, plot_wealth_history=False):
    """
    Propagates the path‐dependent agent’s trading decisions and
    **includes** the initial wealth at t0 in the plot.
    """
    scenarios = batch[0]       # [B, T, D]
    B, T, D = scenarios.shape
    device = next(agent.parameters()).device

    current_wealth = torch.ones(B,1,device=device) * initial_wealth
    # start a tensor for wealth history
    wealth_history = current_wealth.squeeze(1)
    wealth_history = wealth_history.unsqueeze(1)
    wealth_history = wealth_history.unsqueeze(2)  # [B, 1, 1]
    # print(" shape of wealth history: ", wealth_history.shape)

    stock_price_history = torch.zeros((B,D, 1), device=device)
    stock_price_history[:, :, :] = scenarios[:, 0, :].unsqueeze(2)  # stock prices at t0 # [B,D,1]
    for t in range(T - 1):
        S_t     = scenarios[:, t    , :]  # [B, D]
        S_next  = scenarios[:, t + 1, :]
        time_t  = torch.full((B,1), t / (T - 1), device=device)

        # print( "shape of wealth history: ", wealth_history.shape)
        # print( "shape of what we pass :", torch.cat([current_wealth.unsqueeze(2)] + [], dim=2).shape)

        # print("shape of stock history ", stock_price_history.shape)
        # print("shape of the stock we passed : ", torch.cat([S_t.unsqueeze(2)] + [], dim=2).shape)
        path_inputs = get_path_dependent_inputs(
            wealth_history,
            stock_price_history,
            time_t
        )
        alloc = agent(path_inputs)                                  # [B, D]

        rel_ret = S_next / S_t                                      # [B, D]
        current_wealth = current_wealth * (alloc * rel_ret).sum(1, keepdim=True)
        # update wealth history
        wealth_history = torch.cat([wealth_history, current_wealth.unsqueeze(2)], dim=2)  # [B, 1, t+1]
        stock_price_history = torch.cat([stock_price_history, S_next.unsqueeze(2)], dim=2)  # [B, D, t+1]

    final_wealth = current_wealth

    if plot_wealth_history:
        wealth_np = wealth_history.detach().cpu().numpy()
        t_axis    = np.arange(T)

        plt.figure(figsize=(10,6))
        for i in range(min(5, B)):
            plt.plot(t_axis, wealth_np[i], lw=1.5, label=f'Agent {i}')
        plt.title('Path-Dependent Agent Wealth History (including $t_0$)')
        plt.xlabel('Time Step')
        plt.ylabel('Wealth')
        plt.grid(True)
        plt.legend()
        plt.show()

    return final_wealth


# Print device information
print(f"Using device: {device}")
NUM_EPOCHS = 50
LEARNING_RATE = 0.005

# Train the agent with fixed hyperparameters
print("Training Markovian and Path-Dependent Agents:")

print("\n\n\n\nBlack-Scholes model")
print("Markov agent")
trained_markov_agent_BlackScholes = train_markov_agent(MarkovianAgent(D=D), train_paths_BlackScholes, val_paths_BlackScholes, util, lr= 5 * LEARNING_RATE, initial_wealth=100.0, n_epochs= NUM_EPOCHS)
print("path-dependent agent")
trained_path_dependent_agent_BlackScholes = train_path_dependent_agent(PathDependentAgent(D = D), train_paths_BlackScholes, val_paths_BlackScholes, util, lr= 5 * LEARNING_RATE, initial_wealth=100.0, n_epochs= 50)

print("\n\n\n\nCEV model")
print("Markov agent")
trained_markov_agent_CEV = train_markov_agent(MarkovianAgent(D=D), train_paths_CEV, val_paths_CEV, util, lr= 0.01 * LEARNING_RATE, initial_wealth=100.0, n_epochs= NUM_EPOCHS)
print("path-dependent agent")

trained_path_dependent_agent_CEV = train_path_dependent_agent(PathDependentAgent(D = D), train_paths_CEV, val_paths_CEV, util, lr= 0.01 * LEARNING_RATE, initial_wealth=100.0, n_epochs= NUM_EPOCHS)

print("\n\n\n\nHeston model")
print("Markov agent")
trained_markov_agent_Heston = train_markov_agent(MarkovianAgent(D=D), train_paths_Heston, val_paths_Heston, util, lr=LEARNING_RATE, initial_wealth=100.0, n_epochs= NUM_EPOCHS)
print("path-dependent agent")
trained_path_dependent_agent_Heston = train_path_dependent_agent(PathDependentAgent(D = D), train_paths_Heston, val_paths_Heston, util, lr=0.1 * LEARNING_RATE, initial_wealth=100.0, n_epochs= NUM_EPOCHS)

# Save the trained agents
torch.save(trained_markov_agent_BlackScholes.state_dict(), 'trained_markov_agent_BlackScholes.pth')
torch.save(trained_path_dependent_agent_BlackScholes.state_dict(), 'trained_path_dependent_agent_BlackScholes.pth')
torch.save(trained_markov_agent_CEV.state_dict(), 'trained_markov_agent_CEV.pth')
torch.save(trained_path_dependent_agent_CEV.state_dict(), 'trained_path_dependent_agent_CEV.pth')
torch.save(trained_markov_agent_Heston.state_dict(), 'trained_markov_agent_Heston.pth')
torch.save(trained_path_dependent_agent_Heston.state_dict(), 'trained_path_dependent_agent_Heston.pth')