from typing import Any
import torch
from .generator import DataGenerator
from .boundary_operator import Laplacian
from tqdm import tqdm
from .boundary_operator import BoundaryCondition
import random
from .initial_condition_generator_utils import generate_normals, generate_paths, generate_squares

class WaveEquation(DataGenerator):
    """
    Solves the 2D wave equation!
    """

    def __init__(self, 
                 n_samples : int,
                 m: int, 
                 bc : BoundaryCondition,
                 h : float,
                 time : float,
                 num_steps : int,
                 device : torch.device = torch.device("mps"), 
                 dtype : torch.dtype = torch.float32):
        self.m = m
        self.n_samples = n_samples
        self.device = device
        self.dtype = dtype 
        self.bc = bc
        self.h = h
        self.num_steps = num_steps
        self.time = time
        

    def timestep(self, u_curr : torch.Tensor, u_prev : torch.Tensor, kernel : torch.Tensor, r : float) -> torch.Tensor:
        """
        Perform an explicit leapfrog time step.

        :param u_curr: Current solution state, shape (m, m).
        :type u_curr: torch.Tensor
        :param u_prev: Previous solution state, shape (m, m).
        :type u_prev: torch.Tensor
        :param kernel: Laplacian 2D convolution kernel.
        :type kernel: torch.Tensor
        :param r: Squared CFL number (c * dt / h) ** 2.
        :type r: float
        :returns: Solution state at the next time step.
        :rtype: torch.Tensor
        """
        u_next = r * self.bc.apply_boundary_condition(u_curr, kernel) + 2*u_curr - u_prev
        return u_next

    def generate_simulation_run(self, 
                                c : float, 
                                u_n : torch.Tensor)-> torch.Tensor:
        """
        Run a full simulation of the 2D wave equation from an initial condition.

        :param c: Wave speed
        :type c: float

        :param u_n: Initial condition matrix, shape (m, m).
        :type u_n: torch.Tensor

        :param h: Spatial grid spacing (same in x and y).
        :type h: float

        :param time: Total simulation time.
        :type time: float

        :param num_steps: Number of time steps to take.
        :type num_steps: int

        :returns: Tensor of shape (num_steps, m, m) containing
            the solution state at each time step.
        :rtype: torch.Tensor
        """
        dt = self.time / self.num_steps
        r = (c * dt / self.h) ** 2

        lap = Laplacian()
        kernel = lap.get_kernel(self.device, self.dtype)
        u_n = u_n.to(device=self.device, dtype=self.dtype)

        u_curr = u_n + 0.5 * r * self.bc.apply_boundary_condition(u_n, kernel)
        simulation_data = torch.zeros((self.num_steps, self.m, self.m), device=self.device)
        simulation_data[0, :, :] = u_n.reshape((1, self.m, self.m))
        simulation_data[1, :, :] = u_curr.reshape((1, self.m, self.m))

        with torch.no_grad():
             for t in range(2, self.num_steps):
                 u_next = self.timestep(u_curr, u_n, kernel, r)
                 u_n = u_curr
                 u_curr = u_next
                 if torch.isnan(u_curr).any():
                     break
                 simulation_data[t, :, :] = u_curr

        return simulation_data
    
    def generate_dataset(self, c_min: float, c_max: float, folder_path: str) -> None:
        """
        Generate and save simulation runs to disk.

        Samples a wave speed c ~ Uniform(c_min, c_max) for each run,
        simulates the 2D wave equation, and saves one .pt file per run
        containing a dict {'X': tensor} where X has shape
        (2, num_steps, m, m):  index 0 is the solution field and
        index 1 is the wave speed (broadcast as a constant channel).

        :param c_min: Minimum wave speed.
        :type c_min: float
        :param c_max: Maximum wave speed.
        :type c_max: float
        :param folder_path: Directory in which to write sim_{i}.pt files.
        :type folder_path: str
        """
        dt = self.time / self.num_steps
        c_max_stable = self.h / (dt * (2 ** 0.5))
        safe_c_max = min(c_max, c_max_stable)
        if c_min > c_max_stable:
            raise ValueError(
                f"c_min={c_min} exceeds the CFL stability limit c_max_stable={c_max_stable:.4f}. "
                "Increase num_steps, decrease time, or increase h."
            )
        if c_max > c_max_stable:
            import warnings
            warnings.warn(
                f"c_max={c_max} exceeds CFL stability limit {c_max_stable:.4f}; "
                f"clamping to {c_max_stable:.4f}.",
                UserWarning,
                stacklevel=2,
            )

        for i in tqdm(range(self.n_samples)):
            c = random.uniform(c_min, safe_c_max)
            u_init = self.generate_random_initial_condition()
            sample = self.generate_simulation_run(c, u_init)

            c_channel = torch.full(sample.shape, c, dtype=sample.dtype, device=self.device)
            sample = torch.stack([sample, c_channel], dim=0)
            torch.save({'X': sample.cpu()}, f"{folder_path}/sim_{i}.pt")

    def generate_random_initial_condition(self) -> torch.Tensor:
        init_u = torch.zeros((self.m, self.m))
        # init_gen = random.choice([generate_squares, generate_normals, generate_paths])
        init_gen = random.choice([generate_normals])

        init_u = init_gen(init_u)

        return init_u
    