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
        pass

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
        cfl = c * dt / self.h
        if cfl > 1.0 / (2 ** 0.5):
            raise ValueError(
                f"CFL condition violated: c·dt/h = {cfl:.4f} > 1/√2 ≈ 0.7071. "
                "Reduce dt (increase num_steps), reduce c, or increase h."
            )

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
    
    def generate_dataset(self, folder_path : str) -> None:
        # Generates a torch tensor dataset and saves it to the given path

        for i in tqdm(range(self.n_samples)):
            # a = random.uniform(a_min, a_max)
            u_init = self.generate_random_initial_condition()
            # sample = self.generate_simulation_run(a, u_init)

            # alpha_channel = torch.full(sample.shape, a, dtype=sample.dtype, device = self.device)  # same shape as sample
            # sample = torch.stack([sample, alpha_channel], dim=0)
            # torch.save({'X': sample.cpu()}, f"{folder_path}/sim_{i}.pt")

    def generate_random_initial_condition(self) -> torch.Tensor:
        init_u = torch.zeros((self.m, self.m))
        init_gen = random.choice([generate_squares, generate_normals, generate_paths])

        init_u = init_gen(init_u)

        return init_u
    