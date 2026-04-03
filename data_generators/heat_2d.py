from typing import Any
import torch
from .generator import DataGenerator
from .boundary_operator import Laplacian
from tqdm import tqdm
from .boundary_operator import BoundaryCondition
import random
from .initial_condition_generator_utils import generate_normals, generate_paths, generate_squares

class HeatEquation(DataGenerator):
    """
    Solves the 2D heat equation on a periodic square grid using the
    Alternating Direction Implicit Method.

    The heat equation is:  du/dt = a * laplacian(u)
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
        

    def timestep(self, u_n : torch.Tensor, kernel : torch.Tensor, mu : float) -> torch.Tensor:
        """
        Perform a ADI time step.

        :param u_n: Current solution state, shape (m, m).
        :type u_n: torch.Tensor
        :param bc: Boundary condition object used to apply the operator.
        :type bc: BoundaryCondition
        :param kernel: Laplacian 1D convolution kernel.
        :type kernel: torch.Tensor
        :param mu: Scaled time step a*dt/h².
        :type mu: float
        :returns: Solution state at the next time step.
        :rtype: torch.Tensor
        """

        I = torch.eye(u_n.shape[0], dtype=self.dtype, device=self.device)

        D = self.bc.apply_boundary_condition_one_direction(I, kernel)

        # First solve   
        rhs = u_n @ (mu*D + I)
        lhs =  I - mu*D
        u_n = torch.linalg.solve(lhs, rhs.T).T

        # Second solve
        rhs = (I + mu*D)@u_n 
        lhs = (I - mu*D)

        u_n = torch.linalg.solve(lhs, rhs)  

        return u_n

    def generate_simulation_run(self, 
                                a : float, 
                                u_n : torch.Tensor)-> torch.Tensor:
        """
        Run a full simulation of the 2D heat equation from an initial condition.

        :param a: Thermal diffusivity coefficient.
        :type a: float

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

        dt = float(self.time/self.num_steps)

        mu = a*dt/(2*(self.h**2))
  
        kernel = torch.tensor([1., -2., 1.], dtype=self.dtype, device=self.device)
        simulation_data = torch.zeros((self.num_steps, self.m, self.m), device=self.device)
        simulation_data[0, :, :] = u_n.reshape((1, self.m, self.m))

        u_n = u_n.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            for t in range(1, self.num_steps):
                u_n = self.timestep(u_n, kernel=kernel, mu=mu)
                simulation_data[t, :, :] = u_n
                
        return simulation_data
    
    def generate_dataset(self, a_min : float, a_max : float, folder_path : str) -> None:
        # Generates a torch tensor dataset and saves it to the given path

        for i in tqdm(range(self.n_samples)):
            a = random.uniform(a_min, a_max)
            u_init = self.generate_random_initial_condition()
            sample = self.generate_simulation_run(a, u_init)

            alpha_channel = torch.full(sample.shape, a, dtype=sample.dtype, device = self.device)  # same shape as sample
            sample = torch.stack([sample, alpha_channel], dim=0)
            torch.save({'X': sample.cpu()}, f"{folder_path}/sim_{i}.pt")

    def generate_random_initial_condition(self) -> torch.Tensor:
        init_u = torch.zeros((self.m, self.m))
        init_gen = random.choice([generate_squares, generate_normals, generate_paths])

        init_u = init_gen(init_u)

        return init_u
    