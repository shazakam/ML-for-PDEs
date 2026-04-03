from typing import Any
import torch
from .generator import DataGenerator
from .boundary_operator import Laplacian
from tqdm import tqdm
from .boundary_operator import BoundaryCondition
import random
from .initial_condition_generator_utils import generate_normals, generate_paths, generate_squares

class GrayScottEquation(DataGenerator):
    """
    Solves the 2D Gray Scott Reaction Diffusion equation!
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
        self.dt = self.time / self.num_steps
        

    def timestep(self, u_curr : torch.Tensor, v_curr : torch.Tensor, kernel : torch.Tensor, Du : float, Dv : float, F : float, k : float):
        """
        Perform a strang splitting step using Heun / RK2 method to approximate ODEs

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
        f_diff =  lambda D, x : D*self.bc.apply_boundary_condition(x, kernel)
        f_react_u = lambda u, v, F: -torch.mul(u, v**2) + F*(1 - u)
        f_react_v = lambda u, v, F, k : torch.mul(u, v**2) - (F+k)*v

        # Half step diffusion
        u_half = self.heun_update(lambda x :f_diff(Du, x), u_curr, self.dt/2)
        v_half = self.heun_update(lambda x :f_diff(Dv, x), v_curr, self.dt/2)

        # Full step reaction
        u_full = self.heun_update(lambda x : f_react_u(x, v_half, F), u_half, self.dt)
        v_full = self.heun_update(lambda x : f_react_v(u_half, x, F, k), v_half, self.dt)

        # Half step diffusion
        u_next = self.heun_update(lambda x :f_diff(Du, x), u_full, self.dt/2)
        v_next = self.heun_update(lambda x :f_diff(Dv, x), v_full, self.dt/2)

        return u_next, v_next
     
    def heun_update(self, f, x_init : torch.Tensor, dt):
        x_init_dt = f(x_init)
        x_init_half = x_init + dt*x_init_dt
        x_init_half_dt = f(x_init_half)

        x_next = x_init + (dt/2)*(x_init_dt + x_init_half_dt)

        return x_next
    
    def generate_simulation_run(self, 
                                c : float, 
                                u_n : torch.Tensor)-> torch.Tensor:

        return u_n
    
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
    