from typing import Any, Tuple
import torch
from .generator import DataGenerator
from .boundary_operator import Laplacian
from tqdm import tqdm
from .boundary_operator import BoundaryCondition
import random
from .initial_condition_generator_utils import generate_normals, generate_paths, generate_squares

class GrayScottEquation(DataGenerator):
    """
    Solves the 2D Gray-Scott reaction-diffusion equation:

        du/dt = Du * Lap(u) - u*v^2 + F*(1 - u)
        dv/dt = Dv * Lap(v) + u*v^2 - (F + k)*v

    Integration uses Strang operator splitting: a half diffusion step (Heun),
    a full reaction step (Heun), then another half diffusion step (Heun).
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
                                Du : float,
                                Dv : float,
                                F : float,
                                k : float,
                                u_n : torch.Tensor,
                                v_n : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run a full Gray-Scott simulation from the given initial conditions.

        :param Du: Diffusion coefficient for u.
        :type Du: float
        :param Dv: Diffusion coefficient for v.
        :type Dv: float
        :param F: Feed rate.
        :type F: float
        :param k: Kill rate.
        :type k: float
        :param u_n: Initial u field, shape (m, m).
        :type u_n: torch.Tensor
        :param v_n: Initial v field, shape (m, m).
        :type v_n: torch.Tensor
        :returns: Tuple of (u_trajectory, v_trajectory), each shape (num_steps, m, m).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        kernel = Laplacian().get_kernel(self.device, self.dtype)
        u_curr = u_n.to(device=self.device, dtype=self.dtype)
        v_curr = v_n.to(device=self.device, dtype=self.dtype)

        u_traj = torch.empty((self.num_steps, self.m, self.m), dtype=self.dtype)
        v_traj = torch.empty((self.num_steps, self.m, self.m), dtype=self.dtype)

        u_traj[0] = u_curr
        v_traj[0] = v_curr

        for t in range(1, self.num_steps):
            u_curr, v_curr = self.timestep(u_curr, v_curr, kernel, Du, Dv, F, k)
            if torch.isnan(u_curr).any() or torch.isnan(v_curr).any():
                print(f"NaN detected at step {t}, stopping early.")
                break
            u_traj[t] = u_curr
            v_traj[t] = v_curr

        return u_traj, v_traj

    def generate_dataset(self,
                         Du : float,
                         Dv : float,
                         F_min : float,
                         F_max : float,
                         k_min : float,
                         k_max : float,
                         folder_path : str) -> None:
        """
        Generate a dataset of Gray-Scott simulations and save each run to disk.

        Du and Dv are fixed across the dataset; F and k are sampled uniformly
        per simulation run. Each file contains a dict with keys u, v,
        F, and k.

        :param Du: Fixed diffusion coefficient for u.
        :type Du: float
        :param Dv: Fixed diffusion coefficient for v.
        :type Dv: float
        :param F_min: Lower bound for uniform sampling of feed rate F.
        :type F_min: float
        :param F_max: Upper bound for uniform sampling of feed rate F.
        :type F_max: float
        :param k_min: Lower bound for uniform sampling of kill rate k.
        :type k_min: float
        :param k_max: Upper bound for uniform sampling of kill rate k.
        :type k_max: float
        :param folder_path: Directory in which to save sim_{i}.pt files.
        :type folder_path: str
        :returns: None
        """
        for i in tqdm(range(self.n_samples)):
            F = random.uniform(F_min, F_max)
            k = random.uniform(k_min, k_max)
            # NOTE: Why is Du and Dv not being generated randomly here?
            u_init, v_init = self.generate_random_initial_condition()
            u_traj, v_traj = self.generate_simulation_run(Du, Dv, F, k, u_init, v_init)

            torch.save({'u': u_traj.cpu(), 'v': v_traj.cpu(), 'F': F, 'k': k},
                       f"{folder_path}/sim_{i}.pt")

    def generate_random_initial_condition(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a random Gray-Scott initial condition.

        Produces a mask via one of the standard random generators, then sets:
        - u = 1 - 0.5 * mask
        - v = 0.25 * mask

        :returns: Tuple of (u_init, v_init), each shape (m, m).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        mask = torch.zeros((self.m, self.m))
        init_gen = random.choice([generate_normals])
        mask = init_gen(mask)

        # Normalise mask to [0, 1]
        mask_min, mask_max = mask.min(), mask.max()
        if mask_max > mask_min:
            mask = (mask - mask_min) / (mask_max - mask_min)

        u_init = 1.0 - 0.5 * mask
        v_init = 0.25 * mask

        return u_init, v_init
    