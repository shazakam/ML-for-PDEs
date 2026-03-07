from typing import Any
import torch
from .generator import DataGenerator
# from utils.conjugate_gradient import ConjugateGradient
from .boundary_operator import Laplacian
# from boundary_operator import Operator
from tqdm import tqdm
# import h5py
from .boundary_operator import BoundaryCondition

class HeatEquation(DataGenerator):
    """
    Solves the 2D heat equation on a periodic square grid using the
    Crank-Nicolson finite difference scheme.

    The heat equation is:  du/dt = a * laplacian(u)
    """

    def __init__(self, device : torch.device = torch.device("mps"), dtype : torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype
        pass

    def timestep(self, u_n : torch.Tensor, bc : BoundaryCondition, kernel : torch.Tensor, mu : float) -> torch.Tensor:
        """
        Perform a ADI time step.

        :param u_n: Current solution state, shape (root_m, root_m).
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

        D = bc.apply_boundary_condition_one_direction(I, kernel)

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
                                bc : BoundaryCondition,
                                u_n : torch.Tensor,
                                h:float, 
                                time:float, 
                                root_m : int, 
                                num_steps : int)-> torch.Tensor:
        """
        Run a full simulation of the 2D heat equation from an initial condition.

        :param a: Thermal diffusivity coefficient.
        :type a: float
        :param u_n: Initial condition matrix, shape (root_m**2, root_m**2).
        :type u_n: torch.Tensor
        :param h: Spatial grid spacing (same in x and y).
        :type h: float
        :param time: Total simulation time.
        :type time: float
        :param root_m: Number of grid points along one spatial dimension.
            The full grid has root_m**2 interior points.
        :type root_m: int
        :param num_steps: Number of time steps to take.
        :type num_steps: int
        :returns: Tensor of shape (num_steps, root_m**2, root_m**2) containing
            the solution state at each time step.
        :rtype: torch.Tensor
        """

        # TODO: Optimise for GPU - keep everything in matrix format -> Use ADI
        dt = float(time/num_steps)

        mu = a*dt/(2*(h**2))
        print(f'Mu : {mu}')
  
        kernel = torch.tensor([1., -2., 1.], dtype=self.dtype, device=self.device)
        simulation_data = torch.zeros((num_steps, root_m, root_m), device=self.device)
        simulation_data[0, :, :] = u_n.reshape((1, root_m, root_m))

        u_n = u_n.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            for t in tqdm(range(1, num_steps)):
                u_n = self.timestep(u_n, bc=bc, kernel=kernel, mu=mu)
                simulation_data[t, :, :] = u_n

        print(simulation_data.isnan())
                
        return simulation_data
    
    def generate_dataset(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return super().generate_dataset(*args, **kwargs)
    