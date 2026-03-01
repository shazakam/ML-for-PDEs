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

    Spatial discretization uses the Kronecker product of two 1D cyclic
    Laplacian matrices to form the 2D Laplacian operator C. The
    Crank-Nicolson update is:

        (I - mu*C) u_{n+1} = (I + mu*C) u_n

    where mu = a * dt / (2 * h^2). This is unconditionally stable and
    second-order accurate in both space and time.
    """

    def __init__(self):
        pass

    def timestep(self, Ainv : torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Perform a single Crank-Nicolson time step.

        Computes u_{n+1} = A_inv @ B @ u_n where A = (I - mu*C) and
        B = (I + mu*C).

        :param u_n: Current solution state, shape (root_m**2, root_m**2).
        :type u_n: torch.Tensor
        :param A_inv: Precomputed inverse of the implicit matrix A.
        :type A_inv: torch.Tensor
        :param B: The explicit matrix B.
        :type B: torch.Tensor
        :returns: Solution state at the next time step.
        :rtype: torch.Tensor
        """ 

   

        return Ainv @ B

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

        Constructs the Crank-Nicolson matrices from the given physical and grid
        parameters, then advances the solution for num_steps time steps,
        storing the state at each step.

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

        # TODO: Optimise for GPU - keep everything in matrix format. 
        # TODO: Instead of constructing Laplacian from scratch just apply a 2D Convolution on grid as a function which adjusts according to different boundary conditions.
        device = torch.device("mps")
        dtype = torch.float32

        dt = float(time/num_steps)

        mu = a*dt/(2*h**2)

        laplacian = Laplacian()
        A = torch.eye(root_m, dtype=dtype) - mu*bc.apply_boundary_condition(torch.eye(root_m, dtype=dtype), laplacian.get_kernel(dtype=dtype))
        
        A_inv = torch.linalg.inv(A) # type: ignore

        simulation_data = torch.zeros((num_steps, root_m, root_m))
        simulation_data[0, :, :] = u_n.reshape((1, root_m, root_m))
        # u_n = u_n.to(device=device, dtype=dtype).reshape(root_m**2)

        with torch.no_grad():
            for t in tqdm(range(1, num_steps)):
                B = torch.eye(root_m, dtype=dtype) + mu*bc.apply_boundary_condition(u_n, laplacian.get_kernel(dtype=dtype))
                u_n = self.timestep(A_inv, B) # type: ignore
                simulation_data[t, :, :] = u_n.reshape((1, root_m, root_m)) # type: ignore
                
        return simulation_data
    
    def generate_dataset(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return super().generate_dataset(*args, **kwargs)
    