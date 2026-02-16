import torch
from .generator import DataGenerator
from .utils import construct_cyclical_laplacian
# import h5py

class HeatEquation(DataGenerator):
    def __init__(self):
        # self.a: float = a
        # self.root_m: int = root_m
        # self.initial_conditions: torch.Tensor = initial_conditions
        # D: torch.Tensor = construct_cyclical_laplacian(root_m)
        # mu = a*dt/(2*h**2)

        # self.C = torch.kron(torch.eye(root_m), D) + torch.kron(D, torch.eye(root_m))
        
        # A = torch.eye(root_m**2) - mu*self.C
        # self.A_inv = torch.inverse(A)
        
        # self.B = torch.eye(root_m**2) + mu*self.C
        pass

    def timestep(self, u_n : torch.Tensor, A_inv : torch.Tensor, B : torch.Tensor) -> torch.Tensor:
        """
        Performs a single time step for the 2D heat equation.

        Parameters:
        u_n : torch.Tensor : current values for our grid U
        """
        u_n = A_inv @ B @ u_n

        return u_n
    
    def generate_simulation_run(self, a : float, u_n : torch.Tensor, h:float, time:float, root_m : int, num_steps : int)-> torch.Tensor:
        """
        Docstring for generate_data
        
        :param self: Description
        :param a: Description
        :type a: float
        :param u_n: Description
        :type u_n: torch.Tensor
        :param h: Description
        :type h: float
        :param time: Description
        :type time: float
        :param root_m: Description
        :type root_m: int
        :param num_steps: Description
        :type num_steps: int
        """
        D: torch.Tensor = construct_cyclical_laplacian(root_m)
        dt = float(time/num_steps)

        mu = a*dt/(2*h**2)

        C = torch.kron(torch.eye(root_m), D) + torch.kron(D, torch.eye(root_m))
        
        A_inv  = torch.eye(root_m**2) - mu*C
        A_inv = torch.inverse(A_inv)
        B = torch.eye(root_m**2) + mu*C

        simulation_data = torch.zeros((num_steps, root_m**2, root_m**2))
        simulation_data[0, :, :] = u_n
        for t in range(1, num_steps):
            u_n = self.timestep(u_n, A_inv, B)
            simulation_data[t, :, :] = u_n

        return simulation_data
    