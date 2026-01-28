# Heat equation
# du/dt = a*laplace*u
# du/dt = a*(d^2u/dx^2 + d^2u/dy^2)u
# (u^{n+1}_ij - u^{n})/dt = (a/2)*(nabla u^{n} + nabla u^{n+1})
# Where nabla u^n_ij = (u^n_(i+1)j -2u^n_ij + u^n_(i-1)j)/dx + (u^n_i(j+1) -2u^n_ij + u^n_i(j-1))/dy
# TO DO: NEED TO FIGURE OUT MATRIX FORM FOR ABOVE!
import torch
import numpy
from generator import DataGenerator

class HeatEquation(DataGenerator):
    def __init__(self, a : float, initial_conditions : torch.tensor) -> torch.tensor:
        self.a = a
        self.initial_conditions = initial_conditions


    def timestep(self, t : float, a : float, U : torch.tensor) -> torch.tensor:
        """
        Performs a single time step for the 2D heat equation.

        Parameters:
        t : float : next point in time to calculate for
        a : float :  diffusivity constant
        U : torch.tensor : current values for our grid U
        """
        return