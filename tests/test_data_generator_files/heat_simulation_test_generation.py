from data_generators.heat_2d import HeatEquation
import torch
from utils.visualisation_utils import tensor_to_gif
from data_generators.boundary_operator import PeriodicBoundary
import sys

print('Running Test...')
a = 1
h = 0.1
time = 100
m = 256
num_steps = 2000

u_n = torch.zeros((m, m))
u_n[m//2 - m//4:m//2 + m//4, m//2 - m//4:m//2 + m//4] = 1
dt = time / num_steps
mu = a*dt/(h**2)

data_generator = HeatEquation(n_samples = 100, max_t = 100, m = m)
boundary_condition = PeriodicBoundary()

print('Beginning test simulation')
simulation_run = data_generator.generate_simulation_run(a , boundary_condition, u_n, h, time, num_steps)

save_path = "visualisations/heat2d_test.gif"
stride = 20
tensor_to_gif(simulation_run[::stride], save_path, fps=60)
print('Complete')

