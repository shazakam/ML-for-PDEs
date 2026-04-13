from data_generators.heat_2d import HeatEquation
import torch
from utils_general.visualisation_utils import tensor_to_gif
from data_generators.boundary_operator import PeriodicBoundary
import sys

print('Running Test...')
a = 0.01
h = 0.1
time = 50
m = 256
num_steps = 1000

dt = time / num_steps
mu = a*dt/(h**2)

boundary_condition = PeriodicBoundary()

data_generator = HeatEquation(n_samples = 100,
                              m = m, 
                              bc = boundary_condition,
                              h = h,
                              time = time,
                              num_steps = num_steps)

u_n = data_generator.generate_random_initial_condition()

print('Beginning test simulation')
simulation_run = data_generator.generate_simulation_run(a = a, u_n = u_n)

save_path = "visualisations/heat2d_test.gif"
stride = 20
tensor_to_gif(simulation_run[::stride], save_path, fps=60)
print('Complete')

