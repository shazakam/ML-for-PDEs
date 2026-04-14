from data_generators.wave_2d import WaveEquation
import torch
from utils.visualisation_utils import tensor_to_gif
from data_generators.boundary_operator import PeriodicBoundary
import sys

print('Running Test...')
c = 0.01
h = 0.01
time = 5
m = 512
num_steps = 1000

dt = time / num_steps
boundary_condition = PeriodicBoundary()

data_generator = WaveEquation(n_samples = 100,
                              m = m, 
                              bc = boundary_condition,
                              h = h,
                              time = time,
                              num_steps = num_steps)
r = c * dt / h
print(r)
u_n = data_generator.generate_random_initial_condition()

print('Beginning test simulation')
simulation_run = data_generator.generate_simulation_run(c = 1, u_n = u_n)

save_path = "visualisations/wave2d_test.gif"
stride = 20
tensor_to_gif(simulation_run[::stride], save_path, fps=60)
print('Complete')

