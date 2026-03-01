from data_generators.heat_2d import HeatEquation
import torch
from utils.visualisation_utils import tensor_to_gif
from data_generators.boundary_operator import PeriodicBoundary
import sys

data_generator = HeatEquation()
boundary_condition = PeriodicBoundary()

print('Running Test...')
a = 0.2
h = 0.1
time = 100
root_m = 256
num_steps = 2000

u_n = torch.zeros((root_m, root_m))
u_n[root_m//2 - root_m//4:root_m//2 + root_m//4, root_m//2 - root_m//4:root_m//2 + root_m//4] = 1
dt = time / num_steps
mu = a*dt/(h**2)
# print(f'Mu {mu}')
# if mu > 0.25:
#     sys.exit()

print('Beginning test simulation')
simulation_run = data_generator.generate_simulation_run(a , boundary_condition, u_n, h, time, root_m, num_steps)

save_path = "visualisations/heat2d_test.gif"
stride = 20
tensor_to_gif(simulation_run[::stride], save_path, fps=60)

