from data_generators.heat_2d import HeatEquation
import torch
from utils.visualisation_utils import tensor_to_gif

data_generator = HeatEquation()

print('Running Test...')
a = 0.2
h = 0.1
time = 100
root_m = 128
num_steps = 2000

u_n = torch.zeros((root_m, root_m))
u_n[root_m//2 - root_m//4:root_m//2 + root_m//4, root_m//2 - root_m//4:root_m//2 + root_m//4] = 1

u_n = u_n.flatten()

print('Beginning test simulation')
simulation_run = data_generator.generate_simulation_run(a , u_n, h, time, root_m, num_steps)

save_path = "visualisations/heat2d_test.gif"
tensor_to_gif(simulation_run, save_path, fps = 20)

