from data_generators.heat_2d import HeatEquation
import torch
from utils.visualisation_utils import tensor_to_gif

data_generator = HeatEquation()

print('Running Test...')
a = 0.1 
h = 0.1
time = 40
root_m = 80
num_steps = 250

u_n = torch.zeros((root_m, root_m))
u_n[32:48, 32:48] = 1
u_n = u_n.flatten()

print('Beginning test simulation')
simulation_run = data_generator.generate_simulation_run(a , u_n, h, time, root_m, num_steps)

save_path = "visualisations/heat2d_test.gif"
tensor_to_gif(simulation_run, save_path, fps = 20)

