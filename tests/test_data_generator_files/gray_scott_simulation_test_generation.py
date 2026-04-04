from data_generators.gray_scott_2d import GrayScottEquation
import torch
from utils.visualisation_utils import tensor_to_gif
from data_generators.boundary_operator import PeriodicBoundary

print('Running Test...')

# Pearson 1993 "beta" spot pattern — well within the Turing-unstable region
# Non-trivial steady state requires (F+k)^2 / F < 0.25; here: (0.1)^2 / 0.04 = 0.25 (boundary)
Du = 0.16
Dv = 0.08
F = 0.040
k = 0.060

h = 1.0
time = 3000.0
m = 256
num_steps = 3000  # dt = 1.0

boundary_condition = PeriodicBoundary()

data_generator = GrayScottEquation(n_samples=10,
                                   m=m,
                                   bc=boundary_condition,
                                   h=h,
                                   time=time,
                                   num_steps=num_steps)

u_init, v_init = data_generator.generate_random_initial_condition()

print('Beginning test simulation')
u_traj, v_traj = data_generator.generate_simulation_run(Du=Du, Dv=Dv, F=F, k=k,
                                                         u_n=u_init, v_n=v_init)

save_path = "visualisations/gray_scott_test.gif"
stride = 50
tensor_to_gif(u_traj[::stride], save_path, fps=60)
print('Complete')
