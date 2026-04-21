# ML for PDEs

> **Work in progress** — training ML models to solve partial differential equations.

## Goal

Train a **Diffusion model** (inspired by [DiffusionPDE](https://arxiv.org/abs/2406.12667)) and a **Neural Operator** to learn solution operators for 2D PDEs from simulated data.

---

## Simulations

| Heat Equation | Wave Equation | Gray-Scott Reaction-Diffusion |
|:---:|:---:|:---:|
| ![Heat](visualisations/heat2d_test.gif) | ![Wave](visualisations/wave2d_i_like.gif) | ![Gray-Scott](visualisations/gray_scott_test.gif) |
| Crank-Nicolson (ADI) | Leapfrog (explicit) | Strang splitting + Heun |

---

# NOTE TO SELF
What is the dimensionality of the data generated?
For heat and wave equations a single sample has dimensions (2, T, H, W) where (0, T, H, W)

## TODO:
- [x] Pre-processing scripts: MinMax Norm and Z-Norm
- [ ] Need to group configs according to experiment + output files should have files for params
- [ ] Integrate pre-processing, train and test split in data_generation_scripts / or write bash scripts
- [ ] Then go onto Deep ONet once we have a semi-function diffusion model
- [x] Visualised test output in notebook (it didn't work)


## Suggested model changes for wave equation:
- [ ] Reduce speed size distribution
- [ ] Condition on two previous timesteps instead of one
- [ ] Increase model size (should be possible and more efficient now with better data handling)

For wave condition on previous two timesteps, increase model size, issue in the training data as well
