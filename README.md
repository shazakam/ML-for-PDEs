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
- [ ] Need to group configs according to experiment + output files should have files for params
- [x] Update the pre-processing functions AND SCRIPTS now that we have the aggregated data files
- [ ] Test that training works lightning and GPU is faster than MPS
- [ ] Train large model on heat data and pray 
- [ ] Need to make a residual test script / function to compare outputs and labels in test test + long time forecasting i.e. How does the error evolve over multi-step predictions?
- [ ] Then go onto Deep ONet once we have a semi-function diffusion model
- [x] Visualised test output in notebook (it didn't work)


## Suggested model changes for wave equation:
- [ ] Reduce speed size distribution
- [ ] Condition on two previous timesteps instead of one -> Will need to create different torch dataset for wave for this
- [ ] Increase model size (should be possible and more efficient now with better data handling)

For wave condition on previous two timesteps, increase model size, issue in the training data as well
