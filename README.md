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

## Forecasting

Ground truth, DDPM forecast, and absolute error for three held-out heat equation test samples:

![Heat forecast grid](visualisations/heat_forecast_grid.gif)

---

# NOTE TO SELF
What is the dimensionality of the data generated?
For heat and wave equations a single sample has dimensions (2, T, H, W) where (0, T, H, W)

## TODO:
- [ ] Need to make testing scripts for DeepONet and Diffusion Model
- [ ] Testing scripts should produce all desired visualistions (will need expand /visualisations)
- [ ] Link to WandB project
- [ ] Create report / add to website outlining stuff learned + theory with pretty visuals!
