# ML for PDEs

> **Work in progress** — training ML models to solve partial differential equations.

## Goal

Train a **Diffusion model** and a **Neural Operator (FNO)** to learn solution operators for 2D PDEs from simulated data.

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

## TODO:
- [x] Training script for FNO
- [x] inference script
- [ ] inference script validation
- [ ] Visualisation script producing desired animation - target : Model1 : Model2 : Model3
- [ ] Link to WandB project
- [ ] Create report / add to website outlining stuff learned + theory with pretty visuals!
