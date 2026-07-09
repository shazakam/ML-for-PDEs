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
- [x] inference script validation
- [x] Visualisation script producing desired animation - target : Model1 : Model2 : Model3
- [x] Link to WandB project
- [ ] Create report / add to website outlining stuff learned + theory with pretty visuals!

## NOTES TO SELF
- Dropped DeepONet: across extensive diagnosis it consistently failed to learn the heat operator (collapsing to an over-smoothed mean in the absolute formulation and to the identity in the residual formulation), while the FNO learns it well. Normalisation, latent dim, model size, query-point count, weight decay, residual vs absolute, and forecast horizon were all ruled out — the low-rank global branch/trunk is a poor inductive fit for a near-identity linear diffusion operator, which the FNO's spectral convolution handles naturally. Heat forecasting now uses the Diffusion model and FNO.

## Report Notes
- Intro
    - Explain the goal of the project
    - Briefly which models were used and why they were used

- Model explanations:
    - For each model explain the theory and how they are implemented in practice (cite the papers they come from)

- Model training and inference comparison across heat and wave equations

- Why does one model perform over the others

- Summarise