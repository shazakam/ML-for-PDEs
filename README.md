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

## TODO:

- [ ] Need Gray-Scott config
- [ ] Need to ensure simulation samples generated for Wave Equation and Gray-Scott do not break i.e. paramaters and initial conditions ensure outputs are stable when generated
- [ ] Need to handle data loading lightning_dataset folders
- [ ] Set up training script for diffusion model