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
- [ ] Training almost works but loading in the data takes too long we need to combine all the training files into one and use mmap = True in torch.load
- [ ] Then go onto Deep ONet once we have a semi-function diffusion model