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

## Roadmap

### Data
- [x] Heat equation 2D data generator
- [x] Wave equation 2D data generator
- [x] Gray-Scott reaction-diffusion 2D data generator
- [ ] Validate dataset distributions for wave and Gray-Scott

### Models
- [ ] DiffusionPDE architecture
- [ ] Neural Operator architecture

### Training & Evaluation
- [ ] Model training scripts
- [ ] Model tests
- [ ] Inference and comparison scripts
