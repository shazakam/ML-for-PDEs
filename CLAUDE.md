# CLAUDE.md

## Project

ML-for-PDEs — training ML models (diffusion models, neural operators) to solve partial differential equations. Currently focused on 2D PDE data generation, starting with the heat equation.

## Package Manager

Uses **UV** with Python 3.12.

```sh
uv sync          # install/sync all dependencies from uv.lock
uv add <pkg>     # add a dependency
uv run <cmd>     # run a command in the venv
```

## Architecture

### DataGenerator ABC (`data_generators/generator.py`)

All PDE data generators inherit from `DataGenerator` and implement three methods:
- `timestep()` — single time step of the PDE solver
- `generate_simulation_run()` — full simulation from an initial condition
- `generate_dataset()` — multiple simulation runs for training data

### HeatEquation (`data_generators/heat_2d.py`)

Crank-Nicolson finite difference scheme on a periodic square grid:
- 2D Laplacian built via Kronecker products of the 1D cyclical Laplacian (`utils.py`)
- Precomputes `A_inv` and `B` matrices; each timestep is `u_{n+1} = A_inv @ B @ u_n`

### Utilities (`data_generators/utils.py`)

`construct_cyclical_laplacian(m)` — tridiagonal matrix with periodic boundary conditions.

## Conventions

- **Strong typing**: all function parameters and return types annotated; use `torch.Tensor` throughout
- **Docstrings**: Sphinx/reST format (`:param:`, `:type:`, `:returns:`, `:rtype:`)
- **Tensors**: use PyTorch tensors, not NumPy arrays, for all numerical data

## Current State

Working branch: `heat-2d`. The `HeatEquation` class has `timestep()` and `generate_simulation_run()` implemented. Still needed:
- `generate_dataset()` for HeatEquation
- Tests for Laplacian construction, timestep, simulation run, and dataset generation
- Reaction-diffusion and wave equation generators (placeholders exist)
- No linting or test framework configured yet