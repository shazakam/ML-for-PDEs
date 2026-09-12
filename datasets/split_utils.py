import torch
from torch.utils.data import Dataset, Subset


def split_by_simulation(dataset: Dataset, val_split: float, seed: int = 42) -> tuple[Subset, Subset]:
    """Split a flat (simulation, frame) dataset so no simulation appears in both halves.

    Splitting on the flat index instead puts frames from the same trajectory in
    train and validation at once. Consecutive frames of a diffusive PDE are very
    nearly the same field, so the model can score well on validation by recalling
    a trajectory it already fit, and val_loss stops measuring generalisation to
    unseen initial conditions and PDE parameters.

    Assumes the dataset flattens as index = sim_idx * frames_per_sim + frame_idx.
    The stride is derived from len(dataset) // dataset.N rather than assumed, because
    it varies by dataset: the single-step datasets use T - 1 frames per simulation
    while WaveDiffusionDataset consumes two input frames and uses T - 2.

    :param dataset: Dataset exposing N simulations and a flat length that is a multiple of it.
    :type dataset: torch.utils.data.Dataset
    :param val_split: Fraction of *simulations* held out for validation.
    :type val_split: float
    :param seed: Seed for the simulation permutation, so the split is reproducible across runs.
    :type seed: int
    :returns: Train and validation subsets, disjoint by simulation.
    :rtype: tuple[torch.utils.data.Subset, torch.utils.data.Subset]
    """
    num_sims = dataset.N # pyright: ignore[reportAttributeAccessIssue]
    if len(dataset) % num_sims != 0: # pyright: ignore[reportArgumentType]
        raise ValueError(
            f"dataset length {len(dataset)} is not a multiple of {num_sims} simulations; " # type: ignore
            "split_by_simulation needs a uniform number of frames per simulation"
        )
    frames_per_sim = len(dataset) // num_sims # type: ignore

    num_val_sims = int(round(num_sims * val_split))
    if num_val_sims < 1 or num_val_sims >= num_sims:
        raise ValueError(
            f"val_split={val_split} gives {num_val_sims} validation simulations out of {num_sims}; "
            "choose a fraction that leaves at least one simulation on each side"
        )

    generator = torch.Generator().manual_seed(seed)
    shuffled_sims = torch.randperm(num_sims, generator=generator)
    val_sims, train_sims = shuffled_sims[:num_val_sims], shuffled_sims[num_val_sims:]

    frame_offsets = torch.arange(frames_per_sim)
    to_flat_indices = lambda sims: (sims[:, None] * frames_per_sim + frame_offsets).reshape(-1).tolist()

    return Subset(dataset, to_flat_indices(train_sims)), Subset(dataset, to_flat_indices(val_sims))
