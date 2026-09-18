from typing import Sequence

import torch
from torch.utils.data import Dataset, Subset


def deterministic_nested_fractional_subsets(
    datasets: Sequence[Dataset],
    fraction: float,
    seed: int,
) -> list[Dataset]:
    """Subsample every dataset deterministically while preserving all strata.

    Xela object classification stores each object/recording pair as a separate
    dataset. Sampling each component independently keeps every available stratum
    represented and, for a fixed seed, makes smaller budgets exact subsets of
    larger budgets.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        return list(datasets)

    subsets: list[Dataset] = []
    for dataset_idx, dataset in enumerate(datasets):
        dataset_size = len(dataset)
        if dataset_size == 0:
            subsets.append(dataset)
            continue

        subset_size = max(1, int(dataset_size * fraction))
        generator = torch.Generator().manual_seed(seed + 1_000_003 * dataset_idx)
        indices = torch.randperm(dataset_size, generator=generator)[:subset_size]
        # Preserve the original sample order. The training DataLoader shuffles.
        subsets.append(Subset(dataset, indices.sort().values.tolist()))
    return subsets
