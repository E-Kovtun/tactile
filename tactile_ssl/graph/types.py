from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class WeightedSensorGraph:
    edge_index: np.ndarray
    edge_weight: np.ndarray
    num_nodes: int = 368
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        edge_index = np.asarray(self.edge_index, dtype=np.int64)
        edge_weight = np.asarray(self.edge_weight, dtype=np.float32)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape (2, E); got {edge_index.shape}")
        if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
            raise ValueError(
                "edge_weight must have shape (E,) matching edge_index; "
                f"got {edge_weight.shape} and {edge_index.shape}"
            )
        if edge_index.size:
            if edge_index.min() < 0 or edge_index.max() >= self.num_nodes:
                raise ValueError(f"edge_index values must be within 0..{self.num_nodes - 1}")
            if np.any(edge_index[0] == edge_index[1]):
                raise ValueError("Self-loops are not included in sensor graphs")
        if np.any(edge_weight <= 0):
            raise ValueError("All edge weights must be positive distances")
        object.__setattr__(self, "edge_index", edge_index)
        object.__setattr__(self, "edge_weight", edge_weight)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
