from typing import List, Optional

import torch
from torch import nn

from tactile_ssl.model.xela_spatial_gnn import (
    build_cached_pyg_graph_batch,
    load_gatv2_conv,
    physical_graph_edge_pairs_torch,
)


class XelaSpatialGATv2Encoder(nn.Module):
    """Build supervised Xela spatial embeddings with GATv2 on the physical graph."""

    def __init__(
        self,
        spatial_hidden_dims: List[int],
        gat_heads: int = 4,
        gat_dropout: float = 0.0,
        coordinate_dim: int = 3,
        num_nodes: int = 368,
        bridge_k: int = 4,
        edge_mode: str = "distance",
    ) -> None:
        super().__init__()
        if coordinate_dim != 3:
            raise ValueError(f"Xela spatial coordinates must have dimension 3; got {coordinate_dim}")
        if num_nodes != 368:
            raise ValueError(f"XelaSpatialGATv2Encoder expects 368 nodes; got {num_nodes}")
        if not spatial_hidden_dims:
            raise ValueError("spatial_hidden_dims must contain at least one GATv2 output dimension")
        if any(dim <= 0 for dim in spatial_hidden_dims):
            raise ValueError("all spatial_hidden_dims values must be positive")
        if gat_heads <= 0:
            raise ValueError("gat_heads must be positive")
        if any(dim % gat_heads != 0 for dim in spatial_hidden_dims):
            raise ValueError("every spatial_hidden_dims value must be divisible by gat_heads")
        if not 0.0 <= gat_dropout < 1.0:
            raise ValueError("gat_dropout must be in the range [0, 1)")
        if bridge_k < 0:
            raise ValueError("bridge_k must be non-negative")
        if edge_mode not in {"distance", "topology"}:
            raise ValueError(f"unsupported edge_mode={edge_mode!r}")

        self.coordinate_dim = coordinate_dim
        self.num_nodes = num_nodes
        self.bridge_k = int(bridge_k)
        self.edge_mode = edge_mode
        self.spatial_hidden_dims = tuple(spatial_hidden_dims)

        GATv2Conv = load_gatv2_conv()
        edge_dim = 1 if edge_mode == "distance" else None
        layer_dims = [coordinate_dim, *spatial_hidden_dims]
        self.gat_layers = nn.ModuleList(
            GATv2Conv(
                in_dim,
                out_dim // gat_heads,
                heads=gat_heads,
                concat=True,
                dropout=gat_dropout,
                edge_dim=edge_dim,
            )
            for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:])
        )
        self.activation = nn.ELU()
        self.register_buffer(
            "static_physical_edge_index",
            torch.empty((2, 0), dtype=torch.long),
            persistent=False,
        )

    @property
    def output_dim(self) -> int:
        return self.spatial_hidden_dims[-1]

    def _get_static_physical_edge_index(self, positions: torch.Tensor) -> torch.Tensor:
        if self.static_physical_edge_index.numel() > 0:
            return self.static_physical_edge_index.to(device=positions.device)

        edge_pairs = physical_graph_edge_pairs_torch(positions[0], bridge_k=self.bridge_k)
        if edge_pairs:
            undirected = torch.tensor(edge_pairs, dtype=torch.long, device=positions.device).t()
            edge_index = torch.cat([undirected, undirected.flip(0)], dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=positions.device)
        self.static_physical_edge_index = edge_index.detach()
        return self.static_physical_edge_index.to(device=positions.device)

    def _build_graph(
        self,
        positions: torch.Tensor,
        graph_info: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if "edge_count" not in graph_info:
            raise ValueError("graph_info must contain edge_count")

        edge_count = graph_info["edge_count"]
        if edge_count.ndim != 1:
            raise ValueError(f"graph_info edge_count must be one-dimensional; got {tuple(edge_count.shape)}")
        if edge_count.numel() != positions.shape[0]:
            raise ValueError(
                "graph_info and coordinates must contain the same number of graphs; "
                f"got {edge_count.numel()} and {positions.shape[0]}"
            )

        static_edge_index: Optional[torch.Tensor] = None
        if graph_info.get("edge_index") is None:
            static_edge_index = self._get_static_physical_edge_index(positions)

        return build_cached_pyg_graph_batch(
            graph_info,
            num_nodes=self.num_nodes,
            device=positions.device,
            edge_mode=self.edge_mode,
            static_edge_index=static_edge_index,
        )

    def forward(
        self,
        positions: torch.Tensor,
        graph_info: Optional[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if graph_info is None:
            raise ValueError("graph_info is required for XelaSpatialGATv2Encoder")
        if positions.ndim != 3:
            raise ValueError(f"positions must have shape [graphs, 368, 3]; got {tuple(positions.shape)}")
        if positions.shape[1] != self.num_nodes:
            raise ValueError(f"positions must contain {self.num_nodes} Xela nodes; got {positions.shape[1]}")
        if positions.shape[2] != self.coordinate_dim:
            raise ValueError(f"positions must have {self.coordinate_dim} coordinate channels; got {positions.shape[2]}")

        edge_index, edge_attr = self._build_graph(positions, graph_info)
        spatial = positions.reshape(-1, self.coordinate_dim)
        for layer_idx, gat_layer in enumerate(self.gat_layers):
            if self.edge_mode == "distance":
                spatial = gat_layer(spatial, edge_index, edge_attr)
            else:
                spatial = gat_layer(spatial, edge_index)
            if layer_idx < len(self.gat_layers) - 1:
                spatial = self.activation(spatial)
        return spatial.view(positions.shape[0], self.num_nodes, self.output_dim)
