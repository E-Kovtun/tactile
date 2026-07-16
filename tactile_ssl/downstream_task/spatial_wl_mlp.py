from typing import List, Optional

import torch
from torch import nn

from tactile_ssl.model.xela_spatial_gnn import (
    build_cached_pyg_graph_batch,
    physical_graph_edge_pairs_torch,
)


def _load_wl_conv_continuous():
    try:
        from torch_geometric.nn import WLConvContinuous
    except ImportError as exc:
        raise ImportError(
            "XelaSpatialWLMLPEncoder requires torch_geometric. "
            "Install PyG in the active environment before using the WL spatial downstream."
        ) from exc
    return WLConvContinuous


class XelaSpatialWLMLPEncoder(nn.Module):
    """Encode Xela XYZ coordinates with physical-graph WL diffusion followed by an MLP."""

    def __init__(
        self,
        spatial_hidden_dims: List[int],
        wl_num_layers: int = 2,
        coordinate_dim: int = 3,
        num_nodes: int = 368,
        bridge_k: int = 4,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if coordinate_dim != 3:
            raise ValueError(f"Xela spatial coordinates must have dimension 3; got {coordinate_dim}")
        if num_nodes != 368:
            raise ValueError(f"XelaSpatialWLMLPEncoder expects 368 nodes; got {num_nodes}")
        if wl_num_layers <= 0:
            raise ValueError("wl_num_layers must be positive")
        if not spatial_hidden_dims:
            raise ValueError("spatial_hidden_dims must contain at least the output embedding dimension")
        if any(dim <= 0 for dim in spatial_hidden_dims):
            raise ValueError("all spatial_hidden_dims values must be positive")
        if bridge_k < 0:
            raise ValueError("bridge_k must be non-negative")

        self.coordinate_dim = coordinate_dim
        self.num_nodes = num_nodes
        self.bridge_k = int(bridge_k)
        self.wl_num_layers = int(wl_num_layers)
        self.spatial_hidden_dims = tuple(spatial_hidden_dims)

        WLConvContinuous = _load_wl_conv_continuous()
        self.wl_layers = nn.ModuleList(WLConvContinuous() for _ in range(self.wl_num_layers))

        mlp_dims = [coordinate_dim, *spatial_hidden_dims]
        mlp_layers = []
        for layer_idx, (in_dim, out_dim) in enumerate(zip(mlp_dims[:-1], mlp_dims[1:])):
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            if layer_idx < len(mlp_dims) - 2:
                mlp_layers.append(nn.GELU())
        self.mlp = nn.Sequential(*mlp_layers)
        self.norm = nn.LayerNorm(spatial_hidden_dims[-1])

        self.register_buffer(
            "static_physical_edge_index",
            torch.empty((2, 0), dtype=torch.long),
            persistent=False,
        )
        self.apply(lambda module: self._init_weights(module, init_std))

    @staticmethod
    def _init_weights(module: nn.Module, init_std: float) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

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

    def _build_edge_index(self, positions: torch.Tensor, graph_info: dict[str, torch.Tensor]) -> torch.Tensor:
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

        edge_index, _ = build_cached_pyg_graph_batch(
            graph_info,
            num_nodes=self.num_nodes,
            device=positions.device,
            edge_mode="topology",
            static_edge_index=static_edge_index,
        )
        return edge_index

    def forward(self, positions: torch.Tensor, graph_info: Optional[dict[str, torch.Tensor]]) -> torch.Tensor:
        if graph_info is None:
            raise ValueError("graph_info is required for XelaSpatialWLMLPEncoder")
        if positions.ndim != 3:
            raise ValueError(f"positions must have shape [graphs, 368, 3]; got {tuple(positions.shape)}")
        if positions.shape[1] != self.num_nodes:
            raise ValueError(f"positions must contain {self.num_nodes} Xela nodes; got {positions.shape[1]}")
        if positions.shape[2] != self.coordinate_dim:
            raise ValueError(f"positions must have {self.coordinate_dim} coordinate channels; got {positions.shape[2]}")

        edge_index = self._build_edge_index(positions, graph_info)
        spatial = positions.reshape(-1, self.coordinate_dim)
        for wl_layer in self.wl_layers:
            spatial = wl_layer(spatial, edge_index)
        spatial = self.norm(self.mlp(spatial))
        return spatial.view(positions.shape[0], self.num_nodes, self.output_dim)
