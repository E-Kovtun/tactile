from functools import partial
from typing import Callable, List, Literal, Optional

import einops
import torch
import torch.nn as nn
from omegaconf import DictConfig

from tactile_ssl.data.xela.utils import XELA_FLATTEN_ORDER
from tactile_ssl.graph.utils import PHYSICAL_BRIDGE_LINK_PAIRS, iter_sensor_ranges
from tactile_ssl.utils.logging import get_pylogger

from .layers import PatchEmbed1d
from .signal_transformer import SignalTransformer

log = get_pylogger(__name__)


def load_gatv2_conv():
    try:
        from torch_geometric.nn import GATv2Conv
    except ImportError as exc:
        raise ImportError(
            "Xela GATv2 spatial encoders require torch_geometric. "
            "Install PyG in the active environment before using them."
        ) from exc
    return GATv2Conv


def _grid_cells_for_link(link_name: str) -> dict[int, tuple[int, int]]:
    if "aftc" in link_name:
        cells = {}
        for row in range(4):
            for col in range(6):
                cells[row * 6 + col] = (row, col)
        for col in range(1, 5):
            cells[24 + col - 1] = (4, col)
        for col in range(2, 4):
            cells[28 + col - 2] = (5, col)
        return cells
    if "4x4" in link_name:
        return {row * 4 + col: (row, col) for row in range(4) for col in range(4)}
    if "4x6" in link_name:
        return {row * 6 + col: (row, col) for row in range(4) for col in range(6)}
    raise ValueError(f"Unsupported Xela link type: {link_name}")


def _add_undirected_edge(edge_pairs: set[tuple[int, int]], left: int, right: int) -> None:
    if left == right:
        return
    edge_pairs.add((left, right) if left < right else (right, left))


def _local_grid_edge_pairs() -> list[tuple[int, int]]:
    edge_pairs: set[tuple[int, int]] = set()
    for sensor_range in iter_sensor_ranges():
        cells = _grid_cells_for_link(sensor_range.link_name)
        by_cell = {cell: local_id for local_id, cell in cells.items()}
        for local_id, (row, col) in cells.items():
            for neighbor in ((row + 1, col), (row, col + 1)):
                neighbor_id = by_cell.get(neighbor)
                if neighbor_id is not None:
                    _add_undirected_edge(edge_pairs, sensor_range.start + local_id, sensor_range.start + neighbor_id)
    return sorted(edge_pairs)


def _sensor_ranges_by_link():
    return {sensor_range.link_name: sensor_range for sensor_range in iter_sensor_ranges()}


def _nearest_cross_link_edge_pairs(
    positions: torch.Tensor,
    left_ids: list[int],
    right_ids: list[int],
    k: int,
) -> list[tuple[int, int]]:
    if k <= 0:
        return []
    left = torch.tensor(left_ids, dtype=torch.long, device=positions.device)
    right = torch.tensor(right_ids, dtype=torch.long, device=positions.device)
    dist = torch.cdist(positions[left], positions[right])
    flat_order = torch.argsort(dist.flatten()).detach().cpu().tolist()

    used_left: set[int] = set()
    used_right: set[int] = set()
    edge_pairs: set[tuple[int, int]] = set()
    max_edges = min(k, len(left_ids), len(right_ids))
    for flat_id in flat_order:
        i = flat_id // len(right_ids)
        j = flat_id % len(right_ids)
        if i in used_left or j in used_right:
            continue
        _add_undirected_edge(edge_pairs, left_ids[i], right_ids[j])
        used_left.add(i)
        used_right.add(j)
        if len(edge_pairs) >= max_edges:
            break
    return sorted(edge_pairs)


def physical_graph_edge_pairs_torch(positions: torch.Tensor, bridge_k: int = 4) -> list[tuple[int, int]]:
    """Build undirected physical graph edge pairs for one Xela frame.

    The semantics match tactile_ssl.graph.builders.build_physical_graph: sparse grid
    edges inside each pad plus one-to-one nearest bridges for every physically
    adjacent pad pair.
    """
    if positions.shape != (368, 3):
        raise ValueError(f"positions must have shape (368, 3); got {tuple(positions.shape)}")

    edge_pairs = set(_local_grid_edge_pairs())
    ranges_by_link = _sensor_ranges_by_link()
    for left_link, right_link in PHYSICAL_BRIDGE_LINK_PAIRS:
        left_range = ranges_by_link[left_link]
        right_range = ranges_by_link[right_link]
        edge_pairs.update(
            _nearest_cross_link_edge_pairs(
                positions,
                list(left_range.sensor_ids),
                list(right_range.sensor_ids),
                bridge_k,
            )
        )
    return sorted(edge_pairs)


def build_physical_pyg_graph_batch(
    positions: torch.Tensor,
    bridge_k: int = 4,
    edge_mode: Literal["distance", "topology"] = "distance",
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Build a batched directed PyG graph from batched Xela positions."""
    if positions.ndim != 3 or positions.shape[1:] != (368, 3):
        raise ValueError(f"positions must have shape (B, 368, 3); got {tuple(positions.shape)}")
    if edge_mode not in {"distance", "topology"}:
        raise ValueError(f"Unsupported edge_mode={edge_mode!r}")

    edge_indices = []
    edge_attrs = []
    for batch_id in range(positions.shape[0]):
        edge_pairs = physical_graph_edge_pairs_torch(positions[batch_id], bridge_k=bridge_k)
        if not edge_pairs:
            continue
        undirected = torch.tensor(edge_pairs, dtype=torch.long, device=positions.device).t()
        directed = torch.cat([undirected, undirected.flip(0)], dim=1)
        directed = directed + batch_id * positions.shape[1]
        edge_indices.append(directed)

        if edge_mode == "distance":
            src = undirected[0]
            dst = undirected[1]
            distance = torch.linalg.norm(positions[batch_id, src] - positions[batch_id, dst], dim=-1)
            edge_attrs.append(torch.cat([distance, distance], dim=0).unsqueeze(-1))

    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=positions.device)

    if edge_mode == "distance":
        edge_attr = torch.cat(edge_attrs, dim=0) if edge_attrs else torch.zeros((0, 1), device=positions.device)
    else:
        edge_attr = None
    return edge_index, edge_attr


def build_cached_pyg_graph_batch(
    graph_info: dict[str, torch.Tensor],
    num_nodes: int,
    device: torch.device,
    edge_mode: Literal["distance", "topology"] = "distance",
    static_edge_index: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    edge_count_batch = graph_info["edge_count"].to(device=device, dtype=torch.long)
    edge_attr_batch = graph_info.get("edge_attr")
    if edge_attr_batch is not None:
        edge_attr_batch = edge_attr_batch.to(device=device)

    if static_edge_index is not None:
        base_edge_index = static_edge_index.to(device=device, dtype=torch.long)
        batch_size = edge_count_batch.shape[0]
        edge_count = base_edge_index.shape[1]
        if not torch.all(edge_count_batch == edge_count):
            raise ValueError("Static graph edge_count must match the static edge_index size for every batch item")
        offsets = torch.arange(batch_size, device=device, dtype=torch.long).view(batch_size, 1, 1) * num_nodes
        edge_index = (base_edge_index.view(1, 2, edge_count) + offsets).permute(1, 0, 2).reshape(2, -1)
        if edge_mode == "distance":
            if edge_attr_batch is None:
                edge_attr = torch.zeros((batch_size * edge_count, 1), device=device)
            else:
                edge_attr = edge_attr_batch[:, :edge_count].reshape(batch_size * edge_count, -1)
            return edge_index, edge_attr
        return edge_index, None

    edge_index_batch = graph_info["edge_index"].to(device=device, dtype=torch.long)
    if edge_count_batch.numel() > 0 and torch.all(edge_count_batch == edge_count_batch[0]):
        batch_size = edge_index_batch.shape[0]
        edge_count = int(edge_count_batch[0].item())
        base_edge_index = edge_index_batch[:, :, :edge_count]
        offsets = torch.arange(batch_size, device=device, dtype=torch.long).view(batch_size, 1, 1) * num_nodes
        edge_index = (base_edge_index + offsets).permute(1, 0, 2).reshape(2, batch_size * edge_count)
        if edge_mode == "distance":
            if edge_attr_batch is None:
                edge_attr = torch.zeros((batch_size * edge_count, 1), device=device)
            else:
                edge_attr = edge_attr_batch[:, :edge_count].reshape(batch_size * edge_count, -1)
            return edge_index, edge_attr
        return edge_index, None

    edge_indices = []
    edge_attrs = []
    for batch_id in range(edge_index_batch.shape[0]):
        edge_count = int(edge_count_batch[batch_id].item())
        edge_index = edge_index_batch[batch_id, :, :edge_count] + batch_id * num_nodes
        edge_indices.append(edge_index)
        if edge_mode == "distance" and edge_attr_batch is not None:
            edge_attrs.append(edge_attr_batch[batch_id, :edge_count])

    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    if edge_mode == "distance":
        edge_attr = torch.cat(edge_attrs, dim=0) if edge_attrs else torch.zeros((0, 1), device=device)
    else:
        edge_attr = None
    return edge_index, edge_attr


def build_static_edge_attr(positions: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros((positions.shape[0], 0, 1), dtype=positions.dtype, device=positions.device)
    src, dst = edge_index[0], edge_index[1]
    distance = torch.linalg.norm(positions[:, src] - positions[:, dst], dim=-1)
    return distance.unsqueeze(-1)


class XelaSpatialGNNTransformer(SignalTransformer):
    def __init__(
        self,
        in_dim: int,
        in_chans: int,
        time_chunk_size: int,
        sequence_length: int,
        embed_dim: int = 192,
        signal_chans: int = 3,
        pos_chans: int = 3,
        signal_embed_dim: int = 192,
        spatial_embed_dim: int = 192,
        spatial_gat_heads: int = 4,
        spatial_gat_dropout: float = 0.0,
        graph_type: Literal["physical"] = "physical",
        bridge_k: int = 4,
        edge_mode: Literal["distance", "topology"] = "distance",
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        ffn_layer: str = "mlp",
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        head: Optional[nn.Module] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        pos_embed_fn: Literal["sinusoidal", "learned"] = "learned",
        init_values: Optional[float] = None,
        num_register_tokens: int = 0,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        with_masktoken: bool = False,
        causal: bool = False,
        normalization: Optional[DictConfig] = None,
        use_taxel_type_embedding: bool = True,
    ):
        if in_chans != signal_chans + pos_chans:
            raise ValueError("in_chans must equal signal_chans + pos_chans")
        if in_dim != 368:
            raise ValueError("XelaSpatialGNNTransformer currently expects 368 Xela sensors")
        if embed_dim != signal_embed_dim or embed_dim != spatial_embed_dim:
            raise ValueError("embed_dim, signal_embed_dim, and spatial_embed_dim must match for additive fusion")
        if graph_type != "physical":
            raise ValueError("Only graph_type='physical' is supported")
        if spatial_embed_dim % spatial_gat_heads != 0:
            raise ValueError("spatial_embed_dim must be divisible by spatial_gat_heads")

        self.signal_chans = signal_chans
        self.pos_chans = pos_chans
        self.signal_embed_dim = signal_embed_dim
        self.spatial_embed_dim = spatial_embed_dim
        self.graph_type = graph_type
        self.bridge_k = int(bridge_k)
        self.edge_mode = edge_mode
        self.supports_graph_info = True

        super().__init__(
            in_dim=in_dim,
            in_chans=in_chans,
            time_chunk_size=time_chunk_size,
            sequence_length=sequence_length,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            ffn_layer=ffn_layer,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            head=head,
            act_layer=act_layer,
            norm_layer=norm_layer,
            pos_embed_fn=pos_embed_fn,
            init_values=init_values,
            num_register_tokens=num_register_tokens,
            drop_path_rate=drop_path_rate,
            drop_path_uniform=drop_path_uniform,
            with_masktoken=with_masktoken,
            causal=causal,
        )

        if normalization is not None:
            self.register_buffer("xela_mean", torch.as_tensor(normalization.mean, dtype=torch.float32))
            self.register_buffer("xela_std", torch.as_tensor(normalization.std, dtype=torch.float32))
        else:
            self.register_buffer("xela_mean", torch.zeros(self.signal_chans, dtype=torch.float32))
            self.register_buffer("xela_std", torch.ones(self.signal_chans, dtype=torch.float32))
        print(f"Xela mean: {self.xela_mean}, Xela std: {self.xela_std}")

        self.patch_embed = PatchEmbed1d(
            modal_chans=self.signal_chans,
            modal_lens=sequence_length,
            chunk_size=self.time_chunk_size,
            embed_dim=self.signal_embed_dim,
        )
        self.use_taxel_type_embedding = bool(use_taxel_type_embedding)
        self.taxeltype_embed = nn.Parameter(
            torch.zeros(3, self.signal_embed_dim),
            requires_grad=self.use_taxel_type_embedding,
        )

        gat_out_channels = spatial_embed_dim // spatial_gat_heads
        GATv2Conv = load_gatv2_conv()
        edge_dim = 1 if edge_mode == "distance" else None
        self.spatial_gnn_1 = GATv2Conv(
            pos_chans,
            gat_out_channels,
            heads=spatial_gat_heads,
            concat=True,
            dropout=spatial_gat_dropout,
            edge_dim=edge_dim,
        )
        self.spatial_gnn_2 = GATv2Conv(
            spatial_embed_dim,
            gat_out_channels,
            heads=spatial_gat_heads,
            concat=True,
            dropout=spatial_gat_dropout,
            edge_dim=edge_dim,
        )
        self.spatial_activation = nn.ELU()
        self.register_buffer("static_physical_edge_index", torch.empty((2, 0), dtype=torch.long), persistent=False)

        if self.use_taxel_type_embedding:
            nn.init.trunc_normal_(self.taxeltype_embed, std=0.02)
        self.init_weights()

    def update_stats(self, xela_mean, xela_std):
        assert isinstance(xela_mean, torch.Tensor) and isinstance(xela_std, torch.Tensor)
        assert xela_mean.shape[-1] == xela_std.shape[-1] == self.signal_chans
        self.xela_mean = xela_mean
        self.xela_std = xela_std

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "xela_mean") and hasattr(self, "xela_std"):
            mean = self.xela_mean.to(device=x.device, dtype=x.dtype)
            std = self.xela_std.to(device=x.device, dtype=x.dtype)
            signal = (x[..., : self.signal_chans] - mean) / std
            pos = x[..., self.signal_chans :]
            return torch.cat([signal, pos], dim=-1)
        return x

    def normalize_signal(self, signal: torch.Tensor) -> torch.Tensor:
        return self.normalize(signal)[..., : self.signal_chans]

    def signal_pre_embed(self, signal: torch.Tensor) -> torch.Tensor:
        b = signal.shape[0]
        signal = self.normalize_signal(signal)
        signal = einops.rearrange(signal, "b t n c -> (b n) c t")
        signal_embed = self.patch_embed(signal)
        signal_embed = einops.rearrange(signal_embed, "(b n) c t -> b t n c", b=b)

        if self.use_taxel_type_embedding:
            prev_idx = 0
            for key, count in XELA_FLATTEN_ORDER.items():
                if "4x4" in key:
                    taxeltype_embed = self.taxeltype_embed[0]
                elif "4x6" in key:
                    taxeltype_embed = self.taxeltype_embed[1]
                elif "aftc" in key:
                    taxeltype_embed = self.taxeltype_embed[2]
                else:
                    raise ValueError(f"Bad taxel type for {key}")
                signal_embed[..., prev_idx : prev_idx + count, :] += taxeltype_embed[None, None, :]
                prev_idx += count
        return signal_embed

    def _get_static_physical_edge_index(self, pos_ref: torch.Tensor, graph_info: Optional[dict] = None) -> torch.Tensor:
        if self.static_physical_edge_index.numel() > 0:
            return self.static_physical_edge_index.to(device=pos_ref.device)

        if graph_info is not None and graph_info.get("edge_index") is not None:
            edge_count = int(graph_info["edge_count"].reshape(-1)[0].item())
            edge_index = graph_info["edge_index"].reshape(-1, 2, graph_info["edge_index"].shape[-1])[0, :, :edge_count]
            edge_index = edge_index.detach().to(device=pos_ref.device, dtype=torch.long)
        else:
            edge_pairs = physical_graph_edge_pairs_torch(pos_ref[0], bridge_k=self.bridge_k)
            if edge_pairs:
                undirected = torch.tensor(edge_pairs, dtype=torch.long, device=pos_ref.device).t()
                edge_index = torch.cat([undirected, undirected.flip(0)], dim=1)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=pos_ref.device)

        self.static_physical_edge_index = edge_index.detach()
        return self.static_physical_edge_index.to(device=pos_ref.device)

    def spatial_pre_embed(self, pos: torch.Tensor, num_chunks: int, graph_info: Optional[dict] = None) -> torch.Tensor:
        pos_ref = pos.mean(dim=1)
        if graph_info is None:
            static_edge_index = self._get_static_physical_edge_index(pos_ref)
            edge_count = static_edge_index.shape[1]
            graph_info = {
                "edge_count": torch.full((pos_ref.shape[0],), edge_count, dtype=torch.long, device=pos_ref.device),
            }
            if self.edge_mode == "distance":
                graph_info["edge_attr"] = build_static_edge_attr(pos_ref, static_edge_index)
            edge_index, edge_attr = build_cached_pyg_graph_batch(
                graph_info,
                num_nodes=pos.shape[2],
                device=pos.device,
                edge_mode=self.edge_mode,
                static_edge_index=static_edge_index,
            )
        else:
            static_edge_index = None
            if graph_info.get("edge_index") is None:
                static_edge_index = self._get_static_physical_edge_index(pos_ref)
            edge_index, edge_attr = build_cached_pyg_graph_batch(
                graph_info,
                num_nodes=pos.shape[2],
                device=pos.device,
                edge_mode=self.edge_mode,
                static_edge_index=static_edge_index,
            )
        node_features = pos_ref.reshape(-1, self.pos_chans)
        if self.edge_mode == "distance":
            spatial = self.spatial_gnn_1(node_features, edge_index, edge_attr)
            spatial = self.spatial_activation(spatial)
            spatial = self.spatial_gnn_2(spatial, edge_index, edge_attr)
        else:
            spatial = self.spatial_gnn_1(node_features, edge_index)
            spatial = self.spatial_activation(spatial)
            spatial = self.spatial_gnn_2(spatial, edge_index)
        spatial = spatial.view(pos.shape[0], pos.shape[2], self.spatial_embed_dim)
        return einops.repeat(spatial, "b n c -> b t n c", t=num_chunks)

    def pre_embed(self, x: torch.Tensor, graph_info: Optional[dict] = None) -> torch.Tensor:
        signal = x[..., : self.signal_chans]
        pos = x[..., self.signal_chans :]
        signal_embed = self.signal_pre_embed(signal)
        spatial_embed = self.spatial_pre_embed(pos, num_chunks=signal_embed.shape[1], graph_info=graph_info)
        return signal_embed + spatial_embed

    def forward_features(
        self,
        x,
        masks: Optional[List[torch.Tensor]] = None,
        mask_type: Optional[Literal["block", "tubelet"]] = None,
        masktoken_masks: Optional[List[torch.Tensor]] = None,
        graph_info: Optional[dict] = None,
    ):
        x = self.pre_embed(x, graph_info=graph_info)
        x, bias = self.prepare_tokens_with_mask(x, masks, mask_type, masktoken_masks)
        x_prenorm, x_postnorm = self.transform(x, bias)

        reg_tokens = x_postnorm[:, : self.num_register_tokens]
        patch_tokens = x_postnorm[:, self.num_register_tokens :]
        patch_tokens_prenorm = x_prenorm[:, self.num_register_tokens :]
        out = {
            "x_norm_regtokens": reg_tokens,
            "x_norm_patchtokens": patch_tokens,
            "x_prenorm": patch_tokens_prenorm,
        }
        return out

    def forward(self, x, masks=None, mask_type=None, masktoken_masks=None, graph_info: Optional[dict] = None):
        out = self.forward_features(x, masks, mask_type, masktoken_masks, graph_info=graph_info)
        return self.head(out["x_norm_patchtokens"])
