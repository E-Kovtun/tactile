from functools import partial
from typing import Callable, List, Literal, Optional, Sequence

import einops
import torch
import torch.nn as nn
from omegaconf import DictConfig

from tactile_ssl.data.xela.utils import XELA_FLATTEN_ORDER

from .layers import PatchEmbed1d
from .signal_transformer import SignalTransformer
from .xela_spatial_gnn import physical_graph_edge_pairs_torch


def _load_wl_conv_continuous():
    try:
        from torch_geometric.nn import WLConvContinuous
    except ImportError as exc:
        raise ImportError(
            "XelaSpatialWLDINOTransformer requires torch_geometric. "
            "Install PyG in the active environment before using the spatial WL DINO backbone."
        ) from exc
    return WLConvContinuous


class XelaSpatialWLDINOTransformer(SignalTransformer):
    """Xela DINO backbone with crop-local physical-graph coordinate diffusion.

    DINO crops are applied before the spatial branch. Consequently, every WL
    layer sees only the induced physical graph of the sensors retained by that
    crop and cannot pass information through removed sensors.
    """

    def __init__(
        self,
        in_dim: int,
        in_chans: int,
        time_chunk_size: int,
        sequence_length: int,
        embed_dim: int = 192,
        signal_chans: int = 3,
        pos_chans: int = 3,
        spatial_embed_dim: int = 192,
        spatial_wl_layers: int = 2,
        bridge_k: int = 4,
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
    ) -> None:
        if in_dim != 368:
            raise ValueError(f"XelaSpatialWLDINOTransformer expects 368 Xela sensors; got {in_dim}")
        if in_chans != signal_chans + pos_chans:
            raise ValueError(
                "in_chans must equal signal_chans + pos_chans; "
                f"got {in_chans}, {signal_chans}, and {pos_chans}"
            )
        if signal_chans != 3 or pos_chans != 3:
            raise ValueError("Xela spatial WL DINO expects three signal and three XYZ channels")
        if spatial_embed_dim != embed_dim:
            raise ValueError(
                "spatial_embed_dim must equal embed_dim for additive fusion; "
                f"got {spatial_embed_dim} and {embed_dim}"
            )
        if spatial_wl_layers <= 0:
            raise ValueError("spatial_wl_layers must be positive")
        if bridge_k < 0:
            raise ValueError("bridge_k must be non-negative")

        self.signal_chans = int(signal_chans)
        self.pos_chans = int(pos_chans)
        self.spatial_embed_dim = int(spatial_embed_dim)
        self.spatial_wl_layers = int(spatial_wl_layers)
        self.bridge_k = int(bridge_k)
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

        self.patch_embed = PatchEmbed1d(
            modal_chans=self.signal_chans,
            modal_lens=sequence_length,
            chunk_size=time_chunk_size,
            embed_dim=embed_dim,
        )
        self.use_taxel_type_embedding = bool(use_taxel_type_embedding)
        self.taxeltype_embed = nn.Parameter(
            torch.zeros(3, embed_dim),
            requires_grad=self.use_taxel_type_embedding,
        )
        self.spatial_projection = nn.Linear(self.pos_chans, self.spatial_embed_dim)
        self.spatial_activation = nn.GELU()
        WLConvContinuous = _load_wl_conv_continuous()
        self.wl_layers = nn.ModuleList(WLConvContinuous() for _ in range(self.spatial_wl_layers))
        self.register_buffer(
            "static_physical_edge_index",
            torch.empty((2, 0), dtype=torch.long),
            persistent=False,
        )

        if self.use_taxel_type_embedding:
            nn.init.trunc_normal_(self.taxeltype_embed, std=0.02)
        self.init_weights()

    def update_stats(self, xela_mean: torch.Tensor, xela_std: torch.Tensor) -> None:
        if not isinstance(xela_mean, torch.Tensor) or not isinstance(xela_std, torch.Tensor):
            raise TypeError("xela_mean and xela_std must be tensors")
        if xela_mean.shape[-1] != self.signal_chans or xela_std.shape[-1] != self.signal_chans:
            raise ValueError(f"normalization statistics must have {self.signal_chans} channels")
        self.xela_mean = xela_mean
        self.xela_std = xela_std

    def normalize_signal(self, signal: torch.Tensor) -> torch.Tensor:
        mean = self.xela_mean.to(device=signal.device, dtype=signal.dtype)
        std = self.xela_std.to(device=signal.device, dtype=signal.dtype)
        return (signal - mean) / std

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize tactile channels while leaving XYZ in the physical frame."""
        self._validate_input(x)
        signal = self.normalize_signal(x[..., : self.signal_chans])
        return torch.cat([signal, x[..., self.signal_chans :]], dim=-1)

    def signal_pre_embed(self, signal: torch.Tensor) -> torch.Tensor:
        batch_size = signal.shape[0]
        signal = self.normalize_signal(signal)
        signal = einops.rearrange(signal, "b t n c -> (b n) c t")
        signal_embed = self.patch_embed(signal)
        signal_embed = einops.rearrange(signal_embed, "(b n) c t -> b t n c", b=batch_size)

        if self.use_taxel_type_embedding:
            start = 0
            for link_name, count in XELA_FLATTEN_ORDER.items():
                if "4x4" in link_name:
                    type_id = 0
                elif "4x6" in link_name:
                    type_id = 1
                elif "aftc" in link_name:
                    type_id = 2
                else:
                    raise ValueError(f"Unsupported Xela taxel type for {link_name}")
                signal_embed[..., start : start + count, :] += self.taxeltype_embed[type_id][None, None, :]
                start += count
        return signal_embed

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"input must have shape [B,T,368,6]; got {tuple(x.shape)}")
        if x.shape[2] != self.in_dim:
            raise ValueError(f"input must contain {self.in_dim} Xela sensors; got {x.shape[2]}")
        if x.shape[3] != self.in_chans:
            raise ValueError(f"input must contain {self.in_chans} channels; got {x.shape[3]}")
        if not torch.isfinite(x[..., self.signal_chans :]).all():
            raise ValueError("spatial coordinates must be finite")

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

    def _full_edge_indices(
        self,
        positions: torch.Tensor,
        graph_info: Optional[dict[str, torch.Tensor]],
    ) -> List[torch.Tensor]:
        batch_size = positions.shape[0]
        if graph_info is None:
            static_edges = self._get_static_physical_edge_index(positions)
            return [static_edges for _ in range(batch_size)]

        if "edge_count" not in graph_info:
            raise ValueError("graph_info must contain edge_count")
        edge_count = graph_info["edge_count"].to(device=positions.device, dtype=torch.long)
        if edge_count.ndim != 1:
            raise ValueError(f"graph_info edge_count must be one-dimensional; got {tuple(edge_count.shape)}")
        if edge_count.numel() != batch_size:
            raise ValueError(
                "graph_info and input must contain the same number of graphs; "
                f"got {edge_count.numel()} and {batch_size}"
            )

        edge_index_batch = graph_info.get("edge_index")
        if edge_index_batch is None:
            static_edges = self._get_static_physical_edge_index(positions)
            if not torch.all(edge_count == static_edges.shape[1]):
                raise ValueError("static graph edge_count does not match the reconstructed physical topology")
            return [static_edges for _ in range(batch_size)]

        edge_index_batch = edge_index_batch.to(device=positions.device, dtype=torch.long)
        if edge_index_batch.ndim != 3 or edge_index_batch.shape[:2] != (batch_size, 2):
            raise ValueError(
                "graph_info edge_index must have shape [B,2,E]; "
                f"got {tuple(edge_index_batch.shape)}"
            )

        edge_indices = []
        for batch_id in range(batch_size):
            count = int(edge_count[batch_id].item())
            if count < 0 or count > edge_index_batch.shape[-1]:
                raise ValueError(f"invalid edge_count={count} for graph {batch_id}")
            edge_index = edge_index_batch[batch_id, :, :count]
            if edge_index.numel() > 0 and (edge_index.min() < 0 or edge_index.max() >= self.in_dim):
                raise ValueError("graph_info edge_index contains a node outside the Xela sensor range")
            edge_indices.append(edge_index)
        return edge_indices

    def _validate_masks(self, masks: Sequence[torch.Tensor], batch_size: int) -> None:
        if len(masks) == 0:
            raise ValueError("at least one crop mask is required")
        crop_size = masks[0].shape[-1]
        if crop_size < 1:
            raise ValueError("crop masks must retain at least one sensor")
        for mask in masks:
            if mask.ndim != 2 or mask.shape[0] != batch_size:
                raise ValueError(f"each crop mask must have shape [B,K]; got {tuple(mask.shape)}")
            if mask.shape[1] != crop_size:
                raise ValueError("all crop masks in one forward pass must retain the same number of sensors")
            if mask.dtype not in (torch.int32, torch.int64):
                raise ValueError("crop masks must contain integer sensor indices")
            if mask.numel() > 0 and (mask.min() < 0 or mask.max() >= self.in_dim):
                raise ValueError("crop mask contains a sensor index outside the Xela sensor range")
            sorted_mask = torch.sort(mask, dim=1).values
            if crop_size > 1 and torch.any(sorted_mask[:, 1:] == sorted_mask[:, :-1]):
                raise ValueError("crop masks must not contain duplicate sensor indices")

    def _induced_edges(
        self,
        full_edge_indices: Sequence[torch.Tensor],
        mask: torch.Tensor,
        graph_offset: int,
    ) -> torch.Tensor:
        crop_size = mask.shape[1]
        batch_size = mask.shape[0]
        remap = torch.full((batch_size, self.in_dim), -1, dtype=torch.long, device=mask.device)
        local_ids = torch.arange(crop_size, device=mask.device).expand(batch_size, -1)
        remap.scatter_(1, mask, local_ids)

        edge_counts = [edge_index.shape[1] for edge_index in full_edge_indices]
        if len(set(edge_counts)) == 1:
            edge_index_batch = torch.stack(list(full_edge_indices), dim=0)
            src = torch.gather(remap, 1, edge_index_batch[:, 0])
            dst = torch.gather(remap, 1, edge_index_batch[:, 1])
            keep = (src >= 0) & (dst >= 0)
            batch_ids, edge_ids = torch.nonzero(keep, as_tuple=True)
            offsets = graph_offset + batch_ids * crop_size
            return torch.stack([src[batch_ids, edge_ids] + offsets, dst[batch_ids, edge_ids] + offsets])

        # Per-window graphs may contain different edge counts. This fallback is
        # not used by the static-edge configs, but keeps the public interface
        # compatible with cached dynamic graphs.
        induced = []
        for batch_id, full_edge_index in enumerate(full_edge_indices):
            relabeled = remap[batch_id, full_edge_index]
            keep = (relabeled[0] >= 0) & (relabeled[1] >= 0)
            induced.append(relabeled[:, keep] + graph_offset + batch_id * crop_size)
        return (
            torch.cat(induced, dim=1)
            if induced
            else torch.empty((2, 0), dtype=torch.long, device=mask.device)
        )

    def _run_wl(self, positions: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial_activation(self.spatial_projection(positions.reshape(-1, self.pos_chans)))
        for wl_layer in self.wl_layers:
            spatial = wl_layer(spatial, edge_index)
        return spatial.view(*positions.shape[:-1], self.spatial_embed_dim)

    def spatial_pre_embed(
        self,
        positions: torch.Tensor,
        graph_info: Optional[dict[str, torch.Tensor]] = None,
        masks: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Return full-graph or crop-local spatial embeddings without time repetition."""
        if positions.ndim != 3 or positions.shape[1:] != (self.in_dim, self.pos_chans):
            raise ValueError(
                f"positions must have shape [B,{self.in_dim},{self.pos_chans}]; got {tuple(positions.shape)}"
            )
        if not torch.isfinite(positions).all():
            raise ValueError("spatial coordinates must be finite")
        full_edge_indices = self._full_edge_indices(positions, graph_info)
        batch_size = positions.shape[0]

        if masks is None:
            edge_counts = [edge_index.shape[1] for edge_index in full_edge_indices]
            if len(set(edge_counts)) == 1:
                edge_index_batch = torch.stack(list(full_edge_indices), dim=0)
                offsets = torch.arange(batch_size, device=positions.device).view(batch_size, 1, 1) * self.in_dim
                edge_index = (edge_index_batch + offsets).permute(1, 0, 2).reshape(2, -1)
            else:
                edges = [
                    edge_index + batch_id * self.in_dim
                    for batch_id, edge_index in enumerate(full_edge_indices)
                ]
                edge_index = torch.cat(edges, dim=1)
            return self._run_wl(positions, edge_index)

        masks = list(masks)
        self._validate_masks(masks, batch_size)
        crop_size = masks[0].shape[1]
        crop_positions = []
        crop_edges = []
        for view_id, mask in enumerate(masks):
            gather_index = mask.unsqueeze(-1).expand(-1, -1, self.pos_chans)
            crop_positions.append(torch.gather(positions, dim=1, index=gather_index))
            crop_edges.append(
                self._induced_edges(
                    full_edge_indices,
                    mask,
                    graph_offset=view_id * batch_size * crop_size,
                )
            )
        positions_cat = torch.cat(crop_positions, dim=0)
        edge_index = (
            torch.cat(crop_edges, dim=1)
            if crop_edges
            else torch.empty((2, 0), dtype=torch.long, device=positions.device)
        )
        return self._run_wl(positions_cat, edge_index)

    def pre_embed(self, x: torch.Tensor, graph_info: Optional[dict[str, torch.Tensor]] = None) -> torch.Tensor:
        self._validate_input(x)
        signal_embed = self.signal_pre_embed(x[..., : self.signal_chans])
        positions = x[..., self.signal_chans :].mean(dim=1)
        spatial_embed = self.spatial_pre_embed(positions, graph_info=graph_info)
        spatial_embed = einops.repeat(spatial_embed, "b n c -> b t n c", t=signal_embed.shape[1])
        return signal_embed + spatial_embed

    def _prepare_cropped_tokens(
        self,
        signal_embed: torch.Tensor,
        spatial_embed: torch.Tensor,
        masks: Sequence[torch.Tensor],
        masktoken_masks: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        cropped_signal = self.apply_tubelet_masks(signal_embed, masks)
        spatial_embed = einops.repeat(spatial_embed, "b n c -> b t n c", t=signal_embed.shape[1])
        tokens = cropped_signal + spatial_embed

        if self.pos_embed_fn == "sinusoidal":
            pos_embed = self.pos_embed(tokens.device).float().unsqueeze(0)
        elif self.pos_embed_fn == "learned":
            pos_embed = self.pos_embed.float()
        else:
            raise NotImplementedError(f"Unsupported positional embedding {self.pos_embed_fn!r}")
        pos_embed = einops.rearrange(pos_embed, "1 (t n) c -> 1 t n c", n=self.in_dim)
        pos_embed = pos_embed[:, : signal_embed.shape[1]].expand(signal_embed.shape[0], -1, -1, -1)
        tokens = tokens + self.apply_tubelet_masks(pos_embed, masks)

        if masktoken_masks is not None:
            tokens = self.apply_masktokens(tokens, masktoken_masks)
        bias = self.create_causal_mask(tokens) if self.causal else None
        tokens = einops.rearrange(tokens, "b t n c -> b (t n) c")
        if self.register_tokens is not None:
            tokens = torch.cat([self.register_tokens.expand(tokens.shape[0], -1, -1), tokens], dim=1)
        return tokens, bias

    def forward_features(
        self,
        x: torch.Tensor,
        masks: Optional[Sequence[torch.Tensor]] = None,
        mask_type: Optional[Literal["block", "tubelet"]] = None,
        masktoken_masks: Optional[torch.Tensor] = None,
        graph_info: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        self._validate_input(x)
        if masks is not None and mask_type != "tubelet":
            raise ValueError("Xela spatial WL DINO supports crop-local graph construction only for tubelet masks")

        if masks is None:
            tokens = self.pre_embed(x, graph_info=graph_info)
            tokens, bias = self.prepare_tokens_with_mask(tokens, None, mask_type, masktoken_masks)
        else:
            masks = list(masks)
            signal_embed = self.signal_pre_embed(x[..., : self.signal_chans])
            positions = x[..., self.signal_chans :].mean(dim=1)
            spatial_embed = self.spatial_pre_embed(positions, graph_info=graph_info, masks=masks)
            tokens, bias = self._prepare_cropped_tokens(signal_embed, spatial_embed, masks, masktoken_masks)

        x_prenorm, x_postnorm = self.transform(tokens, bias)
        reg_tokens = x_postnorm[:, : self.num_register_tokens]
        patch_tokens = x_postnorm[:, self.num_register_tokens :]
        patch_tokens_prenorm = x_prenorm[:, self.num_register_tokens :]
        return {
            "x_norm_regtokens": reg_tokens,
            "x_norm_patchtokens": patch_tokens,
            "x_prenorm": patch_tokens_prenorm,
        }

    def forward(
        self,
        x: torch.Tensor,
        masks: Optional[Sequence[torch.Tensor]] = None,
        mask_type: Optional[Literal["block", "tubelet"]] = None,
        masktoken_masks: Optional[torch.Tensor] = None,
        graph_info: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        out = self.forward_features(
            x,
            masks=masks,
            mask_type=mask_type,
            masktoken_masks=masktoken_masks,
            graph_info=graph_info,
        )
        return self.head(out["x_norm_patchtokens"])
