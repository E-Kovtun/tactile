from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, Sequence

import numpy as np
import torch

from tactile_ssl.graph.types import WeightedSensorGraph


TAXELS_PER_HAND = 1062
HYPERTAXELS_PER_HAND = 264
NUM_HANDS = 2
NUM_RAW_TAXELS = NUM_HANDS * TAXELS_PER_HAND
NUM_HYPERTAXELS = NUM_HANDS * HYPERTAXELS_PER_HAND


@dataclass(frozen=True)
class DecoRegionSpec:
    key: str
    label: str
    offset: int
    rows: int
    cols: int
    x: float
    y: float
    width: float
    height: float
    angle: float = 0.0

    @property
    def size(self) -> int:
        return self.rows * self.cols

    @property
    def num_hypertaxels(self) -> int:
        if (self.rows, self.cols) == (3, 3):
            return 2
        if self.rows % 2 or self.cols % 2:
            raise ValueError(
                f"Region {self.key!r} cannot be partitioned into 2x2 blocks: "
                f"{self.rows}x{self.cols}"
            )
        return (self.rows // 2) * (self.cols // 2)


# Array slices follow the public DECO pressure-vector order. Coordinates are a
# canonical 2-D hand layout used only to choose and visualize inter-region
# bridges; JEPA edge weights are topological unit distances.
DECO_REGION_SPECS: tuple[DecoRegionSpec, ...] = (
    DecoRegionSpec("f1_tip", "F1 tip", 0, 3, 3, 0.55, 8.05, 0.72, 0.72, -0.05),
    DecoRegionSpec("f1_top", "F1 upper", 9, 12, 8, 0.43, 5.35, 1.05, 2.35, -0.05),
    DecoRegionSpec("f1_palm", "F1 lower", 105, 10, 8, 0.48, 3.05, 1.08, 1.95, -0.05),
    DecoRegionSpec("f2_tip", "F2 tip", 185, 3, 3, 1.92, 8.68, 0.72, 0.72, -0.015),
    DecoRegionSpec("f2_top", "F2 upper", 194, 12, 8, 1.80, 5.86, 1.05, 2.45, -0.015),
    DecoRegionSpec("f2_palm", "F2 lower", 290, 10, 8, 1.83, 3.24, 1.08, 2.12, -0.015),
    DecoRegionSpec("f3_tip", "F3 tip", 370, 3, 3, 3.31, 8.91, 0.72, 0.72, 0.015),
    DecoRegionSpec("f3_top", "F3 upper", 379, 12, 8, 3.20, 5.98, 1.05, 2.55, 0.015),
    DecoRegionSpec("f3_palm", "F3 lower", 475, 10, 8, 3.22, 3.28, 1.08, 2.18, 0.015),
    DecoRegionSpec("f4_tip", "F4 tip", 555, 3, 3, 4.70, 8.38, 0.72, 0.72, 0.05),
    DecoRegionSpec("f4_top", "F4 upper", 564, 12, 8, 4.58, 5.60, 1.05, 2.42, 0.05),
    DecoRegionSpec("f4_palm", "F4 lower", 660, 10, 8, 4.59, 3.14, 1.08, 2.04, 0.05),
    DecoRegionSpec("thumb_tip", "Thumb tip", 740, 3, 3, 7.78, 6.14, 0.72, 0.72, -0.62),
    DecoRegionSpec("thumb_top", "Thumb upper", 749, 12, 8, 6.57, 4.11, 1.05, 2.30, -0.62),
    DecoRegionSpec("thumb_middle", "Thumb middle", 845, 3, 3, 6.13, 3.40, 0.72, 0.72, -0.62),
    DecoRegionSpec("thumb_palm", "Thumb lower", 854, 12, 8, 4.99, 1.34, 1.08, 2.22, -0.62),
    DecoRegionSpec("palm", "Palm", 950, 8, 14, 0.86, 0.20, 4.55, 2.38),
)


DECO_REGION_ADJACENCY: tuple[tuple[str, str], ...] = (
    ("f1_tip", "f1_top"),
    ("f1_top", "f1_palm"),
    ("f1_palm", "palm"),
    ("f2_tip", "f2_top"),
    ("f2_top", "f2_palm"),
    ("f2_palm", "palm"),
    ("f3_tip", "f3_top"),
    ("f3_top", "f3_palm"),
    ("f3_palm", "palm"),
    ("f4_tip", "f4_top"),
    ("f4_top", "f4_palm"),
    ("f4_palm", "palm"),
    ("thumb_tip", "thumb_top"),
    ("thumb_top", "thumb_middle"),
    ("thumb_middle", "thumb_palm"),
    ("thumb_palm", "palm"),
)


# Explicit target slots from the user-supplied palm sketch. The canonical
# geometry has the thumb on the right, so this is the horizontal mirror of the
# sketch (where the blue thumb is drawn on the left). Repeated coordinates are
# intentional: two different finger nodes may terminate at one palm node.
# Negative indices follow NumPy grid indexing.
DECO_PALM_ATTACHMENT_TARGETS: dict[str, tuple[tuple[int, int], ...]] = {
    # F1 is farthest from the thumb: wrap around the upper-left corner.
    "f1_palm": ((-1, 0), (-1, 0), (-2, 0), (-2, 0)),
    "f2_palm": ((-1, 1), (-1, 1), (-1, 2), (-1, 2)),
    "f3_palm": ((-1, 3), (-1, 3), (-1, 4), (-1, 4)),
    "f4_palm": ((-1, 5), (-1, 5), (-1, 6), (-1, 6)),
    # The thumb terminates at the two middle nodes of the four-node outer side.
    "thumb_palm": ((1, -1), (1, -1), (2, -1), (2, -1)),
}

DECO_FINGER_SEGMENT_ATTACHMENTS: tuple[tuple[str, str], ...] = (
    ("f1_top", "f1_palm"),
    ("f2_top", "f2_palm"),
    ("f3_top", "f3_palm"),
    ("f4_top", "f4_palm"),
)


@dataclass(frozen=True)
class DecoGeometry:
    member_indices: np.ndarray
    member_mask: np.ndarray
    group_size: np.ndarray
    hand_id: np.ndarray
    region_id: np.ndarray
    raw_to_hypertaxel: np.ndarray
    raw_hand_id: np.ndarray
    raw_region_id: np.ndarray
    raw_positions: np.ndarray
    hypertaxel_positions: np.ndarray
    raw_graph: WeightedSensorGraph
    coarse_graph: WeightedSensorGraph
    raw_bridge_mask: np.ndarray
    coarse_bridge_mask: np.ndarray
    bridge_k: int

    @property
    def num_raw_taxels(self) -> int:
        return int(self.raw_to_hypertaxel.size)

    @property
    def num_hypertaxels(self) -> int:
        return int(self.member_indices.shape[0])

    def group_hands(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        padding_value: float = 0.0,
    ) -> np.ndarray:
        """Gather two ``[..., 1062]`` arrays into ``[..., 528, 5]``.

        Numeric zero remains a valid pressure value. Padding is determined only
        by ``member_mask`` and is never inferred from signal values.
        """
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        if left_array.shape != right_array.shape:
            raise ValueError(
                f"left and right tactile arrays must have equal shapes; "
                f"got {left_array.shape} and {right_array.shape}"
            )
        if left_array.ndim == 0 or left_array.shape[-1] != TAXELS_PER_HAND:
            raise ValueError(
                f"each hand must end in {TAXELS_PER_HAND} taxels; got {left_array.shape}"
            )
        raw = np.concatenate([left_array, right_array], axis=-1)
        safe_indices = np.where(self.member_mask, self.member_indices, 0)
        grouped = np.take(raw, safe_indices, axis=-1)
        broadcast_mask = self.member_mask.reshape(
            (1,) * (grouped.ndim - self.member_mask.ndim) + self.member_mask.shape
        )
        return np.where(broadcast_mask, grouped, padding_value)

    def graph_dict(self) -> dict[str, torch.Tensor]:
        return {
            "edge_index": torch.from_numpy(self.coarse_graph.edge_index.copy()).long(),
            "edge_attr": torch.from_numpy(self.coarse_graph.edge_weight.copy()).float(),
            "edge_count": torch.tensor(
                self.coarse_graph.edge_index.shape[1], dtype=torch.long
            ),
            "node_group_id": torch.from_numpy(self.hand_id.copy()).long(),
            "region_id": torch.from_numpy(self.region_id.copy()).long(),
        }


def _rotate_grid(spec: DecoRegionSpec) -> np.ndarray:
    cols = spec.x + (np.arange(spec.cols, dtype=np.float32) + 0.5) * spec.width / spec.cols
    rows = spec.y + (np.arange(spec.rows, dtype=np.float32) + 0.5) * spec.height / spec.rows
    x, y = np.meshgrid(cols, rows)
    points = np.column_stack([x.reshape(-1), y.reshape(-1)])
    if spec.angle == 0:
        return points.astype(np.float32, copy=False)
    center = np.asarray(
        [spec.x + spec.width / 2, spec.y + spec.height / 2], dtype=np.float32
    )
    cosine, sine = np.cos(spec.angle), np.sin(spec.angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    return ((points - center) @ rotation.T + center).astype(np.float32, copy=False)


def _local_groups(spec: DecoRegionSpec) -> list[np.ndarray]:
    if (spec.rows, spec.cols) == (3, 3):
        return [
            np.asarray([0, 1, 3, 4], dtype=np.int64),
            np.asarray([2, 5, 6, 7, 8], dtype=np.int64),
        ]
    groups = []
    for row in range(0, spec.rows, 2):
        for col in range(0, spec.cols, 2):
            groups.append(
                np.asarray(
                    [
                        row * spec.cols + col,
                        row * spec.cols + col + 1,
                        (row + 1) * spec.cols + col,
                        (row + 1) * spec.cols + col + 1,
                    ],
                    dtype=np.int64,
                )
            )
    return groups


def _coarse_grid(spec: DecoRegionSpec, group_ids: Sequence[int]) -> np.ndarray:
    """Restore the row-major 2x2-group grid for an even-sized region."""
    if spec.rows % 2 or spec.cols % 2:
        raise ValueError(f"Region {spec.key!r} has no rectangular coarse grid")
    expected = (spec.rows // 2) * (spec.cols // 2)
    if len(group_ids) != expected:
        raise ValueError(
            f"Region {spec.key!r} has {len(group_ids)} groups, expected {expected}"
        )
    return np.asarray(group_ids, dtype=np.int64).reshape(
        spec.rows // 2, spec.cols // 2
    )


def _palm_attachment_candidates(
    finger_region: str,
    groups_by_hand_region: dict[tuple[int, str], list[int]],
    hand: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return four finger nodes and the four explicit palm target slots."""
    specs = {spec.key: spec for spec in DECO_REGION_SPECS}
    finger_grid = _coarse_grid(
        specs[finger_region], groups_by_hand_region[(hand, finger_region)]
    )
    palm_grid = _coarse_grid(specs["palm"], groups_by_hand_region[(hand, "palm")])

    # The lower four nodes of every finger are its row closest to the palm.
    finger_nodes = finger_grid[0, :]
    palm_nodes = np.asarray(
        [palm_grid[row, col] for row, col in DECO_PALM_ATTACHMENT_TARGETS[finger_region]],
        dtype=np.int64,
    )
    return finger_nodes, palm_nodes


def _minimum_cost_pairs_with_slots(
    source_ids: Sequence[int],
    target_slots: Sequence[int],
    positions: np.ndarray,
    k: int,
) -> list[tuple[int, int]]:
    """Assign source nodes to target slots, retaining intentional duplicates."""
    sources = tuple(int(node) for node in source_ids)
    targets = tuple(int(node) for node in target_slots)
    count = min(max(0, int(k)), len(sources), len(targets))
    if count == 0:
        return []
    if count < len(sources):
        # bridge_k is four in the training graph; retain predictable behavior
        # for diagnostic lower-k builds by using the leading semantic slots.
        sources = sources[:count]
        targets = targets[:count]
    best = min(
        permutations(targets),
        key=lambda assignment: (
            sum(
                float(np.linalg.norm(positions[source] - positions[target]))
                for source, target in zip(sources, assignment)
            ),
            assignment,
        ),
    )
    return list(zip(sources, best))


def _finger_segment_attachment_candidates(
    upper_region: str,
    lower_region: str,
    groups_by_hand_region: dict[tuple[int, str], list[int]],
    hand: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Connect two straight finger pads by their facing 4x1 boundary rows."""
    specs = {spec.key: spec for spec in DECO_REGION_SPECS}
    upper_grid = _coarse_grid(
        specs[upper_region], groups_by_hand_region[(hand, upper_region)]
    )
    lower_grid = _coarse_grid(
        specs[lower_region], groups_by_hand_region[(hand, lower_region)]
    )
    return upper_grid[0, :], lower_grid[-1, :]


def _add_edge(edges: dict[tuple[int, int], bool], left: int, right: int, bridge: bool) -> None:
    if left == right:
        return
    key = (left, right) if left < right else (right, left)
    edges[key] = bool(edges.get(key, False) or bridge)


def _nearest_unique_pairs(
    left_ids: Sequence[int],
    right_ids: Sequence[int],
    positions: np.ndarray,
    k: int,
) -> list[tuple[int, int]]:
    left = np.asarray(left_ids, dtype=np.int64)
    right = np.asarray(right_ids, dtype=np.int64)
    count = min(max(0, int(k)), left.size, right.size)
    if count == 0:
        return []
    distances = np.linalg.norm(
        positions[left, None, :] - positions[None, right, :], axis=-1
    )
    order = np.argsort(distances, axis=None, kind="stable")
    used_left: set[int] = set()
    used_right: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for flat_index in order:
        left_local, right_local = np.unravel_index(int(flat_index), distances.shape)
        if left_local in used_left or right_local in used_right:
            continue
        pairs.append((int(left[left_local]), int(right[right_local])))
        used_left.add(int(left_local))
        used_right.add(int(right_local))
        if len(pairs) == count:
            break
    return pairs


def _closest_members(
    left_group: int,
    right_group: int,
    members: Sequence[np.ndarray],
    positions: np.ndarray,
) -> tuple[int, int]:
    left = members[left_group]
    right = members[right_group]
    distances = np.linalg.norm(
        positions[left, None, :] - positions[None, right, :], axis=-1
    )
    left_local, right_local = np.unravel_index(int(np.argmin(distances)), distances.shape)
    return int(left[left_local]), int(right[right_local])


def _make_graph(
    edge_types: dict[tuple[int, int], bool],
    num_nodes: int,
    metadata: dict,
) -> tuple[WeightedSensorGraph, np.ndarray]:
    ordered = sorted(edge_types)
    edge_index = (
        np.asarray(ordered, dtype=np.int64).T
        if ordered
        else np.zeros((2, 0), dtype=np.int64)
    )
    bridge_mask = np.asarray([edge_types[edge] for edge in ordered], dtype=bool)
    graph = WeightedSensorGraph(
        edge_index=edge_index,
        edge_weight=np.ones(len(ordered), dtype=np.float32),
        num_nodes=num_nodes,
        metadata=metadata,
    )
    return graph, bridge_mask


def _validate_region_specs(specs: Iterable[DecoRegionSpec]) -> None:
    specs = tuple(specs)
    if len(specs) != 17:
        raise ValueError(f"DECO hand must contain 17 regions; got {len(specs)}")
    cursor = 0
    for spec in specs:
        if spec.offset != cursor:
            raise ValueError(
                f"DECO region slices must be contiguous: expected offset {cursor}, "
                f"got {spec.offset} for {spec.key}"
            )
        cursor += spec.size
    if cursor != TAXELS_PER_HAND:
        raise ValueError(f"DECO region sizes sum to {cursor}, expected {TAXELS_PER_HAND}")


def build_deco_geometry(bridge_k: int = 4) -> DecoGeometry:
    """Build the fixed DECO grouping and the graph used for JEPA masking."""
    bridge_k = max(0, int(bridge_k))
    _validate_region_specs(DECO_REGION_SPECS)

    raw_positions = np.zeros((NUM_RAW_TAXELS, 2), dtype=np.float32)
    raw_hand_id = np.zeros(NUM_RAW_TAXELS, dtype=np.int64)
    raw_region_id = np.zeros(NUM_RAW_TAXELS, dtype=np.int64)
    raw_to_hypertaxel = np.full(NUM_RAW_TAXELS, -1, dtype=np.int64)
    members: list[np.ndarray] = []
    hand_ids: list[int] = []
    region_ids: list[int] = []
    group_positions: list[np.ndarray] = []
    groups_by_hand_region: dict[tuple[int, str], list[int]] = {}
    raw_edges: dict[tuple[int, int], bool] = {}
    coarse_edges: dict[tuple[int, int], bool] = {}

    for hand in range(NUM_HANDS):
        for region_index, spec in enumerate(DECO_REGION_SPECS):
            local_positions = _rotate_grid(spec)
            if hand == 1:
                local_positions = local_positions.copy()
                local_positions[:, 0] = 18.0 - local_positions[:, 0]
            raw_start = hand * TAXELS_PER_HAND + spec.offset
            raw_ids = np.arange(raw_start, raw_start + spec.size, dtype=np.int64)
            raw_positions[raw_ids] = local_positions
            raw_hand_id[raw_ids] = hand
            raw_region_id[raw_ids] = hand * len(DECO_REGION_SPECS) + region_index

            local_group_ids = []
            for local_members in _local_groups(spec):
                group_id = len(members)
                group_members = raw_ids[local_members]
                members.append(group_members)
                hand_ids.append(hand)
                region_ids.append(hand * len(DECO_REGION_SPECS) + region_index)
                group_positions.append(raw_positions[group_members].mean(axis=0))
                raw_to_hypertaxel[group_members] = group_id
                local_group_ids.append(group_id)
            groups_by_hand_region[(hand, spec.key)] = local_group_ids

            for row in range(spec.rows):
                for col in range(spec.cols):
                    raw_id = raw_start + row * spec.cols + col
                    for neighbor_row, neighbor_col in ((row + 1, col), (row, col + 1)):
                        if neighbor_row >= spec.rows or neighbor_col >= spec.cols:
                            continue
                        neighbor = raw_start + neighbor_row * spec.cols + neighbor_col
                        _add_edge(raw_edges, raw_id, neighbor, bridge=False)
                        left_group = int(raw_to_hypertaxel[raw_id])
                        right_group = int(raw_to_hypertaxel[neighbor])
                        _add_edge(coarse_edges, left_group, right_group, bridge=False)

    hypertaxel_positions = np.asarray(group_positions, dtype=np.float32)
    for hand in range(NUM_HANDS):
        for left_region, right_region in DECO_REGION_ADJACENCY:
            if (left_region, right_region) in DECO_FINGER_SEGMENT_ATTACHMENTS:
                left_groups, right_groups = _finger_segment_attachment_candidates(
                    left_region, right_region, groups_by_hand_region, hand
                )
                # Preserve semantic column order. A greedy nearest-neighbour
                # assignment can swap columns and draw a crossing even when
                # both endpoint sets are the correct facing 4x1 rows.
                pairs = list(zip(left_groups, right_groups))[:bridge_k]
            elif right_region == "palm" and left_region in {
                *DECO_PALM_ATTACHMENT_TARGETS,
            }:
                left_groups, right_groups = _palm_attachment_candidates(
                    left_region, groups_by_hand_region, hand
                )
                pairs = _minimum_cost_pairs_with_slots(
                    left_groups, right_groups, hypertaxel_positions, bridge_k
                )
            else:
                left_groups = groups_by_hand_region[(hand, left_region)]
                right_groups = groups_by_hand_region[(hand, right_region)]
                pairs = _nearest_unique_pairs(
                    left_groups, right_groups, hypertaxel_positions, bridge_k
                )
            for left_group, right_group in pairs:
                _add_edge(coarse_edges, left_group, right_group, bridge=True)
                left_raw, right_raw = _closest_members(
                    left_group, right_group, members, raw_positions
                )
                _add_edge(raw_edges, left_raw, right_raw, bridge=True)

    if len(members) != NUM_HYPERTAXELS:
        raise RuntimeError(
            f"DECO grouping produced {len(members)} hypertaxels, expected {NUM_HYPERTAXELS}"
        )
    if np.any(raw_to_hypertaxel < 0):
        raise RuntimeError("DECO grouping left raw taxels uncovered")

    member_indices = np.full((NUM_HYPERTAXELS, 5), -1, dtype=np.int64)
    member_mask = np.zeros((NUM_HYPERTAXELS, 5), dtype=bool)
    for group_id, group_members in enumerate(members):
        member_indices[group_id, : group_members.size] = group_members
        member_mask[group_id, : group_members.size] = True

    common_metadata = {
        "graph_type": "deco_physical",
        "bridge_k": bridge_k,
        "region_adjacency": DECO_REGION_ADJACENCY,
        "finger_segment_attachments": DECO_FINGER_SEGMENT_ATTACHMENTS,
        "palm_attachment_targets": DECO_PALM_ATTACHMENT_TARGETS,
        "edge_weight": "unit_topological_distance",
    }
    raw_graph, raw_bridge_mask = _make_graph(
        raw_edges,
        NUM_RAW_TAXELS,
        {**common_metadata, "level": "raw_visualization_only"},
    )
    coarse_graph, coarse_bridge_mask = _make_graph(
        coarse_edges,
        NUM_HYPERTAXELS,
        {**common_metadata, "level": "hypertaxel_training_graph"},
    )
    return DecoGeometry(
        member_indices=member_indices,
        member_mask=member_mask,
        group_size=member_mask.sum(axis=1).astype(np.int64),
        hand_id=np.asarray(hand_ids, dtype=np.int64),
        region_id=np.asarray(region_ids, dtype=np.int64),
        raw_to_hypertaxel=raw_to_hypertaxel,
        raw_hand_id=raw_hand_id,
        raw_region_id=raw_region_id,
        raw_positions=raw_positions,
        hypertaxel_positions=hypertaxel_positions,
        raw_graph=raw_graph,
        coarse_graph=coarse_graph,
        raw_bridge_mask=raw_bridge_mask,
        coarse_bridge_mask=coarse_bridge_mask,
        bridge_k=bridge_k,
    )
