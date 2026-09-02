from __future__ import annotations

from collections.abc import Mapping, Sequence

from tactile_ssl.utils.sock_jepa_masking import SockJEPAGraphMaskCollator


class DecoJEPAGraphMaskCollator(SockJEPAGraphMaskCollator):
    """One global context plus local/random targets on each DECO hand."""

    def __init__(
        self,
        context_mask_scale: Sequence[float],
        target_mask_scale: Sequence[float],
        min_context_keep_tokens: int = 32,
        min_context_keep_ratio: float = 0.15,
        target_specs: Sequence[Mapping] = (
            {"hand": 0, "strategy": "connected_region"},
            {"hand": 0, "strategy": "random"},
            {"hand": 1, "strategy": "connected_region"},
            {"hand": 1, "strategy": "random"},
        ),
        growth: str = "dijkstra",
        max_resample_attempts: int = 32,
    ) -> None:
        translated = []
        for spec in target_specs:
            item = dict(spec)
            if "hand" not in item:
                raise ValueError("Every DECO target spec must contain hand=0 or hand=1")
            item["foot"] = item.pop("hand")
            translated.append(item)
        super().__init__(
            context_mask_scale=context_mask_scale,
            target_mask_scale=target_mask_scale,
            min_context_keep_tokens=min_context_keep_tokens,
            min_context_keep_ratio=min_context_keep_ratio,
            target_specs=translated,
            growth=growth,
            max_resample_attempts=max_resample_attempts,
        )

