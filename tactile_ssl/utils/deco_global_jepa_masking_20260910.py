"""DECO graph masking that permits explicitly global target masks."""
from collections.abc import Mapping, Sequence

from tactile_ssl.utils.sock_jepa_masking import SockJEPAGraphMaskCollator


class DecoGlobalJEPAGraphMaskCollator(SockJEPAGraphMaskCollator):
    """Translate DECO hand specs while retaining hand=None as a global mask."""

    def __init__(
        self,
        context_mask_scale: Sequence[float],
        target_mask_scale: Sequence[float],
        min_context_keep_tokens: int = 32,
        min_context_keep_ratio: float = 0.15,
        target_specs: Sequence[Mapping] = (
            {"hand": 0, "strategy": "connected_region"},
            {"hand": 1, "strategy": "connected_region"},
            {"hand": None, "strategy": "random"},
            {"hand": None, "strategy": "random"},
        ),
        growth: str = "dijkstra",
        max_resample_attempts: int = 32,
    ) -> None:
        translated = []
        for spec in target_specs:
            item = dict(spec)
            if "hand" not in item or item["hand"] not in (0, 1, None):
                raise ValueError("Every DECO target must use hand=0, hand=1, or hand=null")
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
