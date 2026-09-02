from .custom_scheduler import WarmupCosineScheduler  # noqa: F401
from .multimodal_transformer import (
    MultimodalTransformer,
    MultimodalDecoder,
)
from .signal_transformer import SignalTransformer
from .xela_transformer import *  # noqa: F401
from .xela_linear import XelaLinear
from .xela_gat import XelaGAT
from .xela_spatial_gnn import XelaSpatialGNNTransformer
from .xela_spatial_wl_dino import XelaSpatialWLDINOTransformer
from .sock_transformer import SockTransformer, sock_tiny, sock_small, sock_base
from .deco_transformer import DecoHypertaxelTokenizer, DecoTransformer, deco_tiny, deco_small, deco_base


VIT_EMBED_DIMS = {
    "vit_tiny": 192,
    "vit_small": 384,
    "vit_base": 768,
    "vit_large": 1024,
    "vit_giant2": 1536,
}
