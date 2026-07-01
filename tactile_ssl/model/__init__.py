from .custom_scheduler import WarmupCosineScheduler  # noqa: F401
from .multimodal_transformer import (
    MultimodalTransformer,
    MultimodalDecoder,
)
from .signal_transformer import SignalTransformer
from .xela_transformer import *  # noqa: F401
from .xela_linear import XelaLinear
from .xela_gat import XelaGAT


VIT_EMBED_DIMS = {
    "vit_tiny": 192,
    "vit_small": 384,
    "vit_base": 768,
    "vit_large": 1024,
    "vit_giant2": 1536,
}
