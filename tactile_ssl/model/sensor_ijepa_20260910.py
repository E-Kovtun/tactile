"""Isolated fixed-atlas encoders; existing graph encoders are unchanged."""
import torch
from tactile_ssl.model.sock_transformer import SockTransformer
from tactile_ssl.model.deco_transformer import DecoTransformer
from tactile_ssl.utils.sensor_ijepa_20260910 import atlas, fixed_embedding


class FixedAtlas:
    def init_pos_embed(self, pos_embed_fn):
        if pos_embed_fn != 'sensor_2d_sincos':
            raise ValueError('I-JEPA atlas encoder requires fixed 2D sin/cos')
        self.pos_embed_fn = pos_embed_fn
        self.register_buffer('pos_embed', fixed_embedding(atlas(self.atlas_sensor, self.atlas_path), self.embed_dim), persistent=True)

    def get_position_embedding(self, device):
        return self.pos_embed.to(device=device, dtype=torch.float32)


class SockIJEPA(FixedAtlas, SockTransformer):
    def __init__(self, sensor_map_path, **kwargs):
        self.atlas_sensor, self.atlas_path = 'sock', sensor_map_path
        kwargs.update(use_sensor_id_embedding=False, use_foot_embedding=False, pos_embed_fn='sensor_2d_sincos')
        super().__init__(**kwargs)


class DecoIJEPA(FixedAtlas, DecoTransformer):
    def __init__(self, **kwargs):
        self.atlas_sensor, self.atlas_path = 'deco', None
        kwargs['pos_embed_fn'] = 'sensor_2d_sincos'
        super().__init__(**kwargs)


def sock_ijepa_tiny(**kwargs):
    return SockIJEPA(embed_dim=192, depth=8, num_heads=3, mlp_ratio=4, **kwargs)


def deco_ijepa_tiny(**kwargs):
    return DecoIJEPA(embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, **kwargs)
