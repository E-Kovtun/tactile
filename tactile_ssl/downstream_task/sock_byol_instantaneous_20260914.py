"""Trainable head projection for frozen single-frame AlexNet features."""
from torch import nn
from tactile_ssl.downstream_task.sock import SockTemporalClassifier, SockTemporalPoseRegressor


class ProjectedSockTemporalClassifier(SockTemporalClassifier):
    def __init__(self, input_embed_dim, encoder_embed_dim=9216, **kwargs):
        super().__init__(input_embed_dim=input_embed_dim, **kwargs)
        self.input_projection = nn.Linear(encoder_embed_dim, input_embed_dim)

    def forward(self, patch_tokens):
        return super().forward(self.input_projection(patch_tokens))


class ProjectedSockTemporalPoseRegressor(SockTemporalPoseRegressor):
    def __init__(self, input_embed_dim, encoder_embed_dim=9216, **kwargs):
        super().__init__(input_embed_dim=input_embed_dim, **kwargs)
        self.input_projection = nn.Linear(encoder_embed_dim, input_embed_dim)

    def forward(self, patch_tokens):
        return super().forward(self.input_projection(patch_tokens))
