# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



from typing import List
import torch
import torch.nn as nn
from tactile_ssl.downstream_task.attentive_pooler import AttentivePooler


class Classifier(nn.Module):
    """
    General module for multi-class classification tasks.
    
    This module supports two pooling strategies:
    1. Attentive pooling: Uses a transformer-based attention mechanism to extract
       relevant features from the input sequence
    2. Mean pooling: Computes the average of input embeddings
    
    Designed to evaluate the quality of learned representations in classification tasks.
    """

    def __init__(
        self,
        input_embed_dim: int = 768,
        classes: List[str] = None,
        class_weights: List[float] = None,
        with_attentive_pooling: bool = False,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
        num_queries=1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.classes = classes
        self.num_classes = len(classes)
        self.class_weights = torch.Tensor(class_weights).float() if class_weights is not None else None
        self.with_attentive_pooling = with_attentive_pooling
        self.num_queries = num_queries if with_attentive_pooling else 1

        if with_attentive_pooling:
            self.pooler = AttentivePooler(
                num_queries=num_queries,
                embed_dim=input_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                depth=depth,
                norm_layer=norm_layer,
                init_std=init_std,
                qkv_bias=qkv_bias,
                complete_block=complete_block,
            )

        self.probe = nn.Sequential(nn.Linear(input_embed_dim * self.num_queries, self.num_classes))

    def forward(self, x):
        if self.with_attentive_pooling:
            x = self.pooler(x).squeeze(1)
            x = x.flatten(start_dim=1)
        else:
            x = x.mean(dim=1)
        x = self.probe(x)
        return x


class LinearClassifier(nn.Module):
    """
    Simple linear classifier for multi-class classification tasks.
    
    This model applies a single linear layer to input features to predict class probabilities.
    Used primarily for evaluating the quality of learned representations in downstream
    classification tasks without introducing complex architecture dependencies.
    """
    def __init__(
        self, input_embed_dim: int = 768, classes: List[str] = None, class_weights: List[float] = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.classes = classes
        self.num_classes = len(classes)
        self.class_weights = torch.Tensor(class_weights).float() if class_weights is not None else None
        self.probe = nn.Sequential(nn.Linear(input_embed_dim, self.num_classes))

    def forward(self, x):
        x = self.probe(x)
        return x


class ProjectedLinearClassifier(LinearClassifier):
    """Linear classifier after a trainable bottleneck for wide SSL features."""

    def __init__(self, input_embed_dim: int, projection_dim: int, *args, **kwargs):
        super().__init__(input_embed_dim=projection_dim, *args, **kwargs)
        self.input_norm = nn.LayerNorm(int(input_embed_dim))
        self.input_projection = nn.Linear(int(input_embed_dim), int(projection_dim))

    def forward(self, x):
        return super().forward(self.input_projection(self.input_norm(x)))
