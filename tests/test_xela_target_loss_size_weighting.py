"""Exercise the actual JEPA forward with deterministic encoder/predictor doubles."""
import ast
from pathlib import Path
import einops
import torch
from torch import nn
from typing import Dict, List, Union
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
source = ast.parse((ROOT / 'tactile_ssl/algorithm/xela_jepa.py').read_text())
cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'XelaJEPAModule')
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('forward', '_reduce_target_losses')]
scope = dict(torch=torch, einops=einops, Dict=Dict, List=List, Union=Union)
exec(compile(ast.Module(body=methods, type_ignores=[]), 'xela_jepa.py', 'exec'), scope)

class Encoder(nn.Module):
    def forward_features(self, xs, masks=None, **kwargs):
        b, _, n, _ = xs.shape
        if masks is not None:
            b *= masks.shape[0]
            n = masks.shape[-1]
        return {'x_norm_patchtokens': torch.zeros(b, n, 1)}
    def get_position_embedding(self, device):
        return None
    def apply_tubelet_masks(self, tokens, masks):
        return torch.cat([tokens.gather(2, m[:, None, :, None]) for m in masks])

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.values = nn.Parameter(torch.arange(1., 33.))
    def forward(self, tokens, context_masks, masks, **kwargs):
        return [self.values[m[:, 0]][:, None, None, None].expand(-1, 1, m.shape[-1], 1) for m in masks]

class Harness(nn.Module):
    forward = scope['forward']
    _reduce_target_losses = scope['_reduce_target_losses']
    def __init__(self, power):
        super().__init__()
        self.context_encoder = Encoder()
        self.target_encoder = Encoder()
        self.predictor = Predictor()
        self.jepa_loss = nn.MSELoss(reduction='none')
        self.target_loss_size_power = power

xs = torch.zeros(2, 1, 32, 1)
context = torch.tensor([[[30, 31], [30, 31]]])
# Interleaved duplicate sizes catches weights aligned to original instead of grouped order.
masks = [torch.arange(start, start + size).repeat(2, 1) for start, size in [(0, 4), (5, 9), (1, 4), (10, 16)]]
for power in (0., .5):
    model = Harness(power)
    loss = model(xs, context, masks)
    weights = torch.tensor([4., 9., 4., 16.]).pow(power)
    weights /= weights.sum()
    expected = (torch.tensor([1., 6., 2., 11.]).square() * weights).sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(model.predictor.values.grad[[0, 5, 1, 10]], 2 * torch.tensor([1., 6., 2., 11.]) * weights)
    same = torch.stack([masks[0], masks[2]])
    torch.testing.assert_close(model(xs, context, same), model(xs, context, list(same)))
    if power == 0:
        losses = torch.randn(4)
        assert torch.equal(model._reduce_target_losses(losses, [4, 4, 9, 16]), losses.mean())
from omegaconf import OmegaConf
baseline = OmegaConf.load(ROOT / 'repro/xela_20261009/configs/4b/p42/pretrain.yaml')
weighted = OmegaConf.load(ROOT / 'repro/xela_20261009/configs/4b-sqrt/p42/pretrain.yaml')
assert baseline.algorithm.get('target_loss_size_power', 0) == 0
assert weighted.algorithm.target_loss_size_power == .5
old = OmegaConf.to_container(baseline.algorithm)
new = OmegaConf.to_container(weighted.algorithm)
del new['target_loss_size_power']
assert old == new
print('PASS: forward, grouped weights, gradients, equal-size parity, legacy exact reduction, config-only ablation')
