"""Opt-in training ablations must preserve the public JEPA defaults."""
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from tactile_ssl.algorithm.xela_jepa import TargetNormalizedMSE
from tactile_ssl.utils.deco_jepa_masking import (
    DecoJEPAGraphMaskCollator,
    uniform_context_subset,
)


def config(name):
    with initialize_config_dir(version_base='1.3', config_dir=str(Path(__file__).resolve().parents[1] / 'config')):
        return compose(config_name='deco/pretrain/' + name)


def test_public_defaults_remain_unchanged():
    cfg = config('jepa')
    assert cfg.algorithm._target_ == 'tactile_ssl.algorithm.XelaJEPAModule'
    assert cfg.algorithm.get('prediction_loss', 'mse') == 'mse'
    assert cfg.algorithm.encoder.tokenizer_mode == 'direct'
    assert cfg.algorithm.encoder.drop_path_rate == .3
    assert cfg.algorithm.wd_scheduler_cfg.final_weight_decay == .4
    assert list(cfg.algorithm.context_mask_scale) == [.5, .9]
    assert list(cfg.algorithm.target_mask_scale) == [.1, .18]
    assert cfg.data.train_dataloader.shuffle is False


def test_winning_config_and_teacher_mode():
    cfg = config('jepa_wd01')
    assert cfg.algorithm.get('prediction_loss', 'mse') == 'mse'
    assert cfg.algorithm.target_layer_norm is False
    assert cfg.algorithm.wd_scheduler_cfg.final_weight_decay == .1
    assert cfg.algorithm.encoder.drop_path_rate == .3
    assert list(cfg.algorithm.moving_average_decay) == [.994, 1.]
    assert list(cfg.algorithm.context_mask_scale) == [.85, 1.]
    assert list(cfg.algorithm.target_mask_scale) == [.3, .4]
    cfg.algorithm.lr_scheduler_cfg.steps_per_epoch = 10
    cfg.algorithm.lr_scheduler_cfg.T_max = 1500
    cfg.algorithm.wd_scheduler_cfg.T_max = 1500
    model = instantiate(cfg.algorithm)
    model.train()
    assert model.context_encoder.training
    assert not model.target_encoder.training
    assert not any(p.requires_grad for p in model.target_encoder.parameters())
    x = torch.randn(3, 7, 192, requires_grad=True)
    y = torch.randn_like(x)
    actual = model.jepa_loss(x, y).mean()
    expected = torch.nn.functional.mse_loss(x, y)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual, x, retain_graph=True)[0],
                               torch.autograd.grad(expected, x)[0], rtol=0, atol=0)


def test_normalized_targets_are_detached():
    x = torch.randn(2, 7, 192, requires_grad=True)
    y = torch.randn_like(x, requires_grad=True)
    actual = TargetNormalizedMSE()(x, y)
    expected = (x - torch.nn.functional.layer_norm(y.detach(), (192,))) ** 2
    torch.testing.assert_close(actual, expected)
    actual.mean().backward()
    assert x.grad is not None and y.grad is None


def test_pairwise_overlap_cap_and_context_exclusion():
    collator = instantiate(config('jepa_wd01').data.graph_mask_collator)
    assert isinstance(collator, DecoJEPAGraphMaskCollator)
    # Two independent hand graphs; connected targets cannot cross hands.
    nodes = torch.arange(264)
    ring = torch.stack([nodes, nodes.roll(-1)])
    graph = {'edge_index': torch.cat([ring, ring + 264], dim=1),
             'edge_attr': torch.ones(528),
             'node_group_id': torch.arange(528) // 264}
    samples = [{'sensor': torch.zeros(3, 528, 5), 'graph': graph} for _ in range(8)]
    torch.manual_seed(17)
    for _ in range(8):
        batch = collator(samples)
        targets = batch['target_masks']
        for b in range(8):
            context = batch['context_masks'][0, b]
            for i in range(4):
                target = targets[i][b]
                assert not torch.isin(context, target).any()
                for j in range(i):
                    previous = targets[j][b]
                    assert torch.isin(target, previous).sum() <= .2 * min(len(target), len(previous))


def test_crop_samples_both_ends_instead_of_a_sorted_prefix():
    generator = torch.Generator().manual_seed(42)
    counts = torch.zeros(100)
    for _ in range(1000):
        kept = uniform_context_subset(torch.arange(100), 20, generator)
        counts[kept] += 1
    assert counts.min() > 140 and counts.max() < 260


def make_model(**options):
    cfg = config('jepa')
    cfg.algorithm.lr_scheduler_cfg.steps_per_epoch = 10
    cfg.algorithm.lr_scheduler_cfg.T_max = 1500
    cfg.algorithm.wd_scheduler_cfg.T_max = 1500
    return instantiate(cfg.algorithm, **options)


def test_legacy_teacher_follows_training_mode():
    model = make_model()
    assert model.prediction_loss == "mse"
    assert not model.target_layer_norm
    model.eval()
    assert not model.target_encoder.training
    model.train()
    assert model.target_encoder.training and model.context_encoder.training


def test_smooth_l1_and_invalid_loss_options():
    model = make_model(prediction_loss="smooth_l1", smooth_l1_beta=1.)
    x = torch.tensor([0., 3.], requires_grad=True)
    y = torch.tensor([.5, 0.])
    actual = model.jepa_loss(x, y).mean()
    expected = torch.nn.functional.smooth_l1_loss(x, y, beta=1.)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for options in [dict(prediction_loss="invalid"),
                    dict(prediction_loss="smooth_l1", smooth_l1_beta=0),
                    dict(prediction_loss="smooth_l1", target_layer_norm=True)]:
        with pytest.raises(Exception, match="Unsupported prediction_loss|must be positive|requires MSE"):
            make_model(**options)


def graph_samples():
    nodes = torch.arange(264)
    ring = torch.stack([nodes, nodes.roll(-1)])
    graph = {"edge_index": torch.cat([ring, ring + 264], dim=1),
             "edge_attr": torch.ones(528),
             "node_group_id": torch.arange(528) // 264}
    return [{"sensor": torch.zeros(3, 528, 5), "graph": graph} for _ in range(4)]


@pytest.mark.parametrize("seed", [42, 17, 3407])
def test_disabled_mask_options_preserve_legacy_masks_and_rng(seed):
    from tactile_ssl.utils.sock_jepa_masking import SockJEPAGraphMaskCollator
    legacy = SockJEPAGraphMaskCollator([.85, 1.], [.3, .4])
    current = DecoJEPAGraphMaskCollator([.85, 1.], [.3, .4])
    torch.manual_seed(seed)
    for _ in range(3):
        expected = legacy(graph_samples())
        actual = current(graph_samples())
        torch.testing.assert_close(actual["context_masks"], expected["context_masks"], rtol=0, atol=0)
        for a, b in zip(actual["target_masks"], expected["target_masks"]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert torch.equal(current._generator.get_state(), legacy._generator.get_state())


@pytest.mark.parametrize("random_crop,cap", [(True, None), (False, .2), (True, .2)])
def test_mask_options_can_be_enabled_independently(random_crop, cap):
    collator = DecoJEPAGraphMaskCollator([.85, 1.], [.3, .4],
        random_context_crop=random_crop, max_target_overlap=cap)
    batch = collator(graph_samples())
    for b in range(4):
        for target in batch["target_masks"]:
            assert not torch.isin(batch["context_masks"][0, b], target[b]).any()


def test_cache_graph_is_opt_in_and_split_is_fixed(tmp_path):
    import json
    import numpy as np
    from tactile_ssl.data.deco_ssl_cache import create_cached_deco_ssl_datasets
    arrays = {"sensor": np.zeros((1, 3, 528, 5), dtype=np.float32),
              "sample_id": np.array([5], dtype=np.int64),
              "group_id": np.array([2], dtype=np.int64)}
    for key, value in arrays.items():
        np.save(tmp_path / (key + ".npy"), value)
    manifest = {"format_version": 1, "split_seed": 42, "window_size": 3, "stride": 3,
                "splits": {split: {"length": 1, "arrays": {k: k + ".npy" for k in arrays}}
                           for split in ("train", "val", "test")}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    plain, _ = create_cached_deco_ssl_datasets(str(tmp_path))
    graph, _ = create_cached_deco_ssl_datasets(str(tmp_path), include_graph=True)
    assert "graph" not in plain[0] and "graph" in graph[0]
    for key in arrays:
        assert torch.equal(plain[0][key], graph[0][key])
    with pytest.raises(ValueError, match="split seed 42"):
        create_cached_deco_ssl_datasets(str(tmp_path), split_seed=17)
