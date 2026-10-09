"""CPU checks for the mechanisms absent from the public baseline commit."""
import json
from types import SimpleNamespace

import hydra
import pytest
import torch

from repro.xela_20261009.run import BUNDLE, environment, manifest, resolve_config, instantiate_for_validation


def config(case):
    s = next(s for s in manifest()['stages'] if s['case'] == case and s['stage'] == 'pretrain' and s['pretrain_seed'] == 42)
    env = environment(SimpleNamespace(data_root='unused-data', output_root='unused-output', cache_root=None))
    return resolve_config(s, env)


def samples():
    n = 368
    edge = torch.stack((torch.arange(n - 1), torch.arange(1, n)))
    groups = torch.arange(n) * 18 // n
    return [{'sensor': torch.randn(10, n, 3), 'graph': {
        'edge_index': edge, 'edge_attr': torch.ones(n - 1, 1),
        'edge_count': torch.tensor(n - 1), 'node_group_id': groups,
    }} for _ in range(2)]


@pytest.mark.parametrize('case', ['0', 'a2', 'a4', 'a7', 'a8', 'bal-both', 'bal-soft', 'bal-mix'])
def test_frozen_masks_obey_overlap_and_context_exclusion(case):
    torch.manual_seed(42)
    cfg = config(case)
    collator = hydra.utils.instantiate(cfg.data.graph_mask_collator)
    batch = collator(samples())
    targets = list(batch['target_masks'])
    for b in range(2):
        for i, left in enumerate(targets):
            assert not torch.isin(batch['context_masks'][0, b], left[b]).any()
            for right in targets[i + 1:]:
                assert int(torch.isin(left[b], right[b]).sum()) <= .2 * min(left.shape[-1], right.shape[-1]) + 1e-8


def test_deterministic_teacher_and_mse_default():
    model = instantiate_for_validation(config('0').algorithm)
    model.train()
    assert model.context_encoder.training
    assert not model.target_encoder.training
    assert isinstance(model.jepa_loss, torch.nn.MSELoss)
    assert model.target_loss_size_power == 0
    assert not any(p.requires_grad for p in model.target_encoder.parameters())


def test_lr_variants_reuse_matching_pretrain_seeds():
    stages = manifest()['stages']
    for s in stages:
        if '-lr' in s['case']:
            assert s['stage'] != 'pretrain'
            owner = s['case'].split('-lr')[0]
            assert s['depends_on'] == f"{owner}/p{s['pretrain_seed']}/pretrain"


def test_scope_matches_workbook_and_preserves_all_seeds():
    m = manifest()
    assert len(m['cases']) == 38
    assert len(m['stages']) == 1128
    assert sum(s['stage'] == 'pretrain' for s in m['stages']) == 102
    assert {s['seed'] for s in m['stages']} == {42, 17, 3407}


def test_recorded_episode_order_overrides_filesystem_order(tmp_path, monkeypatch):
    from tactile_ssl.data.xela.reproduction_order import episode_directories
    folder = tmp_path / 'train'
    folder.mkdir()
    for name in ['b', 'a', 'c']:
        (folder / name).mkdir()
    record = tmp_path / 'order.json'
    record.write_text(json.dumps({'directories': {'train': ['c', 'b', 'a']}}))
    monkeypatch.setenv('XELA_DATA_ROOT', str(tmp_path))
    monkeypatch.setenv('XELA_REPRO_DIRECTORY_ORDER', str(record))
    assert [p.name for p in episode_directories(folder)] == ['c', 'b', 'a']
    (folder / 'unexpected').mkdir()
    with pytest.raises(ValueError, match='Episode directories differ'):
        episode_directories(folder)
