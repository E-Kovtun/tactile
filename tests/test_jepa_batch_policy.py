from copy import deepcopy

import pytest
import torch

from tactile_ssl.utils.jepa_batch_policy import JEPABatchPolicyMaskCollator
from tactile_ssl.utils.jepa_masking import JEPAGraphMaskCollator


def make_collator():
    masking = {'mode': 'multiblock_graph', 'target': {'groups': [{'count': 4, 'tag': 'original'}]}}
    policies = [{'name': 'a2', 'weight': .5, 'groups': [{'count': 4, 'tag': 'a2'}]},
                {'name': 'a4', 'weight': .5, 'groups': [{'count': 4, 'tag': 'a4'}]}]
    return JEPABatchPolicyMaskCollator([.5, .9], [.15, .2], 1, 4, masking, policies)


def test_worker_rng_reproduces_whole_batch_choices_and_restores_base(monkeypatch):
    def parent_call(self, samples):
        return {'policy_for_every_sample': [self.masking['target']['groups'][0]['tag']] * len(samples)}
    monkeypatch.setattr(JEPAGraphMaskCollator, '__call__', parent_call)
    torch.manual_seed(42)
    left, right = make_collator(), make_collator()
    original = deepcopy(left.masking)
    counts = [0, 0]
    for _ in range(1024):
        a, b = left([{}, {}, {}]), right([{}, {}, {}])
        assert a == b
        index = int(a['mask_policy_id'])
        assert a['policy_for_every_sample'] == [left.policy_names[index]] * 3
        counts[index] += 1
    assert all(450 < count < 575 for count in counts)
    assert left.masking == original


def test_policy_does_not_leak_after_sampling_failure(monkeypatch):
    def fail(self, samples):
        raise RuntimeError('invalid graph')
    monkeypatch.setattr(JEPAGraphMaskCollator, '__call__', fail)
    collator = make_collator()
    original = collator.masking
    with pytest.raises(RuntimeError, match='invalid graph'):
        collator([{}])
    assert collator.masking is original
