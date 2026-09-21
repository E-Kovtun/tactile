#!/usr/bin/env python3
"""Prepare the DECO caches used by the paper configurations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_tactile_cache(root: Path, manifest: Path, output: Path, workers: int = 4):
    """Build the existing tactile cache format directly, without decoding images."""
    import numpy as np
    from torch.utils.data import DataLoader
    from tactile_ssl.data.deco import create_deco_ssl_datasets
    from tactile_ssl.data.deco_ssl_cache import CACHE_FORMAT_VERSION, CachedDecoSSLDataset

    signature = hashlib.sha256(manifest.read_bytes()).hexdigest()
    completed = output / 'manifest.json'
    if completed.exists():
        saved = json.loads(completed.read_text())
        if saved.get('source_manifest_sha256') != signature:
            raise ValueError('Tactile cache belongs to a different manifest; choose another cache directory.')
        for split in ('train', 'val', 'test'):
            CachedDecoSSLDataset(output, split)
        print('Tactile cache is ready:', output, flush=True)
        return
    train, val = create_deco_ssl_datasets(
        root=str(root), manifest_path=str(manifest), split_seed=42,
        val_episode_ratio=.15, test_episode_ratio=.15, window_size=3, stride=3,
        normalization='deco_max', left_scale=3486., right_scale=4050., include_graph=False,
    )
    result = dict(format_version=CACHE_FORMAT_VERSION, source_kind='episode_archives',
                  source_manifest_sha256=signature, split_seed=42, window_size=3, stride=3, splits={})
    for split, dataset in [('train', train), ('val', val), ('test', train.test_dataset)]:
        folder = output / split
        folder.mkdir(parents=True, exist_ok=True)
        shapes = {'sensor': (len(dataset), 3, 528, 5), 'sample_id': (len(dataset),), 'group_id': (len(dataset),)}
        arrays = {key: np.lib.format.open_memmap(
            folder / (key + '.npy.partial'), mode='w+', shape=shape,
            dtype=np.float32 if key == 'sensor' else np.int64) for key, shape in shapes.items()}
        offset = 0
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=workers)
        for batch in loader:
            stop = offset + len(batch['sensor'])
            for key, array in arrays.items():
                array[offset:stop] = batch[key].numpy()
            offset = stop
        assert offset == len(dataset)
        for array in arrays.values():
            array.flush()
        arrays.clear()
        for key in shapes:
            (folder / (key + '.npy.partial')).replace(folder / (key + '.npy'))
        result['splits'][split] = {'length': offset, 'arrays': {key: f'{split}/{key}.npy' for key in shapes}}
        dataset.store.close()
        print(f'{split}: {offset} tactile windows', flush=True)
    partial = output / 'manifest.json.partial'
    partial.write_text(json.dumps(result, indent=2) + '\n')
    partial.replace(completed)


def prepare(root: Path, cache: Path, stage: str, device: str, workers: int):
    from tactile_ssl.data.deco import save_deco_manifest
    root, cache = root.expanduser().resolve(), cache.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / 'episodes.json'
    if not manifest.exists():
        save_deco_manifest(root, manifest)
    else:
        saved_root = json.loads(manifest.read_text()).get('root')
        if saved_root != str(root):
            raise ValueError('This cache directory belongs to another dataset root; choose a new --cache.')
    if stage in ('all', 'pretrain'):
        build_tactile_cache(root, manifest, cache / 'tactile', workers)
    if stage in ('all', 'policy'):
        signature = hashlib.sha256(manifest.read_bytes()).hexdigest()
        plan = {'source_manifest_sha256': signature, 'split_seed': 42, 'window_size': 3, 'stride': 3}
        marker = cache / 'policy_preparation.json'
        if marker.exists() and json.loads(marker.read_text()) != plan:
            raise ValueError('Policy cache settings changed; choose a new --cache.')
        marker.write_text(json.dumps(plan, indent=2) + '\n')
        shared = ['--source-root', str(root), '--source-manifest', str(manifest),
                  '--policy-stats', str(cache / 'policy_stats.json'), '--output-root', str(cache / 'policy'),
                  '--num-workers', str(workers)]
        vision = ['--source-root', str(cache / 'policy'), '--source-manifest', str(cache / 'policy/manifest.json'),
                  '--output-root', str(cache / 'vision'), '--device', device]
        for module, args in [('scripts.paper.policy_cache', shared),
                             ('scripts.paper.vision_cache', vision)]:
            for mode in ('build', 'finalize'):
                subprocess.run([sys.executable, '-m', module, mode, *args], cwd=ROOT, check=True)
    print('Ready:', cache, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['deco'])
    parser.add_argument('--root', type=Path, default=Path('datasets/deco50/task4'))
    parser.add_argument('--cache', type=Path, default=Path('cache/deco'))
    parser.add_argument('--stage', choices=['manifest', 'pretrain', 'policy', 'all'], default='all')
    parser.add_argument('--device', default='cuda:0', help='Device for frozen ResNet18 features; cpu is supported.')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 0:
        parser.error('--workers must be nonnegative')
    prepare(args.root, args.cache, args.stage, args.device, args.workers)


if __name__ == '__main__':
    main()
