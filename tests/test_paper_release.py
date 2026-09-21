"""Release configuration and preprocessing contracts; no GPU or full dataset needed."""
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_data import build_tactile_cache
from tactile_ssl.data.deco import create_deco_ssl_datasets, save_deco_manifest
from tactile_ssl.data.deco_ssl_cache import CachedDecoSSLDataset

ROOT = Path(__file__).resolve().parents[1]


def test_public_configurations_are_portable():
    catalog = json.loads((ROOT/'config/paper/catalog.json').read_text())
    OmegaConf.register_new_resolver('int_multiply',lambda a,b:int(a*b),replace=True)
    OmegaConf.register_new_resolver('join',lambda s,x:s.join(map(str,x)),replace=True)
    with initialize_config_dir(version_base='1.3',config_dir=str(ROOT/'config/paper')):
        for name, metadata in catalog.items():
            assert metadata["entrypoint"] in ("train.py", "train_task_force.py", "train_task_object.py", "train_task_pose_estimation.py", "train_task_socks_action.py", "train_task_socks_pose.py", "train_task_deco_policy.py")
            cfg=compose(config_name=name,overrides=['paths.data_root=/portable/data','paths.cache_root=/portable/cache','checkpoint=/portable/encoder.ckpt'])
            cfg.paths.output_dir='/portable/output'
            cfg.paths.work_dir='/portable/repo'
            resolved=OmegaConf.to_container(cfg,resolve=True)
            if name.startswith('deco/policy/'):
                assert cfg.trainer.max_epochs == 150 and cfg.trainer.max_steps is None
                assert cfg.trainer.early_stopping_patience > cfg.trainer.max_epochs
                assert not any(key.startswith('continuation_') for key in cfg)
                assert cfg.ckpt_path is None
            serialized=json.dumps(resolved)
            assert '/workspace-SR004' not in serialized,name
            assert '/portable/data' in serialized,name
        for name in ('socks/action/pretrain/jepa','socks/pose/pretrain/jepa'):
            cfg=compose(config_name=name)
            assert cfg.trainer.max_epochs==500 and cfg.trainer.max_steps is None
            assert cfg.algorithm.lr_scheduler_cfg.warmup_epochs==30
        cfg=compose(config_name='xela/pretrain/jepa')
        assert cfg.algorithm.masking.context.strategy=='random'
        assert cfg.algorithm.num_target_masks==4


def test_tactile_cache_matches_raw_windows_without_images(tmp_path):
    root=tmp_path/'raw'
    for episode in range(3):
        folder=root/'data-t4-1';folder.mkdir(parents=True,exist_ok=True)
        prefix=f'episode_{episode:04d}'
        # Deliberately omit images and action tables: SSL must need neither.
        with tarfile.open(folder/(prefix+'.tar'),'w') as archive:
            for frame in range(7):
                for side in ('left','right'):
                    stream=io.BytesIO();np.save(stream,np.arange(1062,dtype=np.float64)+frame+episode)
                    payload=stream.getvalue()
                    item=tarfile.TarInfo(f'{prefix}/tactiles/{frame:06d}_{side}_ee_tactile.npy')
                    item.size=len(payload);archive.addfile(item,io.BytesIO(payload))
    manifest=save_deco_manifest(root)
    cache=tmp_path/'cache'
    build_tactile_cache(root,manifest,cache,workers=0)
    train,val=create_deco_ssl_datasets(root=str(root),manifest_path=str(manifest),split_seed=42,
        val_episode_ratio=.15,test_episode_ratio=.15,window_size=3,stride=3,include_graph=False)
    groups=[]
    for split,raw in [('train',train),('val',val),('test',train.test_dataset)]:
        cached=CachedDecoSSLDataset(cache,split)
        assert len(cached)==len(raw)==2
        for i in range(len(raw)):
            for key in ('sensor','sample_id','group_id'):
                assert torch.equal(cached[i][key],raw[i][key])
        groups.append(set(cached.group_ids.tolist()))
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    before=(cache/'train/sensor.npy').stat().st_mtime_ns
    build_tactile_cache(root,manifest,cache,workers=0)
    assert (cache/'train/sensor.npy').stat().st_mtime_ns==before


if __name__ == '__main__':
    import tempfile
    (ROOT/'.agent_tmp').mkdir(exist_ok=True)
    test_public_configurations_are_portable()
    print('PASS: portable configurations', flush=True)
    for test in [test_tactile_cache_matches_raw_windows_without_images]:
        with tempfile.TemporaryDirectory(prefix='paper-check-',dir=ROOT/'.agent_tmp') as directory:
            test(Path(directory))
        print('PASS:',test.__name__, flush=True)
