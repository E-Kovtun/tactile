"""Validate public training commands before loading data or allocating devices."""
from pathlib import Path

from omegaconf import OmegaConf


ENTRYPOINTS = {
    'xela/pretrain/': 'train.py',
    'socks/action/pretrain/': 'train.py',
    'socks/pose/pretrain/': 'train.py',
    'socks/pretrain/': 'train.py',
    'deco/pretrain/': 'train.py',
    'xela/force/': 'train_task_force.py',
    'xela/object/': 'train_task_object.py',
    'xela/pose/': 'train_task_pose_estimation.py',
    'socks/action/downstream/': 'train_task_socks_action.py',
    'socks/pose/downstream/': 'train_task_socks_pose.py',
    'deco/policy/': 'train_task_deco_policy.py',
}


def validate_run_config(cfg, entrypoint):
    name = str(cfg.get('experiment_name', ''))
    for prefix, expected in ENTRYPOINTS.items():
        if name.startswith(prefix) and entrypoint != expected:
            raise ValueError(f'{name} requires {expected}; run: python {expected} --config-name {name}')
    if name.startswith('deco/policy/') and int(cfg.get('data_seed', 42)) != 42:
        raise ValueError('Prepared DECO policy caches use data_seed=42; changing data_seed does not resplit a cache.')
    if 'task' not in cfg:
        return

    def check(node):
        if not OmegaConf.is_dict(node):
            return
        if 'checkpoint_encoder' in node:
            # Frozen-random, vision-only and e2e presets deliberately use a
            # literal null. Published pretrained probes refer to ${checkpoint}.
            raw = OmegaConf.to_container(node, resolve=False)['checkpoint_encoder']
            checkpoint = node.checkpoint_encoder
            if raw == '${checkpoint}' and not checkpoint:
                raise ValueError(
                    'A pretrained encoder is required. Pass checkpoint=/path/to/encoder.ckpt '
                    'or select an e2e/random preset to train without one.'
                )
            if checkpoint and not Path(checkpoint).expanduser().is_file():
                raise FileNotFoundError(f'Encoder checkpoint does not exist: {checkpoint}')
            if checkpoint and name.startswith('deco/policy/'):
                check_deco_checkpoint_split(checkpoint, cfg.get('data_seed', 42))
        for key in node:
            if not OmegaConf.is_missing(node, key) and OmegaConf.is_dict(node[key]):
                check(node[key])

    check(cfg.task)


def check_deco_checkpoint_split(checkpoint, data_seed):
    """Historical raw-DECO pretrains sometimes used the optimization seed to split data."""
    run = Path(checkpoint).expanduser().resolve().parent.parent
    for sidecar in (run / '.hydra/config.yaml', run / 'config.yaml'):
        if sidecar.is_file():
            saved = OmegaConf.load(sidecar)
            split_seed = OmegaConf.select(saved, 'data.dataset.split_seed')
            if split_seed is not None and int(split_seed) != int(data_seed):
                raise ValueError(
                    f'DECO encoder split_seed={split_seed} differs from policy data_seed={data_seed}. '
                    'Its pretraining data may include policy validation/test episodes. '
                    'Use an encoder pretrained on the matching split; changing this checkpoint\'s '
                    'configuration does not fix the trained weights.'
                )
            return
