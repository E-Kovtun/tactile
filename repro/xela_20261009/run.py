"""Portable, explicit one-stage runner for the 2026-10-09 Xela workbook."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parents[1]


def manifest():
    return json.loads((BUNDLE / 'manifest.json').read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def select(args):
    found = [s for s in manifest()['stages']
             if (args.case == 'all' or s['case'] == args.case)
             and (args.pretrain_seed is None or s['pretrain_seed'] == args.pretrain_seed)
             and (args.stage is None or s['stage'] == args.stage)]
    if not found:
        raise ValueError('No matching stages; use list to see the IDs.')
    return found


def environment(args):
    env = dict(os.environ)
    env.update(XELA_DATA_ROOT=str(Path(args.data_root).expanduser().resolve()),
               XELA_REPRO_OUTPUT_ROOT=str(Path(args.output_root).expanduser().resolve()),
               XELA_REPRO_CACHE_ROOT=str(Path(args.cache_root or Path(args.output_root) / '.cache').expanduser().resolve()),
               XELA_REPRO_DIRECTORY_ORDER=str(BUNDLE / 'directory_order.json'),
               PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1', OPENBLAS_NUM_THREADS='1')
    env['LD_LIBRARY_PATH'] = str(Path(sys.prefix) / 'lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    return env


def output_dir(s, output_root):
    return Path(output_root) / s['case'] / f"p{s['pretrain_seed']}" / s['stage']


def command(s, env):
    config = BUNDLE / s['config']
    entrypoint = 'train.py' if s['stage'] == 'pretrain' else 'scripts/train_task_seed_isolated.py'
    out = output_dir(s, env['XELA_REPRO_OUTPUT_ROOT'])
    return [sys.executable, str(ROOT / entrypoint), '--config-path', str(config.parent),
            '--config-name', config.stem, f'hydra.run.dir={out}']


def resolve_config(stage, env):
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver('int_multiply', lambda a, b: int(a * b), replace=True)
    OmegaConf.register_new_resolver('int_divide', lambda a, b: a // b, replace=True)
    OmegaConf.register_new_resolver('join', lambda s, xs: s.join(map(str, xs)), replace=True)
    text = (BUNDLE / stage['config']).read_text()
    text = text.replace('${hydra:runtime.cwd}', str(ROOT))
    text = text.replace('${hydra:runtime.output_dir}', str(output_dir(stage, env['XELA_REPRO_OUTPUT_ROOT'])))
    previous = {key: os.environ.get(key) for key in env if key.startswith('XELA_')}
    try:
        os.environ.update({key: value for key, value in env.items() if key.startswith('XELA_')})
        cfg = OmegaConf.create(text)
        OmegaConf.resolve(cfg)
        return cfg
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def instantiate_for_validation(node):
    """Construction-only fixtures for values normally filled by the data loader.

    These never modify stored configs or the training path. Real training computes
    normalization and class lists from its datasets before model construction.
    """
    import hydra
    from omegaconf import OmegaConf
    value = OmegaConf.to_container(node, resolve=True)
    def fill(item):
        if isinstance(item, dict):
            norm = item.get('normalization')
            if isinstance(norm, dict):
                if norm.get('mean') is None: norm['mean'] = [0., 0., 0.]
                if norm.get('std') is None: norm['std'] = [1., 1., 1.]
            if 'classes' in item and item['classes'] is None:
                item['classes'] = [f'fixture_{i}' for i in range(14)]
            for child in item.values(): fill(child)
        elif isinstance(item, list):
            for child in item: fill(child)
    fill(value)
    return hydra.utils.instantiate(OmegaConf.create(value))


def validate(args):
    import hydra
    import yaml
    from omegaconf import OmegaConf
    data = manifest()
    env = environment(args)
    ids = {s['id'] for s in data['stages']}
    targets, models, errors = set(), set(), []
    for s in select(args):
        try:
            assert sha(BUNDLE / s['config']) == s['config_sha256'], 'config hash mismatch'
            cfg = resolve_config(s, env)
            assert int(cfg.seed) == s['seed']
            assert cfg.trainer.devices == s['devices']
            assert cfg.data.train_dataloader.batch_size == s['train_batch_per_device']
            if s.get('depends_on'):
                assert s['depends_on'] in ids
                assert int(cfg.data_seed) == s['data_seed']
            def visit(value):
                if isinstance(value, dict):
                    if '_target_' in value:
                        targets.add(value['_target_'])
                    for item in value.values(): visit(item)
                elif isinstance(value, list):
                    for item in value: visit(item)
            visit(OmegaConf.to_container(cfg, resolve=True))
            # Every non-storage setting must match its original composed config.
            original = yaml.safe_load((BUNDLE / 'original_configs' / Path(s['config']).relative_to('configs')).read_text())
            portable = yaml.safe_load((BUNDLE / s['config']).read_text())
            def strip_storage(d):
                d.pop('paths', None)
                d.pop('ckpt_path', None)
                d.pop('resume_id', None)
                if 'cache' in d['data']: d['data']['cache'].pop('root', None)
                if 'task' in d: d['task'].pop('checkpoint_encoder', None)
                if 'jepa_pretrain' in d: d['jepa_pretrain'].pop('checkpoint_encoder', None)
                return d
            assert strip_storage(original) == strip_storage(portable), 'non-storage setting changed'
            if args.models:
                if s['stage'] == 'pretrain':
                    model_cfgs = [cfg.algorithm]
                else:
                    model_cfgs = [cfg.task.model_encoder, cfg.task.model_task]
                for node in model_cfgs:
                    key = OmegaConf.to_yaml(node, resolve=True)
                    if key not in models:
                        model = instantiate_for_validation(node)
                        if hasattr(model, 'parameters'):
                            assert sum(p.numel() for p in model.parameters()) > 0
                        del model
                        models.add(key)
        except Exception as exc:
            errors.append({'stage': s['id'], 'error': repr(exc)})
    for target in sorted(targets):
        try:
            hydra.utils.get_object(target)
        except Exception as exc:
            errors.append({'target': target, 'error': repr(exc)})
    for rel, expected in json.loads((BUNDLE / 'code_sha256.json').read_text()).items():
        if sha(ROOT / rel) != expected:
            errors.append({'source': rel, 'error': 'frozen source hash mismatch'})
    result = {'stages': len(select(args)), 'targets': len(targets), 'models': len(models),
              'construction_fixtures': 'Data-dependent normalization uses zero/one and class lists use 14 placeholder labels; saved training configs are unchanged.', 'errors': errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return bool(errors)


def run(args):
    stages = select(args)
    if len(stages) != 1:
        raise ValueError('run requires one --case, --pretrain-seed and --stage.')
    s = stages[0]
    env = environment(args)
    cfg = resolve_config(s, env)
    out = output_dir(s, env['XELA_REPRO_OUTPUT_ROOT'])
    if out.exists():
        raise ValueError(f'Refusing to overwrite an existing run directory: {out}. Select a fresh output root.')
    if not Path(env['XELA_DATA_ROOT']).is_dir():
        raise FileNotFoundError(env['XELA_DATA_ROOT'])
    devices = [x.strip() for x in args.gpus.split(',') if x.strip()]
    if len(devices) != int(s['devices']):
        raise ValueError(f"This original run requires {s['devices']} GPU(s), preserving per-device batch size {s['train_batch_per_device']}.")
    if s['stage'] != 'pretrain' and not Path(cfg.task.checkpoint_encoder).is_file():
        raise FileNotFoundError(f"Run dependency {s['depends_on']} first: {cfg.task.checkpoint_encoder}")
    for rel, expected in json.loads((BUNDLE / 'code_sha256.json').read_text()).items():
        if sha(ROOT / rel) != expected:
            raise ValueError(f'Frozen implementation was modified: {rel}')
    env['CUDA_VISIBLE_DEVICES'] = args.gpus
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = '1' if s['task'] in {'force', 'object'} else '4'
    cmd = command(s, env)
    out.mkdir(parents=True, exist_ok=False)
    record = {'stage': s['id'], 'command': cmd, 'config_sha256': s['config_sha256'],
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'environment': {k: env[k] for k in ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'XELA_DATA_ROOT', 'XELA_REPRO_OUTPUT_ROOT', 'XELA_REPRO_CACHE_ROOT']}}
    (out / 'reproduction.json').write_text(json.dumps(record, indent=2) + '\n')
    print(shlex.join(cmd), flush=True)
    with (out / 'training.log').open('w') as log:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    (out / 'exit_status.json').write_text(json.dumps({'returncode': proc.returncode}) + '\n')
    return proc.returncode


def verify_data(args):
    data = json.loads((BUNDLE / 'data_manifest.json').read_text())
    root = Path(args.data_root).expanduser().resolve()
    errors = []
    for item in data['files']:
        path = root / item['path']
        if not path.is_file() or path.stat().st_size != item['size']:
            errors.append(item['path'])
        elif args.hashes and sha(path) != item['sha256']:
            errors.append(item['path'])
    print(json.dumps({'files': len(data['files']), 'full_hashes': args.hashes, 'mismatches': errors}, indent=2))
    return bool(errors)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['list', 'plan', 'run', 'validate', 'verify-data'])
    p.add_argument('--case', default='all')
    p.add_argument('--pretrain-seed', type=int, choices=[42, 17, 3407])
    p.add_argument('--stage', help='pretrain, force-h42, object-h17, pose-h3407, etc.')
    p.add_argument('--data-root', default=os.environ.get('XELA_DATA_ROOT', 'sparsh-skin-dataset'))
    p.add_argument('--output-root', default='outputs/xela_reproduction')
    p.add_argument('--cache-root')
    p.add_argument('--gpus', help='Physical indices already allocated to this stage, e.g. 0 or 0,1,2,3.')
    p.add_argument('--models', action='store_true', help='CPU construction of distinct model configurations.')
    p.add_argument('--hashes', action='store_true', help='Read every data file and compare its SHA-256.')
    args = p.parse_args()
    if args.action == 'list':
        for c in manifest()['cases']:
            counts = Counter(s['stage'] == 'pretrain' for s in manifest()['stages'] if s['case'] == c['id'])
            print(f"{c['id']:12} {counts[True]:2} pretrains, {counts[False]:2} heads | {c['name']}")
        return 0
    if args.action == 'plan':
        env = environment(args)
        rows = []
        for s in select(args):
            rows.append({'id': s['id'], 'depends_on': s.get('depends_on'), 'gpus': s['devices'],
                         'heads_per_gpu': 3 if s['stage'] != 'pretrain' else None,
                         'command': [sys.executable, '-m', 'repro.xela_20261009.run', 'run', '--case', s['case'],
                                     '--pretrain-seed', str(s['pretrain_seed']), '--stage', s['stage'],
                                     '--data-root', env['XELA_DATA_ROOT'], '--output-root', env['XELA_REPRO_OUTPUT_ROOT'],
                                     '--cache-root', env['XELA_REPRO_CACHE_ROOT'], '--gpus', '<allocated GPU indices>']})
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    if args.action == 'validate': return validate(args)
    if args.action == 'verify-data': return verify_data(args)
    if not args.gpus: p.error('run requires --gpus')
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
