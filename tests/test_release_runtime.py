"""Exercise published presets and encoder hand-offs without datasets or a GPU."""
from pathlib import Path
import inspect
import gc
import sys
import unittest
from unittest.mock import patch
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def presets():
    return sorted(path.relative_to(ROOT / 'config').with_suffix('').as_posix()
                  for dataset in ('xela', 'socks', 'deco')
                  for path in (ROOT / 'config' / dataset).rglob('*.yaml')
                  if not path.name.startswith('.'))


def config(name):
    with initialize_config_dir(version_base='1.3', config_dir=str(ROOT / 'config')):
        cfg = compose(config_name=name)
    cfg.paths.output_dir = '/unused/output'
    cfg.paths.work_dir = str(ROOT)
    if 'normalization' in cfg.data:
        cfg.data.normalization.mean = [0.0] * (3 if name.startswith('xela/') else 1)
        cfg.data.normalization.std = [1.0] * (3 if name.startswith('xela/') else 1)
    if 'object_classes' in cfg.data:
        cfg.data.object_classes = ['a', 'b']
        cfg.data.object_class_weights = [1.0, 1.0]
    return cfg


class ReleaseRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        for name, resolver in {
            'int_multiply': lambda a, b: int(a * b),
            'int_divide': lambda a, b: a // b,
            'join': lambda sep, values: sep.join(map(str, values)),
        }.items():
            OmegaConf.register_new_resolver(name, resolver, replace=True)

    def test_all_targets_import_and_accept_configured_keywords(self):
        def visit(value, location):
            if isinstance(value, dict):
                if '_target_' in value:
                    target = hydra.utils.get_object(value['_target_'])
                    signature = inspect.signature(target)
                    if not any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
                        extra = set(value) - set(signature.parameters) - {
                            '_target_', '_partial_', '_recursive_', '_convert_'}
                        self.assertFalse(extra, f'{location}: unsupported {extra}')
                for key, child in value.items():
                    visit(child, f'{location}.{key}')
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, f'{location}[{index}]')
        for name in presets():
            with self.subTest(preset=name):
                visit(OmegaConf.to_container(config(name), resolve=True), name)

    def test_encoder_handoffs(self):
        # No pretrained image weights are needed to verify state-dict compatibility.
        pairs = [(f'xela/pretrain/{method}', f'xela/{task}/{method}')
                 for method in ('jepa', 'dino', 'mae', 'byol', 'ijepa')
                 for task in ('force', 'object', 'pose')]
        pairs += [(f'socks/{task}/pretrain/{method}', f'socks/{task}/downstream/{method}')
                  for task in ('action', 'pose') for method in ('jepa', 'dino')]
        pairs += [(f'deco/pretrain/{method}', f'deco/policy/{method}')
                  for method in ('jepa', 'mae', 'dino_cls', 'ijepa')]
        for source, destination in pairs:
            with self.subTest(source=source, destination=destination):
                pre, down = config(source), config(destination)
                source_cfg = pre.algorithm.get('encoder', pre.algorithm.get('backbone'))
                target_cfg = (down.task.model.tactile_encoder if destination.startswith('deco/')
                              else down.task.model_encoder)
                source_cfg = OmegaConf.create(OmegaConf.to_container(source_cfg, resolve=True))
                target_cfg = OmegaConf.create(OmegaConf.to_container(target_cfg, resolve=True))
                if 'Alex' in source_cfg._target_:
                    source_cfg.pretrained = False
                encoder = hydra.utils.instantiate(source_cfg)
                target = hydra.utils.instantiate(target_cfg)
                target.load_state_dict(encoder.state_dict(), strict=True)
                del encoder, target

    def test_pretrain_construction(self):
        from torchvision.models import alexnet
        with patch('tactile_ssl.model.tactile_alexnet.alexnet',
                   side_effect=lambda **kw: alexnet(weights=None)):
            for name in presets():
                if '/pretrain/' not in name:
                    continue
                with self.subTest(preset=name):
                    cfg = config(name)
                    model = hydra.utils.instantiate(cfg.algorithm)
                    self.assertTrue(any(p.requires_grad for p in model.parameters()))
                    del model
                    gc.collect()

    def test_downstream_forward_backward(self):
        # CPU attention uses PyTorch; pretrained image weights are irrelevant
        # here because policy inputs already contain frozen visual features.
        from tactile_ssl.model.layers import attention, block, decoder_block
        from torchvision.models import resnet18
        names = [name for name in presets() if '/pretrain/' not in name]
        with patch.object(attention, 'XFORMERS_AVAILABLE', False), \
             patch.object(block, 'XFORMERS_AVAILABLE', False), \
             patch.object(decoder_block, 'XFORMERS_AVAILABLE', False), \
             patch('torchvision.models.resnet18', side_effect=lambda **kw: resnet18(weights=None)):
            for name in names:
                with self.subTest(preset=name):
                    cfg = config(name)
                    model = hydra.utils.instantiate(cfg.task)
                    model.eval()
                    if name.startswith('deco/'):
                        output = model.model(
                            torch.randn(2, 3, 528, 5), None, torch.zeros(2, 28),
                            torch.zeros(2, dtype=torch.long), image_features=torch.randn(2, 2, 512))
                        self.assertEqual(tuple(output.shape), (2, 16, 12))
                    elif name.startswith('xela/'):
                        dataset_cfg = (cfg.data.dataset_list[0].dataset.config
                                       if 'dataset_list' in cfg.data else cfg.data.dataset.config)
                        channels = 6 if dataset_cfg.get('features', {}).get('use_spatial_coords', False) else 3
                        output = model.forward({'sensor': torch.randn(2, 10, 368, channels)}, 0)
                    else:
                        length = 45 if '/action/' in name else 60
                        batch = {'sensor': torch.randn(2, length, 453, 1),
                                 'left_grid': torch.randn(2, length, 32, 32),
                                 'right_grid': torch.randn(2, length, 32, 32)}
                        output = model.forward(batch, 0)
                    self.assertTrue(torch.isfinite(output).all())
                    output.square().mean().backward()
                    self.assertTrue(any(p.grad is not None for p in model.parameters() if p.requires_grad))
                    del model

    def test_jepa_loader_masks_and_training_step(self):
        from tactile_ssl.data.deco_geometry import build_deco_geometry
        from tactile_ssl.data.sock import SockGeometry
        from tactile_ssl.model.layers import attention, block, decoder_block
        deco_graph = build_deco_geometry().graph_dict()
        sock_graph = SockGeometry.from_csv(
            str(ROOT / 'assets/socks/sensor_map.csv'),
            str(ROOT / 'assets/socks/edges.csv')).graph_dict()
        edges = torch.stack([torch.arange(367), torch.arange(1, 368)])
        xela_graph = {'edge_index': edges, 'edge_attr': torch.ones(367, 1),
                      'edge_count': torch.tensor(367),
                      'node_group_id': torch.arange(368) // 16}
        with patch.object(attention, 'XFORMERS_AVAILABLE', False), \
             patch.object(block, 'XFORMERS_AVAILABLE', False), \
             patch.object(decoder_block, 'XFORMERS_AVAILABLE', False):
            for name in presets():
                if '/pretrain/' not in name:
                    continue
                cfg = config(name)
                if not cfg.algorithm._target_.endswith('XelaJEPAModule'):
                    continue
                with self.subTest(preset=name):
                    torch.manual_seed(42)
                    if cfg.data.sensor == 'deco':
                        shape, graph = (3, 528, 5), deco_graph
                        self.assertEqual(cfg.data.dataset.include_graph, 'ijepa' not in name)
                    elif cfg.data.sensor == 'sock':
                        shape, graph = (5, 453, 1), sock_graph
                    else:
                        shape, graph = (10, 368, 3), xela_graph
                    collate = hydra.utils.instantiate(cfg.data.train_dataloader.collate_fn)
                    batch = collate([{'sensor': torch.randn(*shape), 'graph': graph} for _ in range(2)])
                    model = hydra.utils.instantiate(cfg.algorithm).eval()
                    loss = model.training_step(batch, 0)['loss']
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertTrue(any(p.grad is not None for p in model.context_encoder.parameters()))
                    self.assertTrue(all(p.grad is None for p in model.target_encoder.parameters()))
                    del model

    def test_data_seed_is_independent_of_optimization_seed(self):
        for name in presets():
            cfg = config(name)
            cfg.seed = 17
            dataset = cfg.data.get('dataset', {})
            for key in ('split_seed', 'classification_split_seed'):
                if key in dataset:
                    with self.subTest(preset=name, key=key):
                        self.assertEqual(dataset[key], cfg.data_seed)

    def test_pretrained_commands_require_checkpoint_before_loading_data(self):
        from tactile_ssl.utils.run_config import ENTRYPOINTS, validate_run_config
        for name in presets():
            if '/pretrain/' in name:
                continue
            entrypoint = next(value for key, value in ENTRYPOINTS.items() if name.startswith(key))
            with self.subTest(preset=name):
                cfg = config(name)
                if name.rsplit('/', 1)[1] in ('e2e', 'random', 'vision'):
                    validate_run_config(cfg, entrypoint)
                else:
                    with self.assertRaisesRegex(ValueError, 'checkpoint='):
                        validate_run_config(cfg, entrypoint)
                    cfg.checkpoint = '/nonexistent/encoder.ckpt'
                    with self.assertRaises(FileNotFoundError):
                        validate_run_config(cfg, entrypoint)
        with self.assertRaisesRegex(ValueError, 'train_task_deco_policy.py'):
            validate_run_config(config('deco/policy/jepa'), 'train_task_object.py')

    def test_readme_training_commands_compose_in_their_entrypoints(self):
        import re
        commands = re.findall(r'^python (train\S*\.py) --config-name (\S+)([^\n]*)',
                              (ROOT / 'README.md').read_text(), flags=re.MULTILINE)
        self.assertTrue(commands)
        def run(command):
            entrypoint, name, args = command
            overrides = [arg for arg in args.split() if '=' in arg and not arg.startswith('--')]
            return subprocess.run([sys.executable, entrypoint, '--config-name', name,
                                   *overrides, '--cfg', 'job', '--resolve'],
                                  cwd=ROOT, text=True, capture_output=True, timeout=90)
        with ThreadPoolExecutor(max_workers=2) as executor:
            for command, result in zip(commands, executor.map(run, commands)):
                with self.subTest(entrypoint=command[0], preset=command[1]):
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_deco_historical_checkpoint_split_is_checked(self):
        from tactile_ssl.utils.run_config import validate_run_config
        scratch = ROOT / '.agent_tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            (root / 'checkpoints').mkdir()
            checkpoint = root / 'checkpoints/epoch-0150.ckpt'
            checkpoint.touch()
            (root / '.hydra').mkdir()
            saved = OmegaConf.create({'seed': 17, 'data_seed': 42,
                                      'data': {'dataset': {'split_seed': '${seed}'}}})
            OmegaConf.save(saved, root / '.hydra/config.yaml')
            cfg = config('deco/policy/jepa')
            cfg.checkpoint = str(checkpoint)
            with self.assertRaisesRegex(ValueError, 'split_seed=17'):
                validate_run_config(cfg, 'train_task_deco_policy.py')
            saved.data.dataset.split_seed = '${data_seed}'
            OmegaConf.save(saved, root / '.hydra/config.yaml')
            validate_run_config(cfg, 'train_task_deco_policy.py')
            cfg.data_seed = 17
            with self.assertRaisesRegex(ValueError, 'does not resplit'):
                validate_run_config(cfg, 'train_task_deco_policy.py')


if __name__ == '__main__':
    unittest.main(verbosity=2)
