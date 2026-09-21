"""CPU regression tests for task loaders and the shared training lifecycle."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from omegaconf import OmegaConf

import train_task_socks_action as action
import train_task_socks_pose as pose
import train_task_deco_policy as policy
from tactile_ssl.trainer import downstream


class Dataset(torch.utils.data.Dataset):
    input_mean = np.array([1., 2.])
    input_std = np.array([3., 4.])
    classes = ['a', 'b']
    class_weights = np.array([.4, .6])

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return index


class EntrypointTests(unittest.TestCase):
    def test_loaders_preserve_metadata_and_test_split(self):
        for module, sensor in [(action, 'sock'), (pose, 'sock'), (policy, 'deco')]:
            with self.subTest(module=module.__name__):
                train, val, test = Dataset(), Dataset(), Dataset()
                train.test_dataset = test
                cfg = OmegaConf.create({'data': {
                    'sensor': sensor, 'dataset': {}, 'normalization': {},
                    'train_dataloader': {'batch_size': 2},
                    'val_dataloader': {'batch_size': 3},
                    'test_dataloader': {'batch_size': 4},
                }})
                datasets = (train, val) if module is policy else (train, val, test)
                with patch.object(module.hydra.utils, 'instantiate', return_value=datasets):
                    loaders = module.get_dataloaders(cfg)
                self.assertIs(loaders[0].dataset, train)
                self.assertIs(loaders[1].dataset, val)
                self.assertIs(loaders[2].dataset, test)
                self.assertEqual(loaders[2].batch_size, 4 if module is pose else 3)
                if module is not policy:
                    self.assertEqual(list(cfg.data.normalization.mean), [1., 2.])
                    self.assertEqual(list(cfg.data.normalization.std), [3., 4.])
                if module is action:
                    self.assertEqual(list(cfg.data.object_classes), ['a', 'b'])
                    self.assertEqual(list(cfg.data.object_class_weights), [.4, .6])

    def test_lifecycle_seeds_and_evaluation_checkpoint(self):
        for saved, early in [(False, False), (True, True), (True, False)]:
            with self.subTest(saved=saved, early=early), tempfile.TemporaryDirectory(
                dir=Path(__file__).resolve().parents[1] / '.agent_tmp'
            ) as directory:
                cfg = OmegaConf.create({'seed': 17, 'data_seed': 42, 'task': {},
                    'trainer': {}, 'tensorboard': {}, 'paths': {'output_dir': directory},
                    'ckpt_path': None})
                events = []
                def loaders(_):
                    events.append('data')
                    return 'train', 'val', 'test'
                def model(_):
                    events.append('model')
                    return 'model'
                trainer, writer = Mock(), Mock()
                trainer.use_early_stopping = early
                trainer.checkpoint_dir = directory
                trainer.early_stopping_checkpoint_name = 'best'
                checkpoint = str(Path(directory) / ('best.ckpt' if early else 'last.ckpt'))
                Path(checkpoint).touch()
                trainer.get_latest_checkpoint.return_value = checkpoint
                with patch.object(downstream, 'init_tensorboard', return_value=writer), \
                     patch.object(downstream, 'print_config_tree'), \
                     patch.object(downstream, 'seed_everything', side_effect=lambda seed, **kw: events.append(seed)), \
                     patch.object(downstream.hydra.utils, 'instantiate', side_effect=model), \
                     patch.object(downstream, 'Trainer', return_value=trainer):
                    downstream.train_downstream(cfg, loaders, evaluate_saved_checkpoint=saved)
                self.assertEqual(events, [42, 'data', 17, 'model'])
                trainer.fit.assert_called_once_with('model', 'train', 'val', ckpt_path=None)
                if saved:
                    trainer.evaluate.assert_called_once_with('model', 'test', ckpt_path_to_eval=checkpoint)
                else:
                    trainer.evaluate.assert_called_once_with('model', 'test')
                writer.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
