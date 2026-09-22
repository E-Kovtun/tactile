"""Public package metadata and cleanup contracts; standard library only."""
import configparser
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import mock_open, patch

ROOT = Path(__file__).resolve().parents[1]


class ReleasePackagingTests(unittest.TestCase):
    def test_package_metadata(self):
        captured = {}
        stub = SimpleNamespace(
            setup=lambda **kwargs: captured.update(kwargs),
            find_packages=lambda: ["tactile_ssl"],
        )
        with patch.dict(sys.modules, {"setuptools": stub}), \
             patch("builtins.open", mock_open(read_data=(ROOT / "README.md").read_text())):
            runpy.run_path(str(ROOT / "setup.py"))
        self.assertEqual(captured["name"], "tactile_ssl")
        self.assertEqual(captured["python_requires"], ">=3.10")
        self.assertEqual(captured["license"], "CC-BY-NC-4.0")
        self.assertNotIn("author", captured)
        self.assertNotIn("url", captured)
        self.assertFalse(any("OSI Approved" in value for value in captured["classifiers"]))
        self.assertTrue((ROOT / "LICENSE.md").read_text().startswith(
            "Attribution-NonCommercial 4.0 International"))

    def test_development_settings(self):
        settings = configparser.ConfigParser()
        settings.read(ROOT / "setup.cfg")
        self.assertEqual(settings["mypy"]["python_version"], "3.10")
        self.assertNotIn("copyright-check", settings["flake8"])
        self.assertNotIn("copyright-regexp", settings["flake8"])
        readme = (ROOT / "README.md").read_text()
        self.assertIn("conda env create -f tactile_environment.yml", readme)
        self.assertTrue((ROOT / "tactile_environment.yml").is_file())

    def test_obsolete_public_examples_are_absent(self):
        for name in (
            "inference.py", "environment.yml", "local_env.sh",
            "scripts/xela/generate_policy_dataset.py",
            "tactile_ssl/data/xela_joystick_control.py",
            "tactile_ssl/downstream_task/joystick_dir.py",
            "tactile_ssl/data/._sock.py",
            "tactile_ssl/downstream_task/._force_sl.py",
            "tactile_ssl/downstream_task/._sl_module.py",
            "tactile_ssl/downstream_task/._sock.py",
            "tactile_ssl/downstream_task/._xela_object.py",
            "tactile_ssl/downstream_task/._xela_relativepose.py",
            "tactile_ssl/model/._xela_transformer.py",
        ):
            with self.subTest(path=name):
                self.assertFalse((ROOT / name).exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
