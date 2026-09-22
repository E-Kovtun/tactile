# Contributing to Tactile-JEPA

Bug reports, documentation improvements, and focused pull requests are welcome.
Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting issues

Include the command and configuration, relevant package versions, expected behavior,
and a minimal reproducer or error traceback. Do not include credentials or private data.

## Pull requests

1. Create a branch from the repository's default branch.
2. Keep the change focused; add tests for changed behavior and update affected documentation.
3. Follow the surrounding code style and describe how you tested the change.

Set up the environment using [README.md](README.md). From the repository root, run:

```bash
python tests/test_release_packaging.py
python tests/test_paper_release.py
python tests/test_release_runtime.py
```

These checks run on CPU without datasets or pretrained weights. They validate
configurations and model interfaces, not full training or published metrics.
For downloader changes, also run `python -m pytest tests/test_download_datasets.py`
after installing the optional test dependency with `python -m pip install pytest`.

## License

Contributions are provided under the repository's [license](LICENSE.md).
Preserve existing copyright notices and attribution for third-party code.
