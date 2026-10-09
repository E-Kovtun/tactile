"""Optional recorded episode enumeration; default behavior is unchanged."""
from functools import lru_cache
import json
import os
from pathlib import Path


@lru_cache(maxsize=4)
def _orders(filename):
    return json.loads(Path(filename).read_text())['directories']


def episode_directories(directory):
    path = Path(directory)
    children = [p for p in path.iterdir() if p.is_dir()]
    filename = os.environ.get('XELA_REPRO_DIRECTORY_ORDER')
    if not filename:
        return children
    root = Path(os.environ['XELA_DATA_ROOT']).resolve()
    key = path.resolve().relative_to(root).as_posix()
    order = _orders(filename)[key]
    by_name = {p.name: p for p in children}
    if set(by_name) != set(order):
        raise ValueError(f'Episode directories differ from the reproduction manifest: {key}')
    return [by_name[name] for name in order]
