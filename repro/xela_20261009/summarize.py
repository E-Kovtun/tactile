"""Recompute the workbook metric definitions from a reproduced run tree."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

from .run import BUNDLE, manifest, output_dir

METRICS = [('force', 'rmse'), ('object', 'accuracy'), ('pose', 'x@2cm'),
           ('pose', 'y@2cm'), ('pose', 'theta@5deg')]


def read_predictions(path, task):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        truth, pred = z['y_true'], z['y_pred']
        if task == 'object':
            if pred.ndim > truth.ndim:
                pred = pred.argmax(-1)
            assert pred.shape == truth.shape
            return {'accuracy': float(np.mean(pred == truth))}
        assert pred.shape == truth.shape
        diff = pred.astype(np.float64) - truth.astype(np.float64)
        assert np.isfinite(diff).all()
        if task == 'force':
            return {'rmse': float(np.sqrt(np.mean(diff * diff)))}
        return {name: float(np.mean(np.abs(diff[..., index]) < threshold))
                for index, name, threshold in [(0, 'x@2cm', .02), (1, 'y@2cm', .02), (2, 'theta@5deg', 5.)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, required=True, help='Root of newly reproduced runs.')
    args = p.parse_args()
    source = manifest()
    rows = []
    for stage in source['stages']:
        if stage['stage'] == 'pretrain': continue
        path = output_dir(stage, args.output_root) / 'evaluation/test_predictions.npz'
        if path.is_file():
            rows.append({'case': stage['case'], 'seed': stage['pretrain_seed'], 'head': stage['head_seed'],
                         'task': stage['task'], 'metrics': read_predictions(path, stage['task'])})
    writer = csv.writer(sys.stdout)
    writer.writerow(['case', 'metric', 'mean', 'sample_sd', 'heads', 'p42_mean', 'p42_sd', 'p42_heads',
                     'p17_mean', 'p17_sd', 'p17_heads', 'p3407_mean', 'p3407_sd', 'p3407_heads'])
    for c in source['cases']:
        for task, metric in METRICS:
            groups = [[r['metrics'][metric] for r in rows if r['case'] == c['id'] and r['task'] == task
                       and (seed is None or int(r['seed']) == seed)] for seed in [None, 42, 17, 3407]]
            values = []
            for group in groups:
                values.extend([statistics.mean(group) if group else None,
                               statistics.stdev(group) if len(group) > 1 else None, len(group)])
            writer.writerow([c['id'], metric, *values])


if __name__ == '__main__':
    main()
