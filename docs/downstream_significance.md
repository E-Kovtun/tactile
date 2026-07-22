# Statistical significance of downstream experiments

New force, relative-pose and object-classification test runs automatically write:

```text
<run root>/evaluation/test_predictions.npz
<run root>/evaluation/manifest.json
```

The analyzer accepts any subset of the three tasks. Copy
`config/significance/example.yaml`, set the experiment directories and select one
experiment ID as the baseline in each task.

```bash
python scripts/analyze_downstream_significance.py --config-name my_analysis
```

If an experiment has no evaluation artifact, the analyzer performs test inference
only. It selects `best.ckpt`, then `last.ckpt`, then the checkpoint with the largest
epoch number. It never calls the training loop and uses a no-op logger, so no new
TensorBoard or W&B data is created.

`directory` may be an exact run root or a glob. Glob patterns are rejected when
ambiguous unless the experiment explicitly sets `select: latest`; incomplete runs
without an artifact or any checkpoint are ignored.

Normally the report column is inferred from `use_spatial_coords`. For experiments
whose downstream head consumes coordinates while the compared SSL encoder is
signal-only, set `report_variant: only_signal` explicitly.

Outputs are written below `output_dir`:

```text
significance_report.xlsx
downstream_means.xlsx
significance_results.csv
```

`downstream_means.xlsx` preserves the presentation layout and the baseline row
for every comparison block, but contains only one row per method with numeric
mean estimates. It intentionally omits standard errors, deltas, improvements
and p-values.

The comparison is paired by `(sample_id, occurrence)`. Different sample-ID
multisets are rejected instead of silently intersected. Bootstrap and permutation
resampling operate on whole recording/episode groups.

In presentation sheets, `estimate ± value` means the global test estimate plus or
minus its cluster-bootstrap standard error. Delta is always `candidate - baseline`;
positive improvement means lower RMSE or higher accuracy.
