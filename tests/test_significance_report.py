import csv

import pytest

openpyxl = pytest.importorskip("openpyxl")

from tactile_ssl.evaluation.report import write_reports


def _row(
    method,
    experiment_id,
    baseline,
    metric,
    estimate,
    variant="only signal",
    block_id=None,
    block_name=None,
    baseline_name=None,
):
    row = {
        "task": "force",
        "experiment_id": experiment_id,
        "method": method,
        "is_baseline": baseline,
        "variant": variant,
        "metric": metric,
        "estimate": estimate,
        "bootstrap_se": 0.01,
        "delta": None if baseline else -0.1,
        "delta_se": None if baseline else 0.02,
        "improvement": None if baseline else 0.1,
        "raw_pvalue": None if baseline else 0.02,
        "checkpoint": "best.ckpt",
        "artifact_path": "evaluation/test_predictions.npz",
        "num_examples": 10,
        "num_groups": 2,
        "dataset_fingerprint": "abc",
        "seed": 42,
    }
    if block_id is not None:
        row.update(
            {
                "comparison_block_id": block_id,
                "comparison_block_name": block_name,
                "baseline_id": f"{block_id}_baseline",
                "baseline_name": baseline_name,
            }
        )
    return row


def test_report_headers_rows_and_flat_csv(tmp_path):
    metrics = ("rmse", "rmse_x", "rmse_y", "rmse_z")
    rows = [_row("Original", "original", True, metric, 1.0) for metric in metrics]
    rows += [_row("WL", "wl", False, metric, 0.9) for metric in metrics]
    xlsx_path, csv_path = write_reports(tmp_path, rows, [])

    workbook = openpyxl.load_workbook(xlsx_path, data_only=False)
    sheet = workbook["Force Estimation"]
    assert sheet["A1"].value == "Force Estimation"
    assert sheet["A2"].value == "method name"
    assert "B1:C1" in {str(cell_range) for cell_range in sheet.merged_cells.ranges}
    assert sheet["A3"].value == "Original"
    assert sheet["A4"].value == "WL"
    assert sheet["A5"].value == "Δ vs Original (improvement)"
    assert sheet["A6"].value == "p-value"
    assert "Statistics" in workbook.sheetnames
    assert "Experiments" in workbook.sheetnames

    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 8
    assert csv_rows[0]["metric"] == "rmse"


def test_report_uses_an_independent_baseline_for_each_block(tmp_path):
    metrics = ("rmse", "rmse_x", "rmse_y", "rmse_z")
    rows = []
    for metric in metrics:
        rows.append(
            _row(
                "Signal baseline",
                "signal_base",
                True,
                metric,
                1.0,
                block_id="signal",
                block_name="Only signal",
                baseline_name="Signal baseline",
            )
        )
        rows.append(
            _row(
                "Signal method",
                "signal_method",
                False,
                metric,
                0.9,
                block_id="signal",
                block_name="Only signal",
                baseline_name="Signal baseline",
            )
        )
        rows.append(
            _row(
                "Position baseline",
                "position_base",
                True,
                metric,
                0.8,
                block_id="position",
                block_name="Signal + coordinates",
                baseline_name="Position baseline",
            )
        )
        rows.append(
            _row(
                "Position method",
                "position_method",
                False,
                metric,
                0.7,
                block_id="position",
                block_name="Signal + coordinates",
                baseline_name="Position baseline",
            )
        )

    xlsx_path, _ = write_reports(tmp_path, rows, [])
    sheet = openpyxl.load_workbook(xlsx_path)["Force Estimation"]
    assert sheet["A3"].value == "Only signal"
    assert sheet["A4"].value == "Signal baseline"
    assert sheet["A6"].value == "Δ vs Signal baseline (improvement)"
    assert sheet["A8"].value == "Signal + coordinates"
    assert sheet["A9"].value == "Position baseline"
    assert sheet["A11"].value == "Δ vs Position baseline (improvement)"
    assert sheet["A4"].fill.fgColor.rgb == "FFD9EAF7"
    assert sheet["A5"].fill.fgColor.rgb == "FFE2F0D9"
    assert sheet["A6"].fill.fgColor.rgb == "FFFCE4D6"
    assert sheet["A7"].fill.fgColor.rgb == "FFFFF2CC"


def test_means_report_keeps_blocks_and_baselines_without_statistics(tmp_path):
    metrics = ("rmse", "rmse_x", "rmse_y", "rmse_z")
    rows = []
    for metric in metrics:
        rows.append(
            _row(
                "Signal baseline",
                "signal_base",
                True,
                metric,
                1.0,
                block_id="signal",
                block_name="Only signal",
                baseline_name="Signal baseline",
            )
        )
        rows.append(
            _row(
                "Signal method",
                "signal_method",
                False,
                metric,
                0.9,
                block_id="signal",
                block_name="Only signal",
                baseline_name="Signal baseline",
            )
        )

    write_reports(tmp_path, rows, [])

    means_path = tmp_path / "downstream_means.xlsx"
    workbook = openpyxl.load_workbook(means_path, data_only=False)
    assert workbook.sheetnames == ["Force Estimation"]
    sheet = workbook["Force Estimation"]
    assert sheet["A3"].value == "Only signal"
    assert sheet["A4"].value == "Signal baseline"
    assert sheet["A5"].value == "Signal method"
    assert sheet.max_row == 5
    assert sheet["C4"].value == pytest.approx(1.0)
    assert sheet["C5"].value == pytest.approx(0.9)
    assert sheet["C4"].number_format == "0.0000"
    assert sheet["A4"].fill.fgColor.rgb == "FFD9EAF7"
    assert sheet["A5"].fill.fgColor.rgb == "FFE2F0D9"
    assert all(
        "p-value" not in str(cell.value) and "improvement" not in str(cell.value)
        for row in sheet.iter_rows()
        for cell in row
    )
