import pickle

import numpy as np

from tactile_ssl.data.xela.preprocessing import compute_baseline_mean, compute_xela_interpolated


def test_baseline_mean_preserves_legacy_float64_precision(tmp_path):
    baseline = np.array(
        [
            [[0.0, 20000.001, 30000.003, 40000.005]],
            [[1.0, 20000.004, 30000.006, 40000.008]],
            [[2.0, 20000.007, 30000.009, 40000.011]],
        ],
        dtype=np.float64,
    )
    baseline_path = tmp_path / "baseline.pkl"
    with baseline_path.open("wb") as f:
        pickle.dump(baseline, f)

    actual = compute_baseline_mean(str(baseline_path))["baseline_mean"]
    expected = np.mean(baseline[:, :, 1:], axis=0)

    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)
    assert np.any(actual != actual.astype(np.float32).astype(np.float64))


def test_interpolated_xela_stays_float64_until_dataset_getitem():
    xela = np.array(
        [
            [[0.0, 25000.001, 30000.003, 35000.005]],
            [[1.0, 25000.007, 30000.009, 35000.011]],
        ],
        dtype=np.float64,
    )
    timestamps = np.array([0.0, 1.0], dtype=np.float64)
    baseline = np.array([[25000.004, 30000.006, 35000.008]], dtype=np.float64)

    result = compute_xela_interpolated(
        xela_array=xela,
        timestamps=timestamps,
        interpolating_freq=2,
        smooth_data=False,
        outlier_min=20000,
        outlier_max=60000,
        subtract_baseline=True,
        baseline_mean=baseline,
    )

    assert result["xela_array"].dtype == np.float64
    np.testing.assert_allclose(
        result["xela_array"],
        xela[:, :, 1:] - baseline[None],
        rtol=0.0,
        atol=0.0,
    )
