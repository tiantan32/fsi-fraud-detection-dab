"""Unit tests for src/lib/fairness.py.

These run in GitHub Actions on every PR. They do NOT need a Spark session,
MLflow, or a Databricks workspace, which is the whole point of pulling the
math out of the notebook in the first place.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from lib.fairness import (
    group_metrics,
    is_fair,
    select_protected_columns,
    worst_disparate_impact,
)

# ----------------------------- group_metrics -----------------------------

def test_group_metrics_perfect_classifier():
    df = pd.DataFrame({"country": ["US"] * 4 + ["UK"] * 4})
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    yp = y.copy()  # perfect predictions

    out = group_metrics(df, "country", y, yp)
    assert set(out["slice_value"]) == {"US", "UK"}
    for row in out.to_dict("records"):
        assert row["precision"] == 1.0
        assert row["recall"] == 1.0
        assert row["false_positive_rate"] == 0.0
        assert row["n"] == 4


def test_group_metrics_worst_case_classifier():
    df = pd.DataFrame({"country": ["US"] * 4 + ["UK"] * 4})
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    yp = 1 - y  # always wrong

    out = group_metrics(df, "country", y, yp)
    for row in out.to_dict("records"):
        assert row["precision"] == 0.0
        assert row["recall"] == 0.0
        assert row["false_positive_rate"] == 1.0


def test_group_metrics_handles_no_positives():
    """If a group has zero true positives, recall must be NaN (not 0/0)."""
    df = pd.DataFrame({"country": ["US"] * 4})
    y = np.zeros(4, dtype=int)
    yp = np.array([0, 1, 0, 0])

    out = group_metrics(df, "country", y, yp)
    assert math.isnan(out.iloc[0]["recall"])
    assert out.iloc[0]["false_positive_rate"] == 0.25


def test_group_metrics_handles_no_negatives():
    """If a group has zero negatives, FPR must be NaN."""
    df = pd.DataFrame({"country": ["US"] * 3})
    y = np.array([1, 1, 1])
    yp = np.array([1, 0, 1])

    out = group_metrics(df, "country", y, yp)
    assert math.isnan(out.iloc[0]["false_positive_rate"])
    assert out.iloc[0]["precision"] == 1.0


def test_group_metrics_multi_group():
    df = pd.DataFrame({"country": ["US", "US", "UK", "UK"]})
    y = np.array([1, 0, 1, 0])
    yp = np.array([1, 0, 0, 1])  # US perfect, UK swapped

    out = group_metrics(df, "country", y, yp).set_index("slice_value")
    assert out.loc["US"]["precision"] == 1.0
    assert out.loc["US"]["recall"] == 1.0
    assert out.loc["UK"]["precision"] == 0.0
    assert out.loc["UK"]["recall"] == 0.0


# --------------------- worst_disparate_impact + is_fair ---------------------

def _slice(dim, val, sel, n=100):
    """Build a fairness-table row with a default n that clears the min-n filter."""
    return {"slice_dimension": dim, "slice_value": val, "selection_rate": sel, "n": n}


def test_worst_disparate_impact_balanced():
    fdf = pd.DataFrame([
        _slice("country", "US", 0.10),
        _slice("country", "UK", 0.10),
    ])
    assert worst_disparate_impact(fdf) == pytest.approx(1.0)


def test_worst_disparate_impact_skewed():
    fdf = pd.DataFrame([
        _slice("country", "US", 0.20),
        _slice("country", "UK", 0.05),
    ])
    assert worst_disparate_impact(fdf) == pytest.approx(4.0)


def test_worst_disparate_impact_picks_worst_across_dimensions():
    fdf = pd.DataFrame([
        _slice("country", "US", 0.10),
        _slice("country", "UK", 0.10),
        _slice("age_group", "young", 0.20),
        _slice("age_group", "old", 0.04),
    ])
    # country ratio = 1.0, age_group ratio = 5.0 → worst is 5.0
    assert worst_disparate_impact(fdf) == pytest.approx(5.0)


def test_worst_disparate_impact_zero_min_returns_nan_safely():
    fdf = pd.DataFrame([
        _slice("country", "US", 0.10),
        _slice("country", "UK", 0.00),
    ])
    # min is 0 → ratio would be inf; dimension is skipped, no other dims,
    # so worst is NaN.
    assert math.isnan(worst_disparate_impact(fdf))


def test_worst_disparate_impact_empty():
    assert math.isnan(worst_disparate_impact(pd.DataFrame()))


def test_min_n_filter_drops_noisy_small_slices():
    """A skewed slice with n < min_n must not count toward the gate."""
    fdf = pd.DataFrame([
        _slice("country", "US", 0.10, n=500),
        _slice("country", "UK", 0.09, n=500),
        # Small noisy slice that would otherwise dominate the ratio:
        _slice("country", "LKA", 0.50, n=2),
    ])
    # Without the filter, ratio would be 0.50/0.09 = 5.5. With min_n=30 the
    # tiny LKA slice is excluded and we're left with 0.10/0.09 = 1.11.
    assert worst_disparate_impact(fdf, min_n=30) == pytest.approx(0.10 / 0.09)


def test_is_fair_under_threshold():
    fdf = pd.DataFrame([
        _slice("country", "US", 0.10),
        _slice("country", "UK", 0.09),
    ])
    # Use raw-DI gate (no positive_rate column needed).
    passes, worst, detail = is_fair(fdf, threshold=1.25, use_calibration=False)
    assert passes is True
    assert worst == pytest.approx(0.10 / 0.09)
    assert detail["dimension"] == "country"
    assert detail["max_slice"] == "US"
    assert detail["min_slice"] == "UK"


def test_is_fair_over_threshold():
    fdf = pd.DataFrame([
        _slice("country", "US", 0.20),
        _slice("country", "UK", 0.05),
    ])
    passes, worst, detail = is_fair(fdf, threshold=1.25, use_calibration=False)
    assert passes is False
    assert worst == pytest.approx(4.0)
    assert detail["max_selection_rate"] == pytest.approx(0.20)
    assert detail["min_selection_rate"] == pytest.approx(0.05)


def test_is_fair_empty_is_vacuously_true():
    """No protected slices in the sample → no bias claim either way."""
    passes, worst, detail = is_fair(pd.DataFrame())
    assert passes is True
    assert math.isnan(worst)
    assert detail is None


def test_calibration_passes_for_well_calibrated_fraud_model():
    """Model selects each subgroup proportional to its base fraud rate."""
    fdf = pd.DataFrame([
        {"slice_dimension": "age_group", "slice_value": "9", "selection_rate": 0.045,
         "positive_rate": 0.040, "n": 199},
        {"slice_dimension": "age_group", "slice_value": "3", "selection_rate": 0.014,
         "positive_rate": 0.014, "n": 217},
    ])
    # calibrations: 1.125, 1.000 → ratio = 1.125, raw DI = 3.21.
    # Calibration gate (default) should PASS; raw-DI gate should FAIL.
    passes_cal, worst_cal, _ = is_fair(fdf, threshold=1.5, use_calibration=True)
    passes_raw, worst_raw, _ = is_fair(fdf, threshold=1.5, use_calibration=False)
    assert passes_cal is True
    assert worst_cal == pytest.approx(1.125, rel=1e-3)
    assert passes_raw is False
    assert worst_raw == pytest.approx(3.21, rel=1e-2)


def test_calibration_fails_for_over_selecting_subgroup():
    """Model selects subgroup X far above its base rate → real bias."""
    fdf = pd.DataFrame([
        {"slice_dimension": "age_group", "slice_value": "X", "selection_rate": 0.10,
         "positive_rate": 0.02, "n": 200},
        {"slice_dimension": "age_group", "slice_value": "Y", "selection_rate": 0.02,
         "positive_rate": 0.02, "n": 200},
    ])
    # calibrations: 5.0, 1.0 → ratio = 5.0 — model over-flags X
    passes, worst, detail = is_fair(fdf, threshold=1.5, use_calibration=True)
    assert passes is False
    assert worst == pytest.approx(5.0)
    assert detail["max_slice"] == "X"


def test_is_fair_all_slices_below_min_n_is_vacuously_true():
    """If every slice is below min_n, gate must default to True (no signal)."""
    fdf = pd.DataFrame([
        _slice("country", "US", 0.20, n=5),
        _slice("country", "UK", 0.05, n=5),
    ])
    passes, worst, detail = is_fair(fdf, threshold=1.25, min_n=30, use_calibration=False)
    assert passes is True
    assert math.isnan(worst)
    assert detail is None


# ---------------------- select_protected_columns ----------------------

def test_select_protected_columns_keeps_present_in_order():
    cols = ["amount", "country", "age_group", "type"]
    assert select_protected_columns(cols) == ["country", "age_group"]


def test_select_protected_columns_returns_empty_when_none_present():
    assert select_protected_columns(["amount", "type"]) == []


def test_select_protected_columns_with_custom_candidates():
    cols = ["region", "amount"]
    assert select_protected_columns(cols, candidates=("region",)) == ["region"]
