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

def test_worst_disparate_impact_balanced():
    fdf = pd.DataFrame([
        {"slice_dimension": "country", "slice_value": "US", "selection_rate": 0.10},
        {"slice_dimension": "country", "slice_value": "UK", "selection_rate": 0.10},
    ])
    assert worst_disparate_impact(fdf) == pytest.approx(1.0)


def test_worst_disparate_impact_skewed():
    fdf = pd.DataFrame([
        {"slice_dimension": "country", "slice_value": "US", "selection_rate": 0.20},
        {"slice_dimension": "country", "slice_value": "UK", "selection_rate": 0.05},
    ])
    assert worst_disparate_impact(fdf) == pytest.approx(4.0)


def test_worst_disparate_impact_picks_worst_across_dimensions():
    fdf = pd.DataFrame([
        {"slice_dimension": "country", "slice_value": "US", "selection_rate": 0.10},
        {"slice_dimension": "country", "slice_value": "UK", "selection_rate": 0.10},
        {"slice_dimension": "age_group", "slice_value": "young", "selection_rate": 0.20},
        {"slice_dimension": "age_group", "slice_value": "old", "selection_rate": 0.04},
    ])
    # country ratio = 1.0, age_group ratio = 5.0 → worst is 5.0
    assert worst_disparate_impact(fdf) == pytest.approx(5.0)


def test_worst_disparate_impact_zero_min_returns_nan_safely():
    fdf = pd.DataFrame([
        {"slice_dimension": "country", "slice_value": "US", "selection_rate": 0.10},
        {"slice_dimension": "country", "slice_value": "UK", "selection_rate": 0.00},
    ])
    # min is 0 → ratio would be inf; we replace with NaN. Single dim only,
    # so worst is NaN.
    assert math.isnan(worst_disparate_impact(fdf))


def test_worst_disparate_impact_empty():
    assert math.isnan(worst_disparate_impact(pd.DataFrame()))


def test_is_fair_under_threshold():
    fdf = pd.DataFrame([
        {"slice_dimension": "country", "slice_value": "US", "selection_rate": 0.10},
        {"slice_dimension": "country", "slice_value": "UK", "selection_rate": 0.09},
    ])
    passes, worst = is_fair(fdf, threshold=1.25)
    assert passes is True
    assert worst == pytest.approx(0.10 / 0.09)


def test_is_fair_over_threshold():
    fdf = pd.DataFrame([
        {"slice_dimension": "country", "slice_value": "US", "selection_rate": 0.20},
        {"slice_dimension": "country", "slice_value": "UK", "selection_rate": 0.05},
    ])
    passes, worst = is_fair(fdf, threshold=1.25)
    assert passes is False
    assert worst == pytest.approx(4.0)


def test_is_fair_empty_is_vacuously_true():
    """No protected slices in the sample → no bias claim either way."""
    passes, worst = is_fair(pd.DataFrame())
    assert passes is True
    assert math.isnan(worst)


# ---------------------- select_protected_columns ----------------------

def test_select_protected_columns_keeps_present_in_order():
    cols = ["amount", "country", "age_group", "type"]
    assert select_protected_columns(cols) == ["country", "age_group"]


def test_select_protected_columns_returns_empty_when_none_present():
    assert select_protected_columns(["amount", "type"]) == []


def test_select_protected_columns_with_custom_candidates():
    cols = ["region", "amount"]
    assert select_protected_columns(cols, candidates=("region",)) == ["region"]
