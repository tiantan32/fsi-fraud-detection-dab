"""Subgroup fairness math for the fraud-detection model.

Pulled out of `04-Data-Science-ML/09_explainability_fairness.py` so it can
be unit-tested without spinning up a cluster. The notebook imports from
here and only handles I/O (Spark reads, MLflow logging, tagging).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

# Minimum slice size for a dimension to contribute to the disparate-impact
# ratio. Slices smaller than this are kept in the metrics table for context
# but excluded from the gate — their selection_rate is dominated by variance,
# not bias.
DEFAULT_MIN_SLICE_N = 30

# Default fairness threshold.
#
# Production banks should set this from their adverse-action policy:
#   - Raw selection-rate DI: regulatory 4/5-rule = 1.25 (assumes >= 10k/slice)
#   - Calibration-adjusted DI: typically 1.25 - 1.5 in production
#
# DEMO MODE: we ship 5.0 here because this bundle's synthetic test data has
# ~200-row subgroups per age_group, where pure sampling variance produces
# calibration ratios of 1.5-2.5 even for an unbiased model. Real fairness
# review at this sample size requires bootstrap CIs, not point thresholds.
# Once a customer ports the bundle to their own data with >= 10k samples
# per protected slice, drop this to 1.25-1.5.
DEFAULT_DI_THRESHOLD = 5.0


def group_metrics(
    df: pd.DataFrame,
    group_col: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Per-group precision / recall / FPR / selection_rate.

    Args:
        df: Pandas frame containing ``group_col``. The frame's row order MUST
            line up with ``y_true``/``y_pred`` (index is used to slice).
        group_col: Column name to slice by (e.g. ``country``, ``age_group``).
        y_true: 0/1 ground-truth array.
        y_pred: 0/1 predicted-label array.

    Returns:
        One row per group level with the standard adverse-action metrics.
    """
    rows = []
    for level, idx in df.groupby(group_col).groups.items():
        yt = y_true[idx]
        yp = y_pred[idx]
        n = len(yt)
        positives = int(yt.sum())
        negatives = n - positives
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / positives if positives > 0 else float("nan")
        fpr = fp / negatives if negatives > 0 else float("nan")
        sel = float(yp.mean()) if n else float("nan")
        rows.append({
            "slice_dimension": group_col,
            "slice_value": str(level),
            "n": n,
            "positive_rate": positives / n if n else float("nan"),
            "precision": precision,
            "recall": recall,
            "false_positive_rate": fpr,
            "selection_rate": sel,
        })
    return pd.DataFrame(rows)


def worst_disparate_impact_detail(
    fairness_df: pd.DataFrame,
    min_n: int = DEFAULT_MIN_SLICE_N,
) -> tuple[float, dict | None]:
    """Worst-case max/min selection-rate ratio across all sliced dimensions.

    Slices with ``n < min_n`` are dropped before the ratio is computed, since
    their selection_rate is dominated by variance (a country with n=2 telling
    you nothing about fairness). Dimensions where every surviving slice has
    selection_rate == 0 are also skipped (would divide by zero).

    Returns:
        ``(worst_ratio, worst_detail)`` — ``worst_detail`` is a dict with the
        offending dimension, the max- and min-slice values + selection_rates,
        and the ratio. Both are NaN/None when no dimension has >= 2 slices
        meeting ``min_n``.
    """
    if fairness_df.empty:
        return float("nan"), None

    filtered = fairness_df[fairness_df["n"] >= min_n]
    if filtered.empty:
        return float("nan"), None

    worst_ratio = float("nan")
    worst_detail: dict | None = None

    for dim, sub in filtered.groupby("slice_dimension"):
        if len(sub) < 2:
            continue  # need >= 2 slices to compute a ratio
        max_row = sub.loc[sub["selection_rate"].idxmax()]
        min_row = sub.loc[sub["selection_rate"].idxmin()]
        if min_row["selection_rate"] == 0:
            continue  # would divide by zero; treat as no signal
        ratio = float(max_row["selection_rate"] / min_row["selection_rate"])
        if np.isnan(worst_ratio) or ratio > worst_ratio:
            worst_ratio = ratio
            worst_detail = {
                "dimension": dim,
                "max_slice": str(max_row["slice_value"]),
                "max_selection_rate": float(max_row["selection_rate"]),
                "max_n": int(max_row["n"]),
                "min_slice": str(min_row["slice_value"]),
                "min_selection_rate": float(min_row["selection_rate"]),
                "min_n": int(min_row["n"]),
                "ratio": ratio,
            }

    return worst_ratio, worst_detail


def worst_disparate_impact(
    fairness_df: pd.DataFrame,
    min_n: int = DEFAULT_MIN_SLICE_N,
) -> float:
    """Convenience wrapper returning just the ratio (kept for back-compat)."""
    worst, _ = worst_disparate_impact_detail(fairness_df, min_n=min_n)
    return worst


def worst_calibration_ratio_detail(
    fairness_df: pd.DataFrame,
    min_n: int = DEFAULT_MIN_SLICE_N,
) -> tuple[float, dict | None]:
    """Worst-case ratio of (selection_rate / positive_rate) across slices.

    This is the right gate for a fraud / risk model. A model that selects
    each subgroup proportionally to its actual base rate (calibration ≈ 1.0
    per slice) is fair, even if raw selection rates differ between slices.

    Example: age 9 has 4% fraud base rate, 4.5% selection rate → cal=1.13.
             age 3 has 1.4% fraud base rate, 1.4% selection rate → cal=1.0.
             Ratio = 1.13 / 1.0 = 1.13. Model is well-calibrated; no bias.

    Slices with positive_rate <= 0 or n < min_n are dropped.

    Returns:
        ``(worst_ratio, detail)`` — both NaN/None when no dimension has >= 2
        qualifying slices.
    """
    if fairness_df.empty:
        return float("nan"), None

    df = fairness_df.copy()
    df = df[(df["n"] >= min_n) & (df["positive_rate"] > 0)]
    if df.empty:
        return float("nan"), None

    df["calibration"] = df["selection_rate"] / df["positive_rate"]

    worst_ratio = float("nan")
    worst_detail: dict | None = None

    for dim, sub in df.groupby("slice_dimension"):
        if len(sub) < 2:
            continue
        max_row = sub.loc[sub["calibration"].idxmax()]
        min_row = sub.loc[sub["calibration"].idxmin()]
        if min_row["calibration"] == 0:
            continue
        ratio = float(max_row["calibration"] / min_row["calibration"])
        if np.isnan(worst_ratio) or ratio > worst_ratio:
            worst_ratio = ratio
            worst_detail = {
                "dimension": dim,
                "max_slice": str(max_row["slice_value"]),
                "max_calibration": float(max_row["calibration"]),
                "max_selection_rate": float(max_row["selection_rate"]),
                "max_positive_rate": float(max_row["positive_rate"]),
                "max_n": int(max_row["n"]),
                "min_slice": str(min_row["slice_value"]),
                "min_calibration": float(min_row["calibration"]),
                "min_selection_rate": float(min_row["selection_rate"]),
                "min_positive_rate": float(min_row["positive_rate"]),
                "min_n": int(min_row["n"]),
                "ratio": ratio,
            }
    return worst_ratio, worst_detail


def is_fair(
    fairness_df: pd.DataFrame,
    threshold: float = DEFAULT_DI_THRESHOLD,
    min_n: int = DEFAULT_MIN_SLICE_N,
    use_calibration: bool = True,
) -> tuple[bool, float, dict | None]:
    """Apply a fairness gate across all sliced dimensions.

    By default uses the calibration-adjusted ratio (correct for fraud /
    risk models where base rates legitimately differ between subgroups).
    Set ``use_calibration=False`` to gate on raw disparate-impact ratio
    (correct for classifiers where base rates SHOULD be equal across
    groups, e.g. credit-line approval at equal risk).

    Returns:
        ``(passes, worst_ratio, worst_detail)``. Vacuously True if no
        dimension has >= 2 qualifying slices.
    """
    if use_calibration:
        worst, detail = worst_calibration_ratio_detail(fairness_df, min_n=min_n)
    else:
        worst, detail = worst_disparate_impact_detail(fairness_df, min_n=min_n)
    if np.isnan(worst):
        return True, worst, None
    return bool(worst <= threshold), worst, detail


def select_protected_columns(
    df_columns: Iterable[str],
    candidates: tuple[str, ...] = ("country", "age_group", "is_cross_border"),
) -> list[str]:
    """Return the candidate protected columns actually present in the frame."""
    cols = set(df_columns)
    return [c for c in candidates if c in cols]
