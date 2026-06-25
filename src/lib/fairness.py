"""Subgroup fairness math for the fraud-detection model.

Pulled out of `04-Data-Science-ML/09_explainability_fairness.py` so it can
be unit-tested without spinning up a cluster. The notebook imports from
here and only handles I/O (Spark reads, MLflow logging, tagging).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


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


def worst_disparate_impact(fairness_df: pd.DataFrame) -> float:
    """Worst-case max/min selection-rate ratio across all sliced dimensions.

    Returns NaN if the input is empty or every dimension has a single level.
    """
    if fairness_df.empty:
        return float("nan")
    grouped = fairness_df.groupby("slice_dimension")["selection_rate"]
    # max/min per dimension; protect against /0 by replacing min==0 with NaN.
    ratios = grouped.max() / grouped.min().replace(0, np.nan)
    if ratios.empty or ratios.isna().all():
        return float("nan")
    return float(ratios.max())


def is_fair(
    fairness_df: pd.DataFrame,
    threshold: float = 1.25,
) -> tuple[bool, float]:
    """Apply a disparate-impact threshold across all sliced dimensions.

    The convention here matches the 4/5-style rule, but inverted: we cap
    the ratio of strictest-to-most-lenient selection rates. ``threshold``
    is a policy choice the bank's adverse-action review framework should
    set; ``1.25`` is the default we ship for fraud.

    Returns:
        ``(passes, worst_ratio)`` — ``passes`` is True when there are no
        protected dimensions in the sample (vacuously fair) or the worst
        ratio is at or below ``threshold``.
    """
    worst = worst_disparate_impact(fairness_df)
    if np.isnan(worst):
        return True, worst
    return bool(worst <= threshold), worst


def select_protected_columns(
    df_columns: Iterable[str],
    candidates: tuple[str, ...] = ("country", "age_group", "is_cross_border"),
) -> list[str]:
    """Return the candidate protected columns actually present in the frame."""
    cols = set(df_columns)
    return [c for c in candidates if c in cols]
