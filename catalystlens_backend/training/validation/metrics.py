"""Binary backtest and calibration metrics."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from training.validation.schemas import CalibrationBucket


def _arrays(y_true: Iterable[float], y_prob: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_prob), dtype=float)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_prob must have the same length")
    if len(yt) == 0:
        raise ValueError("metrics require at least one example")
    return yt, np.clip(yp, 0.0, 1.0)


def brier_score(y_true: Iterable[float], y_prob: Iterable[float]) -> float:
    yt, yp = _arrays(y_true, y_prob)
    return float(np.mean((yp - yt) ** 2))


def log_loss_binary(y_true: Iterable[float], y_prob: Iterable[float], eps: float = 1e-15) -> float:
    yt, yp = _arrays(y_true, y_prob)
    yp = np.clip(yp, eps, 1.0 - eps)
    return float(-np.mean(yt * np.log(yp) + (1.0 - yt) * np.log(1.0 - yp)))


def calibration_by_bucket(
    y_true: Iterable[float],
    y_prob: Iterable[float],
    buckets: list[float] | None = None,
) -> list[CalibrationBucket]:
    yt, yp = _arrays(y_true, y_prob)
    cuts = buckets or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rows: list[CalibrationBucket] = []
    for i, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:])):
        if i == len(cuts) - 2:
            mask = (yp >= lo) & (yp <= hi)
        else:
            mask = (yp >= lo) & (yp < hi)
        n = int(np.sum(mask))
        rows.append(CalibrationBucket(
            bucket_start=float(lo),
            bucket_end=float(hi),
            n_examples=n,
            mean_predicted_probability=float(np.mean(yp[mask])) if n else None,
            observed_event_rate=float(np.mean(yt[mask])) if n else None,
        ))
    return rows


def expected_calibration_error(
    y_true: Iterable[float],
    y_prob: Iterable[float],
    buckets: list[float] | None = None,
) -> float:
    yt, yp = _arrays(y_true, y_prob)
    rows = calibration_by_bucket(yt, yp, buckets)
    total = len(yt)
    ece = 0.0
    for row in rows:
        if row.n_examples and row.mean_predicted_probability is not None and row.observed_event_rate is not None:
            ece += row.n_examples / total * abs(row.mean_predicted_probability - row.observed_event_rate)
    return float(ece)


def observed_vs_predicted_table(
    y_true: Iterable[float],
    y_prob: Iterable[float],
    buckets: list[float] | None = None,
) -> list[CalibrationBucket]:
    return calibration_by_bucket(y_true, y_prob, buckets)


def roc_auc(y_true: Iterable[float], y_prob: Iterable[float]) -> float | None:
    yt, yp = _arrays(y_true, y_prob)
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return None
    if len(set(yt.tolist())) < 2:
        return None
    score = roc_auc_score(yt, yp)
    return None if math.isnan(score) else float(score)


def confusion_matrix_at_threshold(
    y_true: Iterable[float],
    y_prob: Iterable[float],
    threshold: float,
) -> dict[str, int]:
    yt, yp = _arrays(y_true, y_prob)
    pred = yp >= threshold
    truth = yt >= 0.5
    return {
        "tp": int(np.sum(pred & truth)),
        "fp": int(np.sum(pred & ~truth)),
        "tn": int(np.sum(~pred & ~truth)),
        "fn": int(np.sum(~pred & truth)),
    }


def calibration_diagnostics(
    mean_predicted_probability: float,
    observed_event_rate: float,
    tolerance: float = 0.05,
) -> dict[str, float | str]:
    gap = float(mean_predicted_probability - observed_event_rate)
    if gap > tolerance:
        direction = "overpredicting"
    elif gap < -tolerance:
        direction = "underpredicting"
    else:
        direction = "approximately_calibrated"
    return {
        "overprediction_gap": max(gap, 0.0),
        "underprediction_gap": max(-gap, 0.0),
        "calibration_direction": direction,
    }
