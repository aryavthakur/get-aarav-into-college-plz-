"""Markdown report generation for CatalystLens backtests."""

from __future__ import annotations

from training.validation.schemas import BacktestResult


def _fmt_pct(value: float | None) -> str:
    return "NA" if value is None else f"{value:.1%}"


def generate_backtest_report(result: BacktestResult) -> str:
    """Generate a concise Markdown report for a backtest result."""
    synthetic_warning = (
        "\n> This is synthetic test data and does not validate model performance.\n"
        if result.synthetic else ""
    )
    buckets = "\n".join(
        "| {:.0%}-{:.0%} | {} | {} | {} |".format(
            row.bucket_start,
            row.bucket_end,
            row.n_examples,
            _fmt_pct(row.mean_predicted_probability),
            _fmt_pct(row.observed_event_rate),
        )
        for row in result.metric_summary.calibration_buckets
    )
    cm = result.metric_summary.confusion_matrix
    warnings = "\n".join(f"- {w}" for w in result.warnings) or "- None"
    return f"""# CatalystLens Backtest Report
{synthetic_warning}
## 1. Dataset Summary

- Dataset ID: `{result.dataset_id}`
- Synthetic: `{result.synthetic}`
- Examples: `{result.n_examples}`
- Calibration status: `{result.calibration_status}`

## 2. Target Definition

Target: `{result.target_name}`

For synthetic datasets, this target is a pipeline smoke test only and is not evidence of real-world calibration.

## 3. Metrics

| Metric | Value |
|---|---:|
| N examples | {result.metric_summary.n_examples} |
| Event rate | {_fmt_pct(result.metric_summary.event_rate)} |
| Mean predicted probability | {_fmt_pct(result.metric_summary.mean_predicted_probability)} |
| Brier score | {result.metric_summary.brier_score:.4f} |
| Log loss | {result.metric_summary.log_loss:.4f} |
| ROC AUC | {result.metric_summary.roc_auc if result.metric_summary.roc_auc is not None else "NA"} |
| Expected calibration error | {result.metric_summary.expected_calibration_error:.4f} |

## 4. Calibration Table

| Predicted risk bucket | N | Mean predicted | Observed event rate |
|---|---:|---:|---:|
{buckets}

## 5. Observed vs Predicted

Confusion matrix at threshold 0.50:

| TP | FP | TN | FN |
|---:|---:|---:|---:|
| {cm["tp"]} | {cm["fp"]} | {cm["tn"]} | {cm["fn"]} |

## 6. Failure Modes

- Sparse buckets can make calibration unstable.
- Financing labels may combine clean, distressed, and nondilutive events unless the target is narrowed.
- Synthetic data can verify plumbing but cannot prove predictive performance.

## 7. Limitations

- This framework validates the pipeline machinery, not the model, unless run on real historical examples.
- Time-based external validation is required before institutional calibration claims.
- Current audit probabilities remain research-mode estimates unless backed by frozen artifacts and real labels.

## 8. Synthetic Data Warning

{warnings}
"""
