"""Markdown report generation for CatalystLens backtests."""

from __future__ import annotations

from collections import Counter

from training.validation.schemas import BacktestResult
from training.validation.target_mapping import TARGET_DEFINITIONS


def _fmt_pct(value: float | None) -> str:
    return "NA" if value is None else f"{value:.1%}"


def _ai_error_diagnosis_section(result: BacktestResult) -> str:
    diagnosed = [row for row in result.per_example_results if row.ai_method_status]
    if not diagnosed:
        return ""
    mode_counts = Counter(row.diagnosed_failure_mode or "other" for row in diagnosed)
    feature_counts = Counter(
        feature
        for row in diagnosed
        for feature in row.likely_missing_features
    )
    patch_counts = Counter(
        row.suggested_model_patch
        for row in diagnosed
        if row.suggested_model_patch
    )
    mode_rows = "\n".join(
        f"| {mode} | {count} |" for mode, count in mode_counts.most_common()
    )
    feature_rows = "\n".join(
        f"| {feature} | {count} |" for feature, count in feature_counts.most_common(8)
    ) or "| None | 0 |"
    patch_rows = "\n".join(
        f"- {patch} ({count})" for patch, count in patch_counts.most_common(5)
    ) or "- None"
    flag_names = [
        "false_negative_financing_event",
        "false_positive_financing_event",
        "false_negative_program_discontinuation",
        "false_positive_program_discontinuation",
        "scientific_failure_not_captured",
        "partnership_not_captured",
        "clean_refi_not_captured",
        "target_definition_ambiguous",
    ]
    flag_rows = "\n".join(
        f"| {flag} | {sum(1 for row in diagnosed if getattr(row, flag))} |"
        for flag in flag_names
    )
    return f"""## AI-Assisted Error Diagnosis

These diagnoses are heuristic AI-assisted diagnosis, not validated causal explanations. Source verification and human review are required before using them for model changes.

| Diagnosed failure mode | Count |
|---|---:|
{mode_rows}

| Suggested missing feature | Count |
|---|---:|
{feature_rows}

Suggested model improvements:

{patch_rows}

### Diagnostic Flag Counts

| Flag | Count |
|---|---:|
{flag_rows}
"""


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
    target = TARGET_DEFINITIONS[result.target_name]
    mapping_warning = (
        "\n> Probability mapping may use approximate fallback logic for older audit responses that do not expose "
        "explicit financing-state probabilities.\n"
        if target.approximate else ""
    )
    return f"""# CatalystLens Backtest Report
{synthetic_warning}
## 1. Dataset Summary

- Dataset ID: `{result.dataset_id}`
- Synthetic: `{result.synthetic}`
- Examples: `{result.n_examples}`
- Calibration status: `{result.calibration_status}`

## 2. Target Definition

Target: `{result.target_name}`

## Label Definition and Probability Mapping
{mapping_warning}
- Actual label definition: {target.positive_label_definition}
- Label description: {target.label_description}
- Model probability used: {target.probability_description}
- Probability source: {target.probability_source}
- Mapping exact: `{not target.approximate}`
- Fallback logic: {target.fallback_logic}
- Exact aggregate fields when exposed: `p_any_financing_event_before_catalyst` and `p_financing_pressure_before_catalyst`
- Reached-catalyst mapping subtracts financing pressure, not all financing events; clean/proactive financing is tracked separately from distress.

For synthetic datasets, this target is a pipeline smoke test only and is not evidence of real-world calibration.

## 3. Metrics

| Metric | Value |
|---|---:|
| N examples | {result.metric_summary.n_examples} |
| Event rate | {_fmt_pct(result.metric_summary.event_rate)} |
| Mean predicted probability | {_fmt_pct(result.metric_summary.mean_predicted_probability)} |
| Overprediction gap | {_fmt_pct(result.metric_summary.overprediction_gap)} |
| Underprediction gap | {_fmt_pct(result.metric_summary.underprediction_gap)} |
| Calibration direction | {result.metric_summary.calibration_direction} |
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

{_ai_error_diagnosis_section(result)}

## 7. Limitations

- This framework validates the pipeline machinery, not the model, unless run on real historical examples.
- Time-based external validation is required before institutional calibration claims.
- Current audit probabilities remain research-mode estimates unless backed by frozen artifacts and real labels.

## 8. Synthetic Data Warning

{warnings}
"""
