"""CLI entrypoint for CatalystLens historical backtests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from training.validation.backtest import load_historical_examples, run_backtest
from training.validation.backtest_report import generate_backtest_report


def _bucket(probability: float) -> str:
    cuts = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    for lo, hi in cuts:
        if lo <= probability < hi or (hi == 1.0 and probability <= hi):
            return f"{lo:.0%}-{hi:.0%}"
    return "outside_0_1"


def _row_target(row, target: str) -> tuple[int, float]:
    if target == "financing_before_catalyst":
        return int(row.actual_financing_before_catalyst), row.predicted_financing_before_catalyst
    if target == "distressed_financing_or_cashout":
        return int(row.actual_distressed_financing_or_cashout), row.predicted_distressed_or_cashout_before_catalyst
    if target == "program_discontinued_before_catalyst":
        return int(row.actual_program_discontinued_before_catalyst), row.predicted_program_discontinuation
    if target == "reached_catalyst_before_financing_pressure":
        return int(row.actual_reached_catalyst_before_financing_pressure), row.predicted_reaches_catalyst_before_financing_pressure
    if target == "clinical_success":
        return int(bool(row.actual_clinical_success)), row.posterior_mean_pos
    raise ValueError(f"Unsupported target: {target}")


def write_prediction_error_table(result, path: Path, diagnose_errors: bool = False) -> None:
    fieldnames = [
        "example_id",
        "company_name",
        "ticker",
        "target",
        "y_true",
        "y_prob",
        "absolute_error",
        "predicted_risk_bucket",
        "probability_mapping_note",
    ]
    if diagnose_errors:
        fieldnames.extend([
            "error_type",
            "diagnosed_failure_mode",
            "likely_missing_features",
            "suggested_model_patch",
            "ai_diagnosis_confidence",
            "ai_method_status",
            "partnerability_score",
            "proactive_financing_likelihood",
            "scientific_discontinuation_risk_score",
            "safety_sensitive_modality_score",
        ])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.per_example_results:
            y_true, y_prob = _row_target(row, result.target_name)
            out = {
                "example_id": row.example_id,
                "company_name": row.company_name,
                "ticker": row.ticker,
                "target": result.target_name,
                "y_true": y_true,
                "y_prob": f"{y_prob:.6f}",
                "absolute_error": f"{abs(y_prob - y_true):.6f}",
                "predicted_risk_bucket": _bucket(y_prob),
                "probability_mapping_note": row.probability_mapping_note,
            }
            if diagnose_errors:
                out.update({
                    "error_type": row.error_type or "",
                    "diagnosed_failure_mode": row.diagnosed_failure_mode or "",
                    "likely_missing_features": ";".join(row.likely_missing_features),
                    "suggested_model_patch": row.suggested_model_patch or "",
                    "ai_diagnosis_confidence": "" if row.ai_diagnosis_confidence is None else f"{row.ai_diagnosis_confidence:.6f}",
                    "ai_method_status": row.ai_method_status or "",
                    "partnerability_score": "" if row.partnerability_score is None else f"{row.partnerability_score:.6f}",
                    "proactive_financing_likelihood": "" if row.proactive_financing_likelihood is None else f"{row.proactive_financing_likelihood:.6f}",
                    "scientific_discontinuation_risk_score": "" if row.scientific_discontinuation_risk_score is None else f"{row.scientific_discontinuation_risk_score:.6f}",
                    "safety_sensitive_modality_score": "" if row.safety_sensitive_modality_score is None else f"{row.safety_sensitive_modality_score:.6f}",
                })
            writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CatalystLens backtest on a historical catalyst dataset.")
    parser.add_argument("--dataset", required=True, help="Path to historical dataset CSV.")
    parser.add_argument("--target", default="financing_before_catalyst", help="Binary target to score.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for JSON and Markdown outputs.")
    parser.add_argument("--diagnose-errors", action="store_true", help="Add heuristic AI-assisted error diagnosis.")
    args = parser.parse_args()

    dataset = load_historical_examples(args.dataset)
    result = run_backtest(dataset, target_name=args.target, diagnose_errors=args.diagnose_errors)
    report = generate_backtest_report(result)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "backtest_result.json"
    md_path = output_dir / "backtest_report.md"
    error_path = output_dir / "backtest_prediction_errors.csv"
    json_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    md_path.write_text(report)
    write_prediction_error_table(result, error_path, diagnose_errors=args.diagnose_errors)

    print("CatalystLens backtest complete")
    print(f"dataset_id: {result.dataset_id}")
    print(f"target: {result.target_name}")
    print(f"n_examples: {result.n_examples}")
    print(f"calibration_status: {result.calibration_status}")
    print(f"brier_score: {result.metric_summary.brier_score:.4f}")
    print(f"expected_calibration_error: {result.metric_summary.expected_calibration_error:.4f}")
    print(f"json: {json_path}")
    print(f"markdown: {md_path}")
    print(f"prediction_errors: {error_path}")


if __name__ == "__main__":
    main()
