# CatalystLens Validation Backtests

The validation framework runs CatalystLens on point-in-time historical catalyst
examples and compares model probabilities with later outcomes.

Synthetic datasets are for plumbing tests only. They do not validate model
performance.

## Real Seed Dataset Commands

Place the cleaned real seed dataset at:

```bash
training/datasets/real_historical_biotech_panel_backtest_ready.csv
```

Then run:

```bash
python3 -m training.validation.run_backtest \
  --dataset training/datasets/real_historical_biotech_panel_backtest_ready.csv \
  --target financing_before_catalyst \
  --output-dir outputs/backtests/real_seed_financing
```

```bash
python3 -m training.validation.run_backtest \
  --dataset training/datasets/real_historical_biotech_panel_backtest_ready.csv \
  --target program_discontinued_before_catalyst \
  --output-dir outputs/backtests/real_seed_discontinuation
```

```bash
python3 -m training.validation.run_backtest \
  --dataset training/datasets/real_historical_biotech_panel_backtest_ready.csv \
  --target reached_catalyst_before_financing_pressure \
  --output-dir outputs/backtests/real_seed_reached_catalyst
```

Use language such as "preliminary seed backtest" and "exploratory calibration"
until source verification, time-based validation, and frozen model artifacts are
complete.
