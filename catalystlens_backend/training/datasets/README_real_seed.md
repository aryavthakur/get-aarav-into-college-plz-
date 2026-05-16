# Real Seed Backtest Dataset

Place the source-verified real seed panel here when available:

```text
training/datasets/real_historical_biotech_panel_backtest_ready.csv
```

Do not commit unverified real company data or source notes. Synthetic examples in this repository are for pipeline tests only and do not validate model performance.

Run the financing-event seed backtest:

```bash
python -m training.validation.run_backtest \
  --dataset training/datasets/real_historical_biotech_panel_backtest_ready.csv \
  --target financing_before_catalyst \
  --output-dir outputs/backtests/real_seed_financing
```

Run the program-discontinuation seed backtest:

```bash
python -m training.validation.run_backtest \
  --dataset training/datasets/real_historical_biotech_panel_backtest_ready.csv \
  --target program_discontinued_before_catalyst \
  --output-dir outputs/backtests/real_seed_program_discontinuation
```

Run the reached-catalyst seed backtest:

```bash
python -m training.validation.run_backtest \
  --dataset training/datasets/real_historical_biotech_panel_backtest_ready.csv \
  --target reached_catalyst_before_financing_pressure \
  --output-dir outputs/backtests/real_seed_reached_catalyst
```

Label real-data outputs as a preliminary seed backtest until the dataset has source verification, time-based validation splits, and frozen model artifacts.
