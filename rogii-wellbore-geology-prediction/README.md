# ROGII: Wellbore Geology Prediction

## Summary

This project starts a baseline for Kaggle's ROGII wellbore geology prediction competition. The task is to predict missing `TVT` values for horizontal wells after the prediction-start point, using the known pre-start `TVT_input`, well path geometry, gamma ray measurements, and paired typewell data.

## Competition Snapshot

- Competition: `ROGII - Wellbore Geology Prediction`
- Problem type: regression on post-start horizontal-well TVT values
- Metric: RMSE
- Status: active
- Current best public leaderboard score: `14.304`
- Kaggle page: <https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction>

## Project Structure

```text
data/
  raw/
notebooks/
src/
reports/
figures/
submissions/
```

## Baseline

The first baseline is a last-known-TVT carry-forward model:

1. Find the last non-null `TVT_input` value before the prediction-start point.
2. Use that value as the prediction for every requested post-start row in the same well.
3. Validate the approach by scoring the natural post-start suffix in each training well, where `TVT_input` is missing but true `TVT` is available.

This deliberately simple baseline won the first local sanity check against local linear extrapolation and local MD/XYZ ridge regression. The more flexible local models extrapolated badly on many wells, while carry-forward made the smallest assumption: geology remains close to the last known TVT until we add stronger alignment logic.

## Current Result

| Version | Key change | Local validation RMSE | Public LB |
| --- | --- | --- | --- |
| `baseline_last_known_tvt` | Carry forward last known `TVT_input` per well | `15.90985` | `15.883` |
| `residual_correction_hgb` | Add blended HGB residual correction on top of baseline | `15.33328` | `14.304` |
| `typewell_gr_residual_hgb` | Add single-point typewell GR matching features to residual model | `15.28302` | `14.395` |
| `typewell_gr_window_experiment` | Compare short GR windows against rolling typewell windows | sampled `15.20325` | not submitted |
| `typewell_gr_sequence_residual_hgb` | Match normalized short GR sequences against typewell sequences | sampled `15.11805` | `14.596` |

## Reproduction

From the repo root:

```bash
source .venv/bin/activate
python rogii-wellbore-geology-prediction/src/train_baseline_last_known_tvt.py
```

Expected local input files:

- `rogii-wellbore-geology-prediction/data/raw/train/*__horizontal_well.csv`
- `rogii-wellbore-geology-prediction/data/raw/test/*__horizontal_well.csv`
- `rogii-wellbore-geology-prediction/data/raw/sample_submission.csv`

The script writes:

- `rogii-wellbore-geology-prediction/submissions/last_known_tvt_submission.csv`
- `rogii-wellbore-geology-prediction/reports/baseline-metrics.json`
- `rogii-wellbore-geology-prediction/submissions/residual_correction_submission.csv`
- `rogii-wellbore-geology-prediction/reports/residual-correction-metrics.json`
- `rogii-wellbore-geology-prediction/submissions/typewell_gr_residual_submission.csv`
- `rogii-wellbore-geology-prediction/reports/typewell-gr-residual-metrics.json`
- `rogii-wellbore-geology-prediction/submissions/typewell_gr_sequence_residual_submission.csv`
- `rogii-wellbore-geology-prediction/reports/typewell-gr-sequence-residual-metrics.json`

The code-competition submissions are packaged as Kaggle kernels:

- Baseline: `jdow76/rogii-last-known-tvt-baseline`
- Residual correction: `jdow76/rogii-residual-correction`
- Typewell GR residual correction: `jdow76/rogii-typewell-gr-residual`
- Typewell GR sequence residual correction: `jdow76/rogii-sequence-gr-residual`

## Notes

- Raw Kaggle data and generated submission CSVs are intentionally kept local and excluded from Git.
- The `train/` files include true `TVT` for post-start suffixes, so they are useful for validation. The baseline submission itself uses only the visible test fields and does not copy target values from duplicated train files.
