# Playground Series S6E7: Predicting Student Health Risk

## Summary

This project explores a tabular multi-class classification problem from Kaggle's Playground Series. The workflow starts from a simple reproducible baseline, then iterates through metric-aligned training changes and targeted feature engineering to improve balanced accuracy on the public leaderboard.

## Competition Snapshot

- Competition: `Playground Series - Season 6, Episode 7`
- Problem type: tabular multi-class classification
- Metric: balanced accuracy
- Status: active
- Current best public leaderboard score: `0.95003`
- Kaggle page: <https://www.kaggle.com/competitions/playground-series-s6e7>

## Project Structure

```text
data/
  raw/
  interim/
  processed/
notebooks/
src/
reports/
figures/
submissions/
```

## Results Progression

| Version | Key change | Local CV balanced accuracy | Public LB |
| --- | --- | --- | --- |
| `baseline` | HistGradientBoosting with basic preprocessing | `0.85581` | `0.85188` |
| `v2` | Logistic regression with balanced class weights | `0.85657` | `0.85171` |
| `v3` | HistGradientBoosting with balanced sample weights and `id` removed | `0.90892` | `0.90503` |
| `v4` | `v3` plus missingness-indicator features | `0.94919` | `0.94997` |
| `v5` | `v4` blended with CatBoost probabilities | `0.94954` | `0.95003` |

## Modeling Approach

The current workflow uses:

- explicit train/test schema handling
- separate numeric and categorical preprocessing
- missing-value imputation
- one-hot encoding for categorical variables
- class-balanced sample weights during training
- missingness indicator features for columns with nulls
- probability blending between HistGradientBoosting and CatBoost
- stratified cross-validation scored with balanced accuracy

## Why The Best Version Improved

The biggest gain came from two changes:

- aligning the model fit to the competition metric by weighting minority classes more heavily
- treating missingness itself as information by adding binary flags such as `stress_level_missing` and `sleep_quality_missing`

In this dataset, class imbalance is severe and missing-value patterns carry predictive signal, so those changes mattered more than switching to a completely different model family.

## Reproducible Scripts

Key training scripts:

- `src/train_baseline.py`
- `src/train_logreg_balanced.py`
- `src/train_histgb_balanced_no_id.py`
- `src/train_histgb_balanced_missing_flags.py`
- `src/train_hgb_catboost_ensemble.py`
- `src/train_hgb_lgbm_xgb_ensemble.py`

EDA notebook:

- `notebooks/01_initial_eda.ipynb`

## Reproduction

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python playground-s6e7-student-health-risk/src/train_histgb_balanced_missing_flags.py
```

On macOS, LightGBM and XGBoost may also require the OpenMP runtime
(`libomp`) to be installed outside Python.

Expected local input files:

- `playground-s6e7-student-health-risk/data/raw/train.csv`
- `playground-s6e7-student-health-risk/data/raw/test.csv`
- `playground-s6e7-student-health-risk/data/raw/sample_submission.csv`

## Notes

- Raw Kaggle data is intentionally kept local and not committed to the repository.
- Progress notes and experiment history live in [reports/progress-log.md](reports/progress-log.md).
