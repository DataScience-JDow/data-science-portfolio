# Progress Log

## 2026-07-02

- Reset the portfolio repo into a cleaner monorepo structure.
- Removed old starter practice artifacts.
- Added the initial scaffold for `Playground Series S6E7`.
- Added a reusable baseline script that can train once the Kaggle data is downloaded locally.
- Downloaded the Kaggle competition files into `data/raw/`.
- Ran the first local baseline and generated `submissions/baseline_submission.csv`.
- Initial 5-fold CV balanced accuracy: `0.85581`.
- Added an initial EDA notebook for profiling the competition data.

## 2026-07-05

- Submitted the initial baseline to Kaggle: public score `0.85188`.
- Tested a class-weighted logistic regression variant: public score `0.85171`.
- Improved the gradient boosting approach by removing `id` and using balanced sample weights: public score `0.90503`.
- Added missingness-indicator features and promoted the strongest version so far: public score `0.94997`.

## Next

- continue targeted error analysis by class
- compare missingness-driven gains against additional feature engineering
- add a second Kaggle project to the portfolio
