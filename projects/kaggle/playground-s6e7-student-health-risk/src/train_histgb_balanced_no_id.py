from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
SUBMISSIONS_DIR = PROJECT_DIR / "submissions"


def infer_target_column(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    candidate_columns = [col for col in train_df.columns if col not in test_df.columns]
    if len(candidate_columns) != 1:
        raise ValueError(
            "Could not infer the target column automatically. "
            f"Expected exactly one train-only column, found: {candidate_columns}"
        )
    return candidate_columns[0]


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    model = HistGradientBoostingClassifier(random_state=42)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main() -> None:
    train_path = RAW_DATA_DIR / "train.csv"
    test_path = RAW_DATA_DIR / "test.csv"
    sample_submission_path = RAW_DATA_DIR / "sample_submission.csv"

    missing_files = [
        path.name
        for path in [train_path, test_path, sample_submission_path]
        if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing Kaggle input files in data/raw: " + ", ".join(missing_files)
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_submission_df = pd.read_csv(sample_submission_path)

    target_column = infer_target_column(train_df, test_df)
    feature_columns = [
        col
        for col in train_df.columns
        if col not in {target_column, "id"}
    ]

    X = train_df[feature_columns].copy()
    y = train_df[target_column].copy()
    X_test = test_df[feature_columns].copy()

    pipeline = build_pipeline(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_scores: list[float] = []
    for fold_number, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)
        predictions = pipeline.predict(X_valid)
        score = balanced_accuracy_score(y_valid, predictions)
        fold_scores.append(score)
        print(f"Fold {fold_number} balanced accuracy: {score:.5f}")

    mean_score = sum(fold_scores) / len(fold_scores)
    print(f"Mean CV balanced accuracy: {mean_score:.5f}")

    full_sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    pipeline.fit(X, y, model__sample_weight=full_sample_weight)
    test_predictions = pipeline.predict(X_test)

    submission_column = sample_submission_df.columns[-1]
    sample_submission_df[submission_column] = test_predictions

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SUBMISSIONS_DIR / "histgb_balanced_no_id_submission.csv"
    sample_submission_df.to_csv(output_path, index=False)

    print(f"Target column: {target_column}")
    print("Dropped feature: id")
    print(f"Submission file written to: {output_path}")


if __name__ == "__main__":
    main()
