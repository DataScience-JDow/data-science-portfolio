from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
SUBMISSIONS_DIR = PROJECT_DIR / "submissions"

HGB_WEIGHT = 0.52
CATBOOST_WEIGHT = 0.48

CATBOOST_PARAMS = {
    "loss_function": "MultiClass",
    "iterations": 1200,
    "learning_rate": 0.06,
    "depth": 5,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "verbose": False,
    "thread_count": -1,
    "allow_writing_files": False,
}


def infer_target_column(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    candidate_columns = [col for col in train_df.columns if col not in test_df.columns]
    if len(candidate_columns) != 1:
        raise ValueError(
            "Could not infer the target column automatically. "
            f"Expected exactly one train-only column, found: {candidate_columns}"
        )
    return candidate_columns[0]


def add_missing_indicators(
    df: pd.DataFrame,
    feature_columns_with_missingness: list[str],
) -> pd.DataFrame:
    df = df.copy()
    for col in feature_columns_with_missingness:
        df[f"{col}_missing"] = df[col].isna().astype("int8")
    return df


def build_hgb_pipeline(X: pd.DataFrame) -> Pipeline:
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


def prepare_catboost_features(
    df: pd.DataFrame,
    categorical_features: list[str],
) -> pd.DataFrame:
    df = df.copy()
    for col in categorical_features:
        df[col] = df[col].fillna("missing").astype(str)
    return df


def align_hgb_probabilities(
    raw_probabilities: np.ndarray,
    hgb_classes: list[str],
    label_classes: np.ndarray,
) -> np.ndarray:
    aligned_probabilities = np.zeros_like(raw_probabilities)
    for class_index, class_label in enumerate(label_classes):
        aligned_probabilities[:, class_index] = raw_probabilities[
            :, hgb_classes.index(class_label)
        ]
    return aligned_probabilities


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
    base_feature_columns = [
        col for col in train_df.columns if col not in {target_column, "id"}
    ]
    categorical_features = (
        train_df[base_feature_columns].select_dtypes(include=["object"]).columns.tolist()
    )
    feature_columns_with_missingness = [
        col for col in base_feature_columns if train_df[col].isna().any()
    ]

    train_features = train_df[base_feature_columns].copy()
    test_features = test_df[base_feature_columns].copy()
    X = add_missing_indicators(train_features, feature_columns_with_missingness)
    X_test = add_missing_indicators(test_features, feature_columns_with_missingness)
    y = train_df[target_column].copy()

    hgb_pipeline = build_hgb_pipeline(X)
    hgb_sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    hgb_pipeline.fit(X, y, model__sample_weight=hgb_sample_weight)
    hgb_probabilities = align_hgb_probabilities(
        hgb_pipeline.predict_proba(X_test),
        hgb_pipeline.named_steps["model"].classes_.tolist(),
        np.array(sorted(y.unique())),
    )

    label_encoder = LabelEncoder()
    encoded_y = label_encoder.fit_transform(y)
    catboost_train = prepare_catboost_features(X, categorical_features)
    catboost_test = prepare_catboost_features(X_test, categorical_features)
    catboost_feature_indices = [
        catboost_train.columns.get_loc(col) for col in categorical_features
    ]
    catboost_sample_weight = compute_sample_weight(
        class_weight="balanced",
        y=encoded_y,
    )
    catboost_model = CatBoostClassifier(**CATBOOST_PARAMS)
    catboost_model.fit(
        Pool(
            catboost_train,
            encoded_y,
            cat_features=catboost_feature_indices,
            weight=catboost_sample_weight,
        )
    )
    catboost_probabilities = catboost_model.predict_proba(
        Pool(catboost_test, cat_features=catboost_feature_indices)
    )

    blended_probabilities = (
        HGB_WEIGHT * hgb_probabilities + CATBOOST_WEIGHT * catboost_probabilities
    )
    test_predictions = label_encoder.inverse_transform(
        blended_probabilities.argmax(axis=1)
    )

    submission_column = sample_submission_df.columns[-1]
    sample_submission_df[submission_column] = test_predictions

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SUBMISSIONS_DIR / "hgb_catboost_ensemble_submission.csv"
    sample_submission_df.to_csv(output_path, index=False)

    prediction_distribution = sample_submission_df[submission_column].value_counts(
        normalize=True
    )
    print(f"Target column: {target_column}")
    print("Dropped feature: id")
    print("Added feature family: missingness indicator flags")
    print(f"HGB weight: {HGB_WEIGHT:.2f}")
    print(f"CatBoost weight: {CATBOOST_WEIGHT:.2f}")
    print(f"Submission file written to: {output_path}")
    print("Prediction distribution:")
    print((prediction_distribution * 100).round(3).to_string())


if __name__ == "__main__":
    main()
