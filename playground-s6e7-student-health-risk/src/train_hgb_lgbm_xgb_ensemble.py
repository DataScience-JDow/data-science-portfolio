from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
SUBMISSIONS_DIR = PROJECT_DIR / "submissions"

HGB_WEIGHT = 0.40
LGBM_WEIGHT = 0.29
XGB_WEIGHT = 0.31


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

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", HistGradientBoostingClassifier(random_state=42)),
        ]
    )


def prepare_native_tree_features(
    df: pd.DataFrame,
    categorical_features: list[str],
) -> pd.DataFrame:
    df = df.copy()
    for col in categorical_features:
        df[col] = df[col].astype("category")
    return df


def align_probabilities(
    raw_probabilities: np.ndarray,
    model_classes: list[str],
    label_classes: np.ndarray,
) -> np.ndarray:
    aligned_probabilities = np.zeros_like(raw_probabilities)
    for class_index, class_label in enumerate(label_classes):
        aligned_probabilities[:, class_index] = raw_probabilities[
            :, model_classes.index(class_label)
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

    label_encoder = LabelEncoder()
    encoded_y = label_encoder.fit_transform(y)
    label_classes = label_encoder.classes_

    hgb_pipeline = build_hgb_pipeline(X)
    hgb_sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    hgb_pipeline.fit(X, y, model__sample_weight=hgb_sample_weight)
    hgb_probabilities = align_probabilities(
        hgb_pipeline.predict_proba(X_test),
        hgb_pipeline.named_steps["model"].classes_.tolist(),
        label_classes,
    )

    native_tree_X = prepare_native_tree_features(X, categorical_features)
    native_tree_X_test = prepare_native_tree_features(X_test, categorical_features)
    tree_sample_weight = compute_sample_weight(class_weight="balanced", y=encoded_y)

    lgbm_model = LGBMClassifier(
        objective="multiclass",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=100,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    lgbm_model.fit(
        native_tree_X,
        encoded_y,
        sample_weight=tree_sample_weight,
    )
    lgbm_probabilities = lgbm_model.predict_proba(native_tree_X_test)

    xgb_model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_classes),
        tree_method="hist",
        enable_categorical=True,
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=20,
        reg_lambda=5.0,
        subsample=1.0,
        colsample_bytree=1.0,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(
        native_tree_X,
        encoded_y,
        sample_weight=tree_sample_weight,
    )
    xgb_probabilities = xgb_model.predict_proba(native_tree_X_test)

    blended_probabilities = (
        HGB_WEIGHT * hgb_probabilities
        + LGBM_WEIGHT * lgbm_probabilities
        + XGB_WEIGHT * xgb_probabilities
    )
    test_predictions = label_encoder.inverse_transform(
        blended_probabilities.argmax(axis=1)
    )

    submission_column = sample_submission_df.columns[-1]
    sample_submission_df[submission_column] = test_predictions

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SUBMISSIONS_DIR / "hgb_lgbm_xgb_ensemble_submission.csv"
    sample_submission_df.to_csv(output_path, index=False)

    prediction_distribution = sample_submission_df[submission_column].value_counts(
        normalize=True
    )
    print(f"Target column: {target_column}")
    print("Dropped feature: id")
    print("Added feature family: missingness indicator flags")
    print(f"HGB weight: {HGB_WEIGHT:.2f}")
    print(f"LightGBM weight: {LGBM_WEIGHT:.2f}")
    print(f"XGBoost weight: {XGB_WEIGHT:.2f}")
    print(f"Submission file written to: {output_path}")
    print("Prediction distribution:")
    print((prediction_distribution * 100).round(3).to_string())


if __name__ == "__main__":
    main()
