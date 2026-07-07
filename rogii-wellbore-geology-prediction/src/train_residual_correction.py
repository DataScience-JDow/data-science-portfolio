from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
TRAIN_DIR = RAW_DIR / "train"
TEST_DIR = RAW_DIR / "test"
SUBMISSION_TEMPLATE = RAW_DIR / "sample_submission.csv"
SUBMISSIONS_DIR = PROJECT_DIR / "submissions"
REPORTS_DIR = PROJECT_DIR / "reports"

TRAIN_SAMPLE_PER_WELL = 1400
RESIDUAL_BLEND = 0.60
RESIDUAL_CLIP = 60.0
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "steps_after_ps",
    "target_frac",
    "md_delta",
    "x_delta",
    "y_delta",
    "z_delta",
    "md_from_start",
    "z_from_start",
    "last_known_tvt",
    "known_rows",
    "target_rows",
    "prefix_tvt_slope_50",
    "prefix_tvt_slope_200",
    "prefix_z_slope_200",
    "gr",
    "gr_missing",
    "gr_delta_last",
    "prefix_gr_mean_200",
    "prefix_gr_std_200",
]


def split_submission_id(value: str) -> tuple[str, int]:
    well_id, row_index = value.rsplit("_", 1)
    return well_id, int(row_index)


def linear_slope(x_values: np.ndarray, y_values: np.ndarray) -> float:
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if mask.sum() < 2:
        return 0.0

    x = x_values[mask]
    y = y_values[mask]
    centered_x = x - x.mean()
    denominator = float(np.dot(centered_x, centered_x))
    if denominator == 0.0:
        return 0.0

    return float(np.dot(centered_x, y - y.mean()) / denominator)


def well_features(path: Path, include_target: bool) -> pd.DataFrame:
    usecols = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
    if include_target:
        usecols.append("TVT")

    well_id = path.name.split("__", 1)[0]
    frame = pd.read_csv(path, usecols=usecols)
    target_mask = frame["TVT_input"].isna().to_numpy()
    if not target_mask.any():
        raise ValueError(f"No post-start target rows found for {well_id}.")

    known_indices = np.where(~target_mask)[0]
    target_indices = np.where(target_mask)[0]
    if len(known_indices) == 0:
        raise ValueError(f"No known TVT_input rows found for {well_id}.")

    last_known_index = known_indices[-1]
    prefix = frame.iloc[known_indices]
    suffix = frame.iloc[target_indices]
    last_known = frame.iloc[last_known_index]

    prefix_gr = prefix["GR"].dropna()
    last_known_gr = float(prefix_gr.iloc[-1]) if len(prefix_gr) else np.nan
    target_steps = target_indices - last_known_index
    target_count = len(target_indices)

    constants = {
        "last_known_tvt": float(last_known["TVT_input"]),
        "known_rows": float(len(known_indices)),
        "target_rows": float(target_count),
        "prefix_tvt_slope_50": linear_slope(
            prefix["MD"].tail(50).to_numpy(dtype=float),
            prefix["TVT_input"].tail(50).to_numpy(dtype=float),
        ),
        "prefix_tvt_slope_200": linear_slope(
            prefix["MD"].tail(200).to_numpy(dtype=float),
            prefix["TVT_input"].tail(200).to_numpy(dtype=float),
        ),
        "prefix_z_slope_200": linear_slope(
            prefix["MD"].tail(200).to_numpy(dtype=float),
            prefix["Z"].tail(200).to_numpy(dtype=float),
        ),
        "prefix_gr_mean_200": float(prefix_gr.tail(200).mean())
        if len(prefix_gr)
        else np.nan,
        "prefix_gr_std_200": float(prefix_gr.tail(200).std())
        if len(prefix_gr) > 1
        else 0.0,
    }

    features = pd.DataFrame(
        {
            "well_id": well_id,
            "row_index": target_indices,
            "steps_after_ps": target_steps.astype(float),
            "target_frac": target_steps / max(target_count, 1),
            "md_delta": suffix["MD"].to_numpy(dtype=float) - float(last_known["MD"]),
            "x_delta": suffix["X"].to_numpy(dtype=float) - float(last_known["X"]),
            "y_delta": suffix["Y"].to_numpy(dtype=float) - float(last_known["Y"]),
            "z_delta": suffix["Z"].to_numpy(dtype=float) - float(last_known["Z"]),
            "md_from_start": suffix["MD"].to_numpy(dtype=float)
            - float(frame["MD"].iloc[0]),
            "z_from_start": suffix["Z"].to_numpy(dtype=float)
            - float(frame["Z"].iloc[0]),
            "gr": suffix["GR"].to_numpy(dtype=float),
            "gr_missing": suffix["GR"].isna().astype(float).to_numpy(),
            "gr_delta_last": suffix["GR"].to_numpy(dtype=float) - last_known_gr,
            "baseline_tvt": constants["last_known_tvt"],
        }
    )
    for column, value in constants.items():
        features[column] = value

    if include_target:
        features["tvt"] = suffix["TVT"].to_numpy(dtype=float)
        features["residual"] = features["tvt"] - features["baseline_tvt"]

    for column in FEATURE_COLUMNS + ["baseline_tvt"]:
        features[column] = pd.to_numeric(features[column], downcast="float")
    if include_target:
        features["tvt"] = pd.to_numeric(features["tvt"], downcast="float")
        features["residual"] = pd.to_numeric(features["residual"], downcast="float")

    return features


def load_training_features() -> pd.DataFrame:
    frames = [
        well_features(path, include_target=True)
        for path in sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
    ]
    return pd.concat(frames, ignore_index=True)


def sample_training_rows(data: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    subset = data.iloc[indices]
    sampled_indices = []

    for _, local_indices in subset.groupby("well_id").indices.items():
        absolute_indices = subset.index.to_numpy()[local_indices]
        if len(absolute_indices) > TRAIN_SAMPLE_PER_WELL:
            positions = (
                np.linspace(0, len(absolute_indices) - 1, TRAIN_SAMPLE_PER_WELL)
                .round()
                .astype(int)
            )
            absolute_indices = absolute_indices[positions]
        sampled_indices.append(absolute_indices)

    return np.concatenate(sampled_indices)


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=260,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=4.0,
        random_state=RANDOM_STATE,
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def validate_residual_model(data: pd.DataFrame) -> dict:
    groups = data["well_id"].to_numpy()
    residual_oof = np.zeros(len(data), dtype=np.float32)
    fold_rows = []

    splitter = GroupKFold(n_splits=5)
    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(data[FEATURE_COLUMNS], data["residual"], groups), start=1
    ):
        sampled_train_idx = sample_training_rows(data, train_idx)
        model = make_model()
        model.fit(
            data.loc[sampled_train_idx, FEATURE_COLUMNS],
            data.loc[sampled_train_idx, "residual"],
        )

        raw_residual = model.predict(data.iloc[valid_idx][FEATURE_COLUMNS])
        residual_oof[valid_idx] = np.clip(raw_residual, -RESIDUAL_CLIP, RESIDUAL_CLIP)

        baseline_prediction = data.iloc[valid_idx]["baseline_tvt"].to_numpy()
        target = data.iloc[valid_idx]["tvt"].to_numpy()
        residual_prediction = baseline_prediction + residual_oof[valid_idx]
        blended_prediction = (
            baseline_prediction + RESIDUAL_BLEND * residual_oof[valid_idx]
        )

        fold_rows.append(
            {
                "fold": fold,
                "validation_rows": int(len(valid_idx)),
                "train_sample_rows": int(len(sampled_train_idx)),
                "baseline_rmse": rmse(target, baseline_prediction),
                "raw_residual_rmse": rmse(target, residual_prediction),
                "blended_residual_rmse": rmse(target, blended_prediction),
            }
        )

    baseline = data["baseline_tvt"].to_numpy()
    target = data["tvt"].to_numpy()
    alpha_rows = []
    for alpha in np.round(np.arange(0.0, 1.01, 0.1), 2):
        prediction = baseline + alpha * residual_oof
        alpha_rows.append({"alpha": float(alpha), "rmse": rmse(target, prediction)})

    final_prediction = baseline + RESIDUAL_BLEND * residual_oof
    return {
        "method": "residual_correction_hgb",
        "base_method": "baseline_last_known_tvt",
        "train_wells": int(data["well_id"].nunique()),
        "validation_rows": int(len(data)),
        "train_sample_per_well": TRAIN_SAMPLE_PER_WELL,
        "residual_blend": RESIDUAL_BLEND,
        "residual_clip": RESIDUAL_CLIP,
        "baseline_rmse": rmse(target, baseline),
        "residual_rmse": rmse(target, final_prediction),
        "absolute_rmse_improvement": rmse(target, baseline)
        - rmse(target, final_prediction),
        "relative_rmse_improvement_pct": (
            (rmse(target, baseline) - rmse(target, final_prediction))
            / rmse(target, baseline)
            * 100.0
        ),
        "alpha_sweep": alpha_rows,
        "folds": fold_rows,
    }


def train_final_model(data: pd.DataFrame) -> HistGradientBoostingRegressor:
    all_indices = np.arange(len(data))
    sampled_indices = sample_training_rows(data, all_indices)
    model = make_model()
    model.fit(data.loc[sampled_indices, FEATURE_COLUMNS], data.loc[sampled_indices, "residual"])
    return model


def build_submission(model: HistGradientBoostingRegressor) -> pd.DataFrame:
    submission = pd.read_csv(SUBMISSION_TEMPLATE)
    parsed = submission["id"].map(split_submission_id)
    submission["well_id"] = parsed.map(lambda value: value[0])
    submission["row_index"] = parsed.map(lambda value: value[1])

    prediction_frames = []
    for well_id in sorted(submission["well_id"].unique()):
        path = TEST_DIR / f"{well_id}__horizontal_well.csv"
        features = well_features(path, include_target=False)
        raw_residual = model.predict(features[FEATURE_COLUMNS])
        blended_residual = RESIDUAL_BLEND * np.clip(
            raw_residual, -RESIDUAL_CLIP, RESIDUAL_CLIP
        )
        features["tvt"] = features["baseline_tvt"] + blended_residual
        prediction_frames.append(features[["well_id", "row_index", "tvt"]])

    predictions = pd.concat(prediction_frames, ignore_index=True)
    output = submission.merge(predictions, on=["well_id", "row_index"], how="left")
    if output["tvt_y"].isna().any():
        missing = int(output["tvt_y"].isna().sum())
        raise ValueError(f"Submission has {missing} missing predictions.")

    output["tvt"] = output["tvt_y"].astype(float)
    return output[["id", "tvt"]]


def main() -> None:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    training_features = load_training_features()
    metrics = validate_residual_model(training_features)
    model = train_final_model(training_features)
    submission = build_submission(model)

    metrics_path = REPORTS_DIR / "residual-correction-metrics.json"
    submission_path = SUBMISSIONS_DIR / "residual_correction_submission.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    submission.to_csv(submission_path, index=False)

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote submission: {submission_path}")
    print(f"Baseline RMSE: {metrics['baseline_rmse']:.5f}")
    print(f"Residual correction RMSE: {metrics['residual_rmse']:.5f}")
    print(
        "Relative improvement: "
        f"{metrics['relative_rmse_improvement_pct']:.2f}%"
    )


if __name__ == "__main__":
    main()
