from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
TRAIN_DIR = RAW_DIR / "train"
TEST_DIR = RAW_DIR / "test"
SUBMISSION_TEMPLATE = RAW_DIR / "sample_submission.csv"
SUBMISSIONS_DIR = PROJECT_DIR / "submissions"
REPORTS_DIR = PROJECT_DIR / "reports"

VALIDATION_SAMPLE_PER_WELL = 800
TRAIN_SAMPLE_PER_WELL = 700
SEQUENCE_WINDOW = 31
SEQUENCE_HALF_WINDOW = 15
MIN_SEQUENCE_VALUES = 14
TYPEWELL_TVT_WINDOW = 240.0
RESIDUAL_BLEND = 0.60
RESIDUAL_CLIP = 60.0
RANDOM_STATE = 42

BASE_FEATURE_COLUMNS = [
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

SEQUENCE_FEATURE_COLUMNS = [
    "seq_valid_frac",
    "seq_gr_mean",
    "seq_gr_std",
    "seq_match_tvt_delta",
    "seq_match_distance",
    "seq_match_mean_delta",
    "seq_match_std_delta",
    "seq_match_available",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + SEQUENCE_FEATURE_COLUMNS


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


def normalized_gr_window(values: np.ndarray) -> tuple[np.ndarray | None, float, float, float]:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    valid_fraction = float(valid.mean())
    if valid.sum() < MIN_SEQUENCE_VALUES:
        return None, valid_fraction, np.nan, np.nan

    positions = np.arange(len(values))
    filled = values.copy()
    filled[~valid] = np.interp(positions[~valid], positions[valid], values[valid])

    mean = float(filled.mean())
    std = float(filled.std())
    if std < 1e-6:
        return None, valid_fraction, mean, std

    return (filled - mean) / std, valid_fraction, mean, std


def typewell_sequence_windows(
    typewell_path: Path,
    baseline_tvt: float,
) -> tuple[np.ndarray | None, pd.DataFrame | None]:
    typewell = (
        pd.read_csv(typewell_path, usecols=["TVT", "GR"])
        .dropna()
        .sort_values("TVT")
    )
    typewell = typewell.groupby("TVT", as_index=False)["GR"].mean()
    tvt = typewell["TVT"].to_numpy(dtype=float)
    gr = typewell["GR"].to_numpy(dtype=float)
    centers = np.where(
        (tvt >= baseline_tvt - TYPEWELL_TVT_WINDOW)
        & (tvt <= baseline_tvt + TYPEWELL_TVT_WINDOW)
    )[0]

    vectors = []
    rows = []
    for center in centers:
        low = center - SEQUENCE_HALF_WINDOW
        high = center + SEQUENCE_HALF_WINDOW + 1
        if low < 0 or high > len(gr):
            continue

        vector, _, mean, std = normalized_gr_window(gr[low:high])
        if vector is None:
            continue

        vectors.append(vector)
        rows.append({"match_tvt": tvt[center], "match_mean": mean, "match_std": std})

    if not vectors:
        return None, None

    return np.vstack(vectors), pd.DataFrame(rows)


def sequence_match_features(
    frame: pd.DataFrame,
    target_indices: np.ndarray,
    typewell_path: Path,
    baseline_tvt: float,
) -> dict[str, np.ndarray]:
    typewell_vectors, typewell_meta = typewell_sequence_windows(typewell_path, baseline_tvt)
    row_count = len(target_indices)
    output = {
        "seq_valid_frac": np.zeros(row_count),
        "seq_gr_mean": np.full(row_count, np.nan),
        "seq_gr_std": np.full(row_count, np.nan),
        "seq_match_tvt_delta": np.full(row_count, np.nan),
        "seq_match_distance": np.full(row_count, np.nan),
        "seq_match_mean_delta": np.full(row_count, np.nan),
        "seq_match_std_delta": np.full(row_count, np.nan),
        "seq_match_available": np.zeros(row_count),
    }
    if typewell_vectors is None or typewell_meta is None:
        return output

    matcher = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(typewell_vectors)
    horizontal_gr = frame["GR"].to_numpy(dtype=float)
    query_vectors = []
    query_meta = []
    query_positions = []

    for position, row_index in enumerate(target_indices):
        low = row_index - SEQUENCE_HALF_WINDOW
        high = row_index + SEQUENCE_HALF_WINDOW + 1
        if low < 0 or high > len(horizontal_gr):
            continue

        vector, valid_fraction, mean, std = normalized_gr_window(horizontal_gr[low:high])
        output["seq_valid_frac"][position] = valid_fraction
        output["seq_gr_mean"][position] = mean
        output["seq_gr_std"][position] = std
        if vector is None:
            continue

        query_vectors.append(vector)
        query_meta.append((mean, std))
        query_positions.append(position)

    if not query_vectors:
        return output

    distances, indices = matcher.kneighbors(np.vstack(query_vectors))
    matched = typewell_meta.iloc[indices[:, 0]].reset_index(drop=True)
    query_positions = np.asarray(query_positions)
    query_mean = np.asarray([value[0] for value in query_meta])
    query_std = np.asarray([value[1] for value in query_meta])

    output["seq_match_tvt_delta"][query_positions] = (
        matched["match_tvt"].to_numpy(dtype=float) - baseline_tvt
    )
    output["seq_match_distance"][query_positions] = distances[:, 0]
    output["seq_match_mean_delta"][query_positions] = (
        query_mean - matched["match_mean"].to_numpy(dtype=float)
    )
    output["seq_match_std_delta"][query_positions] = (
        query_std - matched["match_std"].to_numpy(dtype=float)
    )
    output["seq_match_available"][query_positions] = 1.0
    return output


def sampled_target_indices(target_indices: np.ndarray) -> np.ndarray:
    if len(target_indices) <= VALIDATION_SAMPLE_PER_WELL:
        return target_indices
    positions = (
        np.linspace(0, len(target_indices) - 1, VALIDATION_SAMPLE_PER_WELL)
        .round()
        .astype(int)
    )
    return target_indices[positions]


def well_features(
    path: Path,
    include_target: bool,
    sample_targets: bool,
) -> pd.DataFrame:
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
    if sample_targets:
        target_indices = sampled_target_indices(target_indices)
    if len(known_indices) == 0:
        raise ValueError(f"No known TVT_input rows found for {well_id}.")

    last_known_index = known_indices[-1]
    prefix = frame.iloc[known_indices]
    suffix = frame.iloc[target_indices]
    last_known = frame.iloc[last_known_index]

    prefix_gr = prefix["GR"].dropna()
    last_known_gr = float(prefix_gr.iloc[-1]) if len(prefix_gr) else np.nan
    target_steps = target_indices - last_known_index
    full_target_count = int(target_mask.sum())
    baseline_tvt = float(last_known["TVT_input"])
    sequence_features = sequence_match_features(
        frame,
        target_indices,
        path.with_name(f"{well_id}__typewell.csv"),
        baseline_tvt,
    )

    features = pd.DataFrame(
        {
            "well_id": well_id,
            "row_index": target_indices,
            "steps_after_ps": target_steps.astype(float),
            "target_frac": target_steps / max(full_target_count, 1),
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
            "baseline_tvt": baseline_tvt,
            "last_known_tvt": baseline_tvt,
            "known_rows": float(len(known_indices)),
            "target_rows": float(full_target_count),
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
            **sequence_features,
        }
    )

    if include_target:
        features["tvt"] = suffix["TVT"].to_numpy(dtype=float)
        features["residual"] = features["tvt"] - features["baseline_tvt"]

    for column in FEATURE_COLUMNS + ["baseline_tvt"]:
        features[column] = pd.to_numeric(features[column], downcast="float")
    if include_target:
        features["tvt"] = pd.to_numeric(features["tvt"], downcast="float")
        features["residual"] = pd.to_numeric(features["residual"], downcast="float")

    return features


def load_training_features(sample_targets: bool) -> pd.DataFrame:
    frames = [
        well_features(path, include_target=True, sample_targets=sample_targets)
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
        max_iter=140,
        learning_rate=0.055,
        max_leaf_nodes=15,
        min_samples_leaf=120,
        l2_regularization=6.0,
        random_state=RANDOM_STATE,
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def grouped_oof_score(data: pd.DataFrame, feature_columns: list[str]) -> dict:
    groups = data["well_id"].to_numpy()
    residual_oof = np.zeros(len(data), dtype=np.float32)
    fold_rows = []

    splitter = GroupKFold(n_splits=5)
    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(data[feature_columns], data["residual"], groups), start=1
    ):
        sampled_train_idx = sample_training_rows(data, train_idx)
        model = make_model()
        model.fit(
            data.loc[sampled_train_idx, feature_columns],
            data.loc[sampled_train_idx, "residual"],
        )
        raw_residual = model.predict(data.iloc[valid_idx][feature_columns])
        residual_oof[valid_idx] = np.clip(raw_residual, -RESIDUAL_CLIP, RESIDUAL_CLIP)

        baseline_prediction = data.iloc[valid_idx]["baseline_tvt"].to_numpy()
        target = data.iloc[valid_idx]["tvt"].to_numpy()
        blended_prediction = (
            baseline_prediction + RESIDUAL_BLEND * residual_oof[valid_idx]
        )
        fold_rows.append(
            {
                "fold": fold,
                "validation_rows": int(len(valid_idx)),
                "train_sample_rows": int(len(sampled_train_idx)),
                "rmse": rmse(target, blended_prediction),
            }
        )

    baseline = data["baseline_tvt"].to_numpy()
    target = data["tvt"].to_numpy()
    final_prediction = baseline + RESIDUAL_BLEND * residual_oof
    return {"rmse": rmse(target, final_prediction), "folds": fold_rows}


def validate_sequence_model(data: pd.DataFrame) -> dict:
    base_score = grouped_oof_score(data, BASE_FEATURE_COLUMNS)
    sequence_score = grouped_oof_score(data, FEATURE_COLUMNS)
    sequence_available_rate = float(data["seq_match_available"].mean())

    return {
        "method": "typewell_gr_sequence_residual_hgb",
        "base_method": "residual_correction_hgb",
        "validation_type": "grouped_by_well_sampled_rows",
        "validation_rows": int(len(data)),
        "train_wells": int(data["well_id"].nunique()),
        "validation_sample_per_well": VALIDATION_SAMPLE_PER_WELL,
        "train_sample_per_well": TRAIN_SAMPLE_PER_WELL,
        "sequence_window": SEQUENCE_WINDOW,
        "min_sequence_values": MIN_SEQUENCE_VALUES,
        "typewell_tvt_window": TYPEWELL_TVT_WINDOW,
        "residual_blend": RESIDUAL_BLEND,
        "residual_clip": RESIDUAL_CLIP,
        "sequence_available_rate": sequence_available_rate,
        "base_sampled_rmse": base_score["rmse"],
        "sequence_sampled_rmse": sequence_score["rmse"],
        "absolute_rmse_improvement_vs_sampled_base": base_score["rmse"]
        - sequence_score["rmse"],
        "relative_rmse_improvement_pct_vs_sampled_base": (
            (base_score["rmse"] - sequence_score["rmse"])
            / base_score["rmse"]
            * 100.0
        ),
        "base_folds": base_score["folds"],
        "sequence_folds": sequence_score["folds"],
    }


def train_final_model(data: pd.DataFrame) -> HistGradientBoostingRegressor:
    all_indices = np.arange(len(data))
    sampled_indices = sample_training_rows(data, all_indices)
    model = make_model()
    model.fit(
        data.loc[sampled_indices, FEATURE_COLUMNS],
        data.loc[sampled_indices, "residual"],
    )
    return model


def build_submission(model: HistGradientBoostingRegressor) -> pd.DataFrame:
    submission = pd.read_csv(SUBMISSION_TEMPLATE)
    parsed = submission["id"].map(split_submission_id)
    submission["well_id"] = parsed.map(lambda value: value[0])
    submission["row_index"] = parsed.map(lambda value: value[1])

    prediction_frames = []
    for well_id in sorted(submission["well_id"].unique()):
        path = TEST_DIR / f"{well_id}__horizontal_well.csv"
        features = well_features(path, include_target=False, sample_targets=False)
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

    validation_features = load_training_features(sample_targets=True)
    metrics = validate_sequence_model(validation_features)
    final_model = train_final_model(validation_features)
    submission = build_submission(final_model)

    metrics_path = REPORTS_DIR / "typewell-gr-sequence-residual-metrics.json"
    submission_path = SUBMISSIONS_DIR / "typewell_gr_sequence_residual_submission.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    submission.to_csv(submission_path, index=False)

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote submission: {submission_path}")
    print(f"Base sampled RMSE: {metrics['base_sampled_rmse']:.5f}")
    print(f"Sequence sampled RMSE: {metrics['sequence_sampled_rmse']:.5f}")
    print(
        "Relative sampled improvement: "
        f"{metrics['relative_rmse_improvement_pct_vs_sampled_base']:.2f}%"
    )


if __name__ == "__main__":
    main()
