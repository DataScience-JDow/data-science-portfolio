from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors


LOCAL_INPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
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


def find_raw_dir() -> Path:
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = sorted(kaggle_input.glob("**/sample_submission.csv"))
        if matches:
            return matches[0].parent
    return LOCAL_INPUT_DIR


RAW_DIR = find_raw_dir()
TRAIN_DIR = RAW_DIR / "train"
TEST_DIR = RAW_DIR / "test"
SUBMISSION_TEMPLATE = RAW_DIR / "sample_submission.csv"
OUTPUT_PATH = Path("/kaggle/working/submission.csv")

if not OUTPUT_PATH.parent.exists():
    OUTPUT_PATH = Path(__file__).resolve().parents[1] / "submissions" / "submission.csv"


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
    if len(target_indices) <= TRAIN_SAMPLE_PER_WELL:
        return target_indices
    positions = (
        np.linspace(0, len(target_indices) - 1, TRAIN_SAMPLE_PER_WELL)
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
    known_indices = np.where(~target_mask)[0]
    target_indices = np.where(target_mask)[0]
    if sample_targets:
        target_indices = sampled_target_indices(target_indices)

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
        features["residual"] = pd.to_numeric(features["residual"], downcast="float")

    return features


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=140,
        learning_rate=0.055,
        max_leaf_nodes=15,
        min_samples_leaf=120,
        l2_regularization=6.0,
        random_state=RANDOM_STATE,
    )


def main() -> None:
    train_features = pd.concat(
        [
            well_features(path, include_target=True, sample_targets=True)
            for path in sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
        ],
        ignore_index=True,
    )
    print(f"Training sequence GR residual model on {len(train_features):,} rows")

    model = make_model()
    model.fit(train_features[FEATURE_COLUMNS], train_features["residual"])

    submission = pd.read_csv(SUBMISSION_TEMPLATE)
    parsed = submission["id"].map(split_submission_id)
    submission["well_id"] = parsed.map(lambda value: value[0])
    submission["row_index"] = parsed.map(lambda value: value[1])

    prediction_frames = []
    for well_id in sorted(submission["well_id"].unique()):
        features = well_features(
            TEST_DIR / f"{well_id}__horizontal_well.csv",
            include_target=False,
            sample_targets=False,
        )
        raw_residual = model.predict(features[FEATURE_COLUMNS])
        blended_residual = RESIDUAL_BLEND * np.clip(
            raw_residual, -RESIDUAL_CLIP, RESIDUAL_CLIP
        )
        features["tvt"] = features["baseline_tvt"] + blended_residual
        prediction_frames.append(features[["well_id", "row_index", "tvt"]])

    predictions = pd.concat(prediction_frames, ignore_index=True)
    output = submission.merge(predictions, on=["well_id", "row_index"], how="left")
    output["tvt"] = output["tvt_y"].astype(float)
    output[["id", "tvt"]].to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH} with {len(output):,} rows")


if __name__ == "__main__":
    main()
