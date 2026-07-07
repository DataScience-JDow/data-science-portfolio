from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


LOCAL_INPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
TRAIN_SAMPLE_PER_WELL = 1400
RESIDUAL_BLEND = 0.60
RESIDUAL_CLIP = 60.0
TYPEWELL_TVT_WINDOW = 180.0
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

TYPEWELL_GR_FEATURE_COLUMNS = [
    "typewell_gr_at_baseline",
    "typewell_gr_delta_at_baseline",
    "typewell_match_tvt_delta",
    "typewell_match_abs_gr_error",
    "typewell_match_available",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + TYPEWELL_GR_FEATURE_COLUMNS


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


def load_typewell_curve(typewell_path: Path) -> tuple[np.ndarray, np.ndarray]:
    typewell = (
        pd.read_csv(typewell_path, usecols=["TVT", "GR"])
        .dropna()
        .sort_values("TVT")
    )
    typewell = typewell.groupby("TVT", as_index=False)["GR"].mean()
    return typewell["TVT"].to_numpy(dtype=float), typewell["GR"].to_numpy(dtype=float)


def typewell_gr_match_features(
    horizontal_gr: np.ndarray,
    baseline_tvt: float,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
) -> dict[str, np.ndarray]:
    row_count = len(horizontal_gr)
    gr_at_baseline = np.interp(
        np.full(row_count, baseline_tvt),
        typewell_tvt,
        typewell_gr,
        left=np.nan,
        right=np.nan,
    )
    gr_delta_at_baseline = horizontal_gr - gr_at_baseline
    match_tvt_delta = np.full(row_count, np.nan, dtype=float)
    match_abs_gr_error = np.full(row_count, np.nan, dtype=float)
    match_available = np.isfinite(horizontal_gr).astype(float)

    candidate_mask = (
        (typewell_tvt >= baseline_tvt - TYPEWELL_TVT_WINDOW)
        & (typewell_tvt <= baseline_tvt + TYPEWELL_TVT_WINDOW)
    )
    candidate_tvt = typewell_tvt[candidate_mask]
    candidate_gr = typewell_gr[candidate_mask]
    if len(candidate_tvt) == 0:
        return {
            "typewell_gr_at_baseline": gr_at_baseline,
            "typewell_gr_delta_at_baseline": gr_delta_at_baseline,
            "typewell_match_tvt_delta": match_tvt_delta,
            "typewell_match_abs_gr_error": match_abs_gr_error,
            "typewell_match_available": match_available,
        }

    for row_index in np.where(np.isfinite(horizontal_gr))[0]:
        best_index = int(np.argmin(np.abs(candidate_gr - horizontal_gr[row_index])))
        match_tvt_delta[row_index] = candidate_tvt[best_index] - baseline_tvt
        match_abs_gr_error[row_index] = abs(
            candidate_gr[best_index] - horizontal_gr[row_index]
        )

    return {
        "typewell_gr_at_baseline": gr_at_baseline,
        "typewell_gr_delta_at_baseline": gr_delta_at_baseline,
        "typewell_match_tvt_delta": match_tvt_delta,
        "typewell_match_abs_gr_error": match_abs_gr_error,
        "typewell_match_available": match_available,
    }


def well_features(
    path: Path,
    include_target: bool,
    sample_targets: bool = False,
) -> pd.DataFrame:
    usecols = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
    if include_target:
        usecols.append("TVT")

    well_id = path.name.split("__", 1)[0]
    frame = pd.read_csv(path, usecols=usecols)
    target_mask = frame["TVT_input"].isna().to_numpy()
    known_indices = np.where(~target_mask)[0]
    target_indices = np.where(target_mask)[0]
    if sample_targets and len(target_indices) > TRAIN_SAMPLE_PER_WELL:
        positions = (
            np.linspace(0, len(target_indices) - 1, TRAIN_SAMPLE_PER_WELL)
            .round()
            .astype(int)
        )
        target_indices = target_indices[positions]

    last_known_index = known_indices[-1]
    prefix = frame.iloc[known_indices]
    suffix = frame.iloc[target_indices]
    last_known = frame.iloc[last_known_index]

    prefix_gr = prefix["GR"].dropna()
    last_known_gr = float(prefix_gr.iloc[-1]) if len(prefix_gr) else np.nan
    target_steps = target_indices - last_known_index
    target_count = len(np.where(target_mask)[0])
    baseline_tvt = float(last_known["TVT_input"])
    horizontal_gr = suffix["GR"].to_numpy(dtype=float)
    typewell_tvt, typewell_gr = load_typewell_curve(
        path.with_name(f"{well_id}__typewell.csv")
    )
    typewell_features = typewell_gr_match_features(
        horizontal_gr,
        baseline_tvt,
        typewell_tvt,
        typewell_gr,
    )

    prefix_tvt_slope_50 = linear_slope(
        prefix["MD"].tail(50).to_numpy(dtype=float),
        prefix["TVT_input"].tail(50).to_numpy(dtype=float),
    )
    prefix_tvt_slope_200 = linear_slope(
        prefix["MD"].tail(200).to_numpy(dtype=float),
        prefix["TVT_input"].tail(200).to_numpy(dtype=float),
    )
    prefix_z_slope_200 = linear_slope(
        prefix["MD"].tail(200).to_numpy(dtype=float),
        prefix["Z"].tail(200).to_numpy(dtype=float),
    )

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
            "gr": horizontal_gr,
            "gr_missing": suffix["GR"].isna().astype(float).to_numpy(),
            "gr_delta_last": horizontal_gr - last_known_gr,
            "baseline_tvt": baseline_tvt,
            "last_known_tvt": baseline_tvt,
            "known_rows": float(len(known_indices)),
            "target_rows": float(target_count),
            "prefix_tvt_slope_50": prefix_tvt_slope_50,
            "prefix_tvt_slope_200": prefix_tvt_slope_200,
            "prefix_z_slope_200": prefix_z_slope_200,
            "prefix_gr_mean_200": float(prefix_gr.tail(200).mean())
            if len(prefix_gr)
            else np.nan,
            "prefix_gr_std_200": float(prefix_gr.tail(200).std())
            if len(prefix_gr) > 1
            else 0.0,
            **typewell_features,
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
        max_iter=260,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=4.0,
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
    print(f"Training typewell GR residual model on {len(train_features):,} rows")

    model = make_model()
    model.fit(train_features[FEATURE_COLUMNS], train_features["residual"])

    submission = pd.read_csv(SUBMISSION_TEMPLATE)
    parsed = submission["id"].map(split_submission_id)
    submission["well_id"] = parsed.map(lambda value: value[0])
    submission["row_index"] = parsed.map(lambda value: value[1])

    prediction_frames = []
    for well_id in sorted(submission["well_id"].unique()):
        features = well_features(TEST_DIR / f"{well_id}__horizontal_well.csv", False)
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
