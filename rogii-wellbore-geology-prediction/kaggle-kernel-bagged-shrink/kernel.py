from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


LOCAL_INPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
TRAIN_SAMPLE_PER_WELL = 900
ENSEMBLE_SEEDS = (17, 43, 89, 131, 197)
RESIDUAL_BLEND = 0.60
RESIDUAL_CLIP = 60.0

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


def well_features(
    path: Path,
    include_target: bool,
    sample_targets: bool = False,
    seed: int | None = None,
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
        if seed is None:
            positions = (
                np.linspace(0, len(target_indices) - 1, TRAIN_SAMPLE_PER_WELL)
                .round()
                .astype(int)
            )
            target_indices = target_indices[positions]
        else:
            rng = np.random.default_rng(seed)
            target_indices = np.sort(
                rng.choice(target_indices, size=TRAIN_SAMPLE_PER_WELL, replace=False)
            )

    last_known_index = known_indices[-1]
    prefix = frame.iloc[known_indices]
    suffix = frame.iloc[target_indices]
    last_known = frame.iloc[last_known_index]

    prefix_gr = prefix["GR"].dropna()
    last_known_gr = float(prefix_gr.iloc[-1]) if len(prefix_gr) else np.nan
    target_steps = target_indices - last_known_index
    target_count = len(np.where(target_mask)[0])

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
        features["residual"] = (
            suffix["TVT"].to_numpy(dtype=float) - features["baseline_tvt"]
        )

    for column in FEATURE_COLUMNS + ["baseline_tvt"]:
        features[column] = pd.to_numeric(features[column], downcast="float")
    if include_target:
        features["residual"] = pd.to_numeric(features["residual"], downcast="float")

    return features


def make_model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=260,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=4.0,
        random_state=seed,
    )


def load_sampled_training(seed: int) -> pd.DataFrame:
    frames = [
        well_features(
            path,
            include_target=True,
            sample_targets=True,
            seed=seed,
        )
        for path in sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
    ]
    return pd.concat(frames, ignore_index=True)


def train_models() -> list[HistGradientBoostingRegressor]:
    models = []
    for seed in ENSEMBLE_SEEDS:
        train_features = load_sampled_training(seed)
        print(
            f"Training residual model seed={seed} on {len(train_features):,} rows",
            flush=True,
        )
        model = make_model(seed)
        model.fit(train_features[FEATURE_COLUMNS], train_features["residual"])
        models.append(model)
    return models


def ensemble_residual(models: list[HistGradientBoostingRegressor], frame: pd.DataFrame) -> np.ndarray:
    raw_predictions = np.vstack(
        [model.predict(frame[FEATURE_COLUMNS]) for model in models]
    )
    clipped_predictions = np.clip(raw_predictions, -RESIDUAL_CLIP, RESIDUAL_CLIP)
    return clipped_predictions.mean(axis=0)


def main() -> None:
    models = train_models()

    submission = pd.read_csv(SUBMISSION_TEMPLATE)
    parsed = submission["id"].map(split_submission_id)
    submission["well_id"] = parsed.map(lambda value: value[0])
    submission["row_index"] = parsed.map(lambda value: value[1])

    prediction_frames = []
    for well_id in sorted(submission["well_id"].unique()):
        features = well_features(TEST_DIR / f"{well_id}__horizontal_well.csv", False)
        residual_mean = ensemble_residual(models, features)
        features["tvt"] = features["baseline_tvt"] + RESIDUAL_BLEND * residual_mean
        prediction_frames.append(features[["well_id", "row_index", "tvt"]])

    predictions = pd.concat(prediction_frames, ignore_index=True)
    output = submission.merge(predictions, on=["well_id", "row_index"], how="left")
    if output["tvt_y"].isna().any():
        missing = int(output["tvt_y"].isna().sum())
        raise ValueError(f"Submission has {missing} missing predictions.")
    output["tvt"] = output["tvt_y"].astype(float)
    output[["id", "tvt"]].to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH} with {len(output):,} rows", flush=True)


if __name__ == "__main__":
    main()
