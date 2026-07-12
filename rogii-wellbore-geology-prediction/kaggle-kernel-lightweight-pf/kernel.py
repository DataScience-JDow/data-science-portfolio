from __future__ import annotations

from pathlib import Path
import glob

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor


LOCAL_INPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PARTICLES = 350
PF_SEEDS = (11, 23, 37, 53)
GR_SIGMA_MIN = 10.0
GR_SIGMA_MAX = 60.0
PROCESS_NOISE_POSITION = 0.015
PROCESS_NOISE_VELOCITY = 0.004
RESAMPLE_THRESHOLD = 0.55
ROUGHEN_POSITION = 0.15
ROUGHEN_VELOCITY = 0.002
PREFIX_TAIL = 40
TRAIN_SAMPLE_PER_WELL = 900
RESIDUAL_BLEND = 0.90
RESIDUAL_CLIP = 60.0
FORMATION_COLUMNS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
CONTROLS_PER_WELL = 30
NEIGHBORS_TO_QUERY = 48
NEIGHBORS_TO_AVERAGE = 12
DISTANCE_FLOOR = 250.0

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
PF_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    "pf_delta",
    "pf_std",
    "pf_vs_prefix_slope_50",
    "pf_vs_prefix_slope_200",
]
OFFSET_FEATURE_COLUMNS = PF_FEATURE_COLUMNS + [
    *(f"offset_depth_{column.lower()}" for column in FORMATION_COLUMNS),
    *(f"offset_delta_{column.lower()}" for column in FORMATION_COLUMNS),
    "offset_depth_mean",
    "offset_delta_mean",
    "offset_neighbor_distance",
]


def find_raw_dir() -> Path:
    for candidate in [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    ]:
        if (candidate / "train").exists() and (candidate / "sample_submission.csv").exists():
            return candidate
    for match in glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True):
        parent = Path(match).parent
        if (parent / "train").exists():
            return parent
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


def fill_gr(values: pd.Series, fallback: float) -> np.ndarray:
    return (
        values.astype(float)
        .interpolate(limit_direction="both")
        .fillna(fallback)
        .to_numpy(dtype=float)
    )


def well_features(path: Path, include_target: bool) -> pd.DataFrame:
    usecols = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
    if include_target:
        usecols.append("TVT")

    well_id = path.name.split("__", 1)[0]
    frame = pd.read_csv(path, usecols=usecols)
    target_mask = frame["TVT_input"].isna().to_numpy()
    known_indices = np.where(~target_mask)[0]
    target_indices = np.where(target_mask)[0]
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
            "query_x": suffix["X"].to_numpy(dtype=float),
            "query_y": suffix["Y"].to_numpy(dtype=float),
            "start_x": float(last_known["X"]),
            "start_y": float(last_known["Y"]),
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
        features["residual"] = pd.to_numeric(features["residual"], downcast="float")
    return features


def estimate_initial_velocity(prefix: pd.DataFrame) -> float:
    tail = prefix.tail(PREFIX_TAIL)
    md_delta = np.diff(tail["MD"].to_numpy(dtype=float))
    tvt_delta = np.diff(tail["TVT_input"].to_numpy(dtype=float))
    z_delta = np.diff(tail["Z"].to_numpy(dtype=float))
    mask = np.isfinite(md_delta) & np.isfinite(tvt_delta) & (md_delta > 0.0)
    if mask.sum() < 3:
        return 0.0
    return float(np.median((tvt_delta[mask] + z_delta[mask]) / md_delta[mask]))


def calibrate_typewell_gr(
    prefix: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
) -> tuple[float, float, float]:
    mask = prefix["GR"].notna() & prefix["TVT_input"].notna()
    if int(mask.sum()) < 20:
        return 1.0, 0.0, 30.0
    horizontal_gr = prefix.loc[mask, "GR"].to_numpy(dtype=float)
    expected_gr = np.interp(
        prefix.loc[mask, "TVT_input"].to_numpy(dtype=float),
        typewell_tvt,
        typewell_gr,
    )
    valid = np.isfinite(horizontal_gr) & np.isfinite(expected_gr)
    if valid.sum() < 20 or np.nanstd(expected_gr[valid]) < 1e-6:
        residual = horizontal_gr[valid] - expected_gr[valid]
        sigma = float(np.clip(np.nanstd(residual), GR_SIGMA_MIN, GR_SIGMA_MAX))
        return 1.0, 0.0, sigma
    slope, intercept = np.polyfit(expected_gr[valid], horizontal_gr[valid], 1)
    residual = horizontal_gr[valid] - (slope * expected_gr[valid] + intercept)
    sigma = float(np.clip(np.nanstd(residual), GR_SIGMA_MIN, GR_SIGMA_MAX))
    return float(slope), float(intercept), sigma


def systematic_resample(
    position: np.ndarray,
    velocity: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_particles = len(position)
    positions = (rng.random() + np.arange(n_particles)) / n_particles
    cumulative = np.cumsum(weights)
    indices = np.searchsorted(cumulative, positions, side="right")
    indices = np.clip(indices, 0, n_particles - 1)
    new_position = position[indices] + rng.normal(0.0, ROUGHEN_POSITION, n_particles)
    new_velocity = velocity[indices] + rng.normal(0.0, ROUGHEN_VELOCITY, n_particles)
    return new_position, new_velocity, np.full(n_particles, 1.0 / n_particles)


def run_particle_filter_once(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    seed: int,
    n_particles: int = PARTICLES,
) -> tuple[np.ndarray, np.ndarray]:
    target_mask = horizontal["TVT_input"].isna().to_numpy()
    known_indices = np.flatnonzero(~target_mask)
    target_indices = np.flatnonzero(target_mask)
    if len(known_indices) == 0 or len(target_indices) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    typewell = typewell.sort_values("TVT")
    typewell_tvt = typewell["TVT"].to_numpy(dtype=float)
    typewell_gr = fill_gr(typewell["GR"], float(typewell["GR"].mean()))
    prefix = horizontal.iloc[known_indices]
    suffix = horizontal.iloc[target_indices]
    last_known = prefix.iloc[-1]
    gr_slope, gr_intercept, gr_sigma = calibrate_typewell_gr(
        prefix,
        typewell_tvt,
        typewell_gr,
    )
    gr_values = fill_gr(horizontal["GR"], float(np.nanmean(typewell_gr)))[target_indices]
    md_values = suffix["MD"].to_numpy(dtype=float)
    z_values = suffix["Z"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    last_tvt = float(last_known["TVT_input"])
    last_z = float(last_known["Z"])
    initial_velocity = estimate_initial_velocity(prefix)
    position = last_tvt + last_z + rng.normal(0.0, 1.5, n_particles)
    velocity = initial_velocity + rng.normal(0.0, 0.02, n_particles)
    weights = np.full(n_particles, 1.0 / n_particles)
    predictions = np.empty(len(target_indices), dtype=np.float32)
    stds = np.empty(len(target_indices), dtype=np.float32)
    previous_md = float(last_known["MD"])
    tvt_min = float(typewell_tvt.min() - 150.0)
    tvt_max = float(typewell_tvt.max() + 150.0)
    for i, (md, z, gr) in enumerate(zip(md_values, z_values, gr_values)):
        md_step = max(float(md - previous_md), 1.0)
        position += velocity * md_step
        position += rng.normal(0.0, PROCESS_NOISE_POSITION * np.sqrt(md_step), n_particles)
        velocity += rng.normal(0.0, PROCESS_NOISE_VELOCITY, n_particles)
        tvt = position - float(z)
        tvt = np.clip(tvt, tvt_min, tvt_max)
        position = tvt + float(z)
        expected_gr = np.interp(tvt, typewell_tvt, typewell_gr)
        expected_gr = gr_slope * expected_gr + gr_intercept
        scaled_error = (float(gr) - expected_gr) / gr_sigma
        likelihood = np.exp(-0.5 * np.minimum(scaled_error * scaled_error, 600.0))
        likelihood = np.maximum(likelihood, 1e-300)
        weights *= likelihood
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0 or not np.isfinite(weight_sum):
            weights[:] = 1.0 / n_particles
        else:
            weights /= weight_sum
        mean_tvt = float(np.dot(weights, tvt))
        predictions[i] = mean_tvt
        stds[i] = float(np.sqrt(np.dot(weights, (tvt - mean_tvt) ** 2)))
        effective_n = 1.0 / float(np.dot(weights, weights))
        if effective_n < RESAMPLE_THRESHOLD * n_particles:
            position, velocity, weights = systematic_resample(
                position,
                velocity,
                weights,
                rng,
            )
        previous_md = float(md)
    return predictions, stds


def run_particle_filter_well(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    stds = []
    for seed in PF_SEEDS:
        pred, std = run_particle_filter_once(horizontal, typewell, seed=seed)
        predictions.append(pred)
        stds.append(std)
    pred_stack = np.vstack(predictions)
    std_stack = np.vstack(stds)
    return pred_stack.mean(axis=0), pred_stack.std(axis=0) + std_stack.mean(axis=0)


def prediction_frame_for_well(well_id: str, split: str) -> pd.DataFrame:
    base_dir = TRAIN_DIR if split == "train" else TEST_DIR
    horizontal = pd.read_csv(base_dir / f"{well_id}__horizontal_well.csv")
    typewell = pd.read_csv(base_dir / f"{well_id}__typewell.csv")
    target_mask = horizontal["TVT_input"].isna().to_numpy()
    target_indices = np.flatnonzero(target_mask)
    known = horizontal.loc[~target_mask]
    baseline = float(known["TVT_input"].iloc[-1])
    pf_pred, pf_std = run_particle_filter_well(horizontal, typewell)
    return pd.DataFrame(
        {
            "well_id": well_id,
            "row_index": target_indices,
            "baseline_tvt": baseline,
            "pf_tvt": pf_pred,
            "pf_delta": pf_pred - baseline,
            "pf_std": pf_std,
        }
    )


def add_pf_features(base_features: pd.DataFrame, pf_predictions: pd.DataFrame) -> pd.DataFrame:
    merged = base_features.merge(
        pf_predictions[["well_id", "row_index", "pf_tvt", "pf_delta", "pf_std"]],
        on=["well_id", "row_index"],
        how="left",
    )
    if merged["pf_tvt"].isna().any():
        missing = int(merged["pf_tvt"].isna().sum())
        raise ValueError(f"Missing particle-filter predictions for {missing} rows.")
    md_delta = merged["md_delta"].to_numpy(dtype=float)
    slope_50 = merged["prefix_tvt_slope_50"].to_numpy(dtype=float)
    slope_200 = merged["prefix_tvt_slope_200"].to_numpy(dtype=float)
    baseline = merged["baseline_tvt"].to_numpy(dtype=float)
    merged["pf_vs_prefix_slope_50"] = merged["pf_tvt"] - (baseline + slope_50 * md_delta)
    merged["pf_vs_prefix_slope_200"] = merged["pf_tvt"] - (baseline + slope_200 * md_delta)
    for column in ["pf_delta", "pf_std", "pf_vs_prefix_slope_50", "pf_vs_prefix_slope_200"]:
        merged[column] = pd.to_numeric(merged[column], downcast="float")
    return merged


def build_control_points() -> pd.DataFrame:
    frames = []
    for path in sorted(TRAIN_DIR.glob("*__horizontal_well.csv")):
        well_id = path.name.split("__", 1)[0]
        frame = pd.read_csv(path, usecols=["X", "Y", *FORMATION_COLUMNS])
        positions = np.linspace(0, len(frame) - 1, min(CONTROLS_PER_WELL, len(frame)))
        positions = np.unique(positions.round().astype(int))
        sampled = frame.iloc[positions].copy()
        sampled.insert(0, "well_id", well_id)
        frames.append(sampled)
    return pd.concat(frames, ignore_index=True)


def predict_surfaces(
    query_xy: np.ndarray,
    query_wells: np.ndarray,
    controls: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    control_xy = controls[["X", "Y"]].to_numpy(dtype=float)
    control_values = controls[list(FORMATION_COLUMNS)].to_numpy(dtype=float)
    control_wells = controls["well_id"].to_numpy()
    tree = cKDTree(control_xy)
    k = min(NEIGHBORS_TO_QUERY, len(controls))
    distances, indices = tree.query(query_xy, k=k, workers=-1)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    predictions = np.full((len(query_xy), len(FORMATION_COLUMNS)), np.nan, dtype=np.float32)
    mean_distances = np.full(len(query_xy), np.nan, dtype=np.float32)
    for row in range(len(query_xy)):
        candidate_indices = indices[row]
        candidate_distances = distances[row]
        different_well = control_wells[candidate_indices] != query_wells[row]
        candidate_indices = candidate_indices[different_well][:NEIGHBORS_TO_AVERAGE]
        candidate_distances = candidate_distances[different_well][:NEIGHBORS_TO_AVERAGE]
        if len(candidate_indices) == 0:
            continue
        values = control_values[candidate_indices]
        weights = 1.0 / np.maximum(candidate_distances, DISTANCE_FLOOR) ** 2
        for column in range(values.shape[1]):
            valid = np.isfinite(values[:, column])
            if valid.any():
                predictions[row, column] = np.average(
                    values[valid, column],
                    weights=weights[valid],
                )
        mean_distances[row] = float(np.average(candidate_distances, weights=weights))
    return predictions, mean_distances


def add_offset_features(frame: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    query_wells = output["well_id"].to_numpy()
    query_values, query_distance = predict_surfaces(
        output[["query_x", "query_y"]].to_numpy(dtype=float),
        query_wells,
        controls,
    )
    starts = output[["well_id", "start_x", "start_y"]].drop_duplicates("well_id")
    start_values, _ = predict_surfaces(
        starts[["start_x", "start_y"]].to_numpy(dtype=float),
        starts["well_id"].to_numpy(),
        controls,
    )
    start_lookup = {well_id: values for well_id, values in zip(starts["well_id"], start_values)}
    predicted_start = np.vstack([start_lookup[well_id] for well_id in query_wells])

    depths = -query_values
    deltas = -(query_values - predicted_start)
    for i, column in enumerate(FORMATION_COLUMNS):
        output[f"offset_depth_{column.lower()}"] = depths[:, i]
        output[f"offset_delta_{column.lower()}"] = deltas[:, i]
    output["offset_depth_mean"] = np.nanmean(depths, axis=1)
    output["offset_delta_mean"] = np.nanmean(deltas, axis=1)
    output["offset_neighbor_distance"] = query_distance
    for column in OFFSET_FEATURE_COLUMNS[len(PF_FEATURE_COLUMNS) :]:
        output[column] = pd.to_numeric(output[column], downcast="float")
    return output


def sample_training_rows(data: pd.DataFrame) -> np.ndarray:
    sampled_indices = []
    for _, local_indices in data.groupby("well_id", sort=False).indices.items():
        absolute_indices = data.index.to_numpy()[local_indices]
        if len(absolute_indices) > TRAIN_SAMPLE_PER_WELL:
            positions = (
                np.linspace(0, len(absolute_indices) - 1, TRAIN_SAMPLE_PER_WELL)
                .round()
                .astype(int)
            )
            absolute_indices = absolute_indices[positions]
        sampled_indices.append(absolute_indices)
    return np.concatenate(sampled_indices)


def make_model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=260,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=4.0,
        random_state=seed,
    )


def main() -> None:
    train_wells = sorted(
        path.name.split("__", 1)[0]
        for path in TRAIN_DIR.glob("*__horizontal_well.csv")
    )
    test_wells = sorted(
        path.name.split("__", 1)[0]
        for path in TEST_DIR.glob("*__horizontal_well.csv")
    )
    print(f"train wells: {len(train_wells)} | test wells: {len(test_wells)}", flush=True)

    train_base_frames = []
    train_pf_frames = []
    for i, well_id in enumerate(train_wells, start=1):
        if i % 50 == 0:
            print(f"training PF wells: {i}/{len(train_wells)}", flush=True)
        train_base_frames.append(well_features(TRAIN_DIR / f"{well_id}__horizontal_well.csv", True))
        train_pf_frames.append(prediction_frame_for_well(well_id, "train"))
    train_data = add_pf_features(
        pd.concat(train_base_frames, ignore_index=True),
        pd.concat(train_pf_frames, ignore_index=True),
    )
    controls = build_control_points()
    sampled_indices = sample_training_rows(train_data)
    offset_train = add_offset_features(train_data.loc[sampled_indices], controls)
    print(
        f"training offset/PF residual model on {len(sampled_indices):,} rows",
        flush=True,
    )
    model = make_model(59)
    model.fit(
        offset_train[OFFSET_FEATURE_COLUMNS],
        offset_train["residual"],
    )

    test_base_frames = []
    test_pf_frames = []
    for well_id in test_wells:
        test_base_frames.append(well_features(TEST_DIR / f"{well_id}__horizontal_well.csv", False))
        test_pf_frames.append(prediction_frame_for_well(well_id, "test"))
    test_data = add_pf_features(
        pd.concat(test_base_frames, ignore_index=True),
        pd.concat(test_pf_frames, ignore_index=True),
    )
    test_data = add_offset_features(test_data, controls)
    raw_residual = model.predict(test_data[OFFSET_FEATURE_COLUMNS])
    blended_residual = RESIDUAL_BLEND * np.clip(
        raw_residual,
        -RESIDUAL_CLIP,
        RESIDUAL_CLIP,
    )
    test_data["tvt_prediction"] = test_data["baseline_tvt"] + blended_residual

    submission = pd.read_csv(SUBMISSION_TEMPLATE)
    parsed = submission["id"].map(split_submission_id)
    submission["well_id"] = parsed.map(lambda value: value[0])
    submission["row_index"] = parsed.map(lambda value: value[1])
    output = submission.merge(
        test_data[["well_id", "row_index", "tvt_prediction"]],
        on=["well_id", "row_index"],
        how="left",
    )
    if output["tvt_prediction"].isna().any():
        missing = int(output["tvt_prediction"].isna().sum())
        raise ValueError(f"Submission has {missing} missing predictions.")
    output["tvt"] = output["tvt_prediction"].astype(float)
    output[["id", "tvt"]].to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH} with {len(output):,} rows", flush=True)


if __name__ == "__main__":
    main()
