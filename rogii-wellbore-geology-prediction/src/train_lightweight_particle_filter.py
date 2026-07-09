from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from train_residual_correction import (
    FEATURE_COLUMNS,
    RAW_DIR,
    REPORTS_DIR,
    SUBMISSION_TEMPLATE,
    SUBMISSIONS_DIR,
    TEST_DIR,
    TRAIN_DIR,
    load_training_features,
    rmse,
    split_submission_id,
    well_features,
)


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
VALIDATION_SAMPLE_PER_WELL = 800
CURRENT_BEST_SUBMISSION = SUBMISSIONS_DIR / "bagged_shrink_residual_submission.csv"
TRAIN_SAMPLE_PER_WELL = 900
RESIDUAL_BLEND = 0.60
RESIDUAL_CLIP = 60.0
PF_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    "pf_delta",
    "pf_std",
    "pf_vs_prefix_slope_50",
    "pf_vs_prefix_slope_200",
]


def fill_gr(values: pd.Series, fallback: float) -> np.ndarray:
    return (
        values.astype(float)
        .interpolate(limit_direction="both")
        .fillna(fallback)
        .to_numpy(dtype=float)
    )


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


def prediction_frame_for_well(
    well_id: str,
    split: str,
    include_target: bool,
) -> pd.DataFrame:
    base_dir = TRAIN_DIR if split == "train" else TEST_DIR
    horizontal = pd.read_csv(base_dir / f"{well_id}__horizontal_well.csv")
    typewell = pd.read_csv(base_dir / f"{well_id}__typewell.csv")
    target_mask = horizontal["TVT_input"].isna().to_numpy()
    target_indices = np.flatnonzero(target_mask)
    known = horizontal.loc[~target_mask]
    baseline = float(known["TVT_input"].iloc[-1])
    pf_pred, pf_std = run_particle_filter_well(horizontal, typewell)

    frame = pd.DataFrame(
        {
            "well_id": well_id,
            "row_index": target_indices,
            "baseline_tvt": baseline,
            "pf_tvt": pf_pred,
            "pf_delta": pf_pred - baseline,
            "pf_std": pf_std,
        }
    )
    if include_target:
        frame["tvt"] = horizontal.loc[target_indices, "TVT"].to_numpy(dtype=float)
    return frame


def sample_validation_rows(data: pd.DataFrame) -> pd.DataFrame:
    sampled = []
    for _, group in data.groupby("well_id", sort=False):
        if len(group) > VALIDATION_SAMPLE_PER_WELL:
            positions = (
                np.linspace(0, len(group) - 1, VALIDATION_SAMPLE_PER_WELL)
                .round()
                .astype(int)
            )
            group = group.iloc[positions]
        sampled.append(group)
    return pd.concat(sampled, ignore_index=True)


def evaluate_particle_filter(data: pd.DataFrame) -> dict:
    sampled = sample_validation_rows(data)
    target = sampled["tvt"].to_numpy(dtype=float)
    baseline = sampled["baseline_tvt"].to_numpy(dtype=float)
    pf = sampled["pf_tvt"].to_numpy(dtype=float)

    alpha_rows = []
    for alpha in np.round(np.arange(0.0, 1.01, 0.05), 2):
        pred = baseline + alpha * (pf - baseline)
        alpha_rows.append({"alpha": float(alpha), "rmse": rmse(target, pred)})
    best_alpha = min(alpha_rows, key=lambda row: row["rmse"])

    current_best = None
    current_best_path = CURRENT_BEST_SUBMISSION
    if current_best_path.exists():
        current = pd.read_csv(current_best_path)
        current["well_id"] = current["id"].map(lambda value: split_submission_id(value)[0])
        current["row_index"] = current["id"].map(lambda value: split_submission_id(value)[1])
        merged = sampled.merge(
            current[["well_id", "row_index", "tvt"]].rename(
                columns={"tvt": "current_best_tvt"}
            ),
            on=["well_id", "row_index"],
            how="left",
        )
        if not merged["current_best_tvt"].isna().any():
            blend_rows = []
            current_pred = merged["current_best_tvt"].to_numpy(dtype=float)
            for alpha in np.round(np.arange(0.0, 1.01, 0.05), 2):
                pred = (1.0 - alpha) * current_pred + alpha * merged[
                    "pf_tvt"
                ].to_numpy(dtype=float)
                blend_rows.append({"pf_weight": float(alpha), "rmse": rmse(target, pred)})
            current_best = {
                "current_best_rmse": rmse(target, current_pred),
                "blend_sweep": blend_rows,
                "best_blend": min(blend_rows, key=lambda row: row["rmse"]),
            }

    per_well = []
    for well_id, group in sampled.groupby("well_id"):
        y = group["tvt"].to_numpy(dtype=float)
        b = group["baseline_tvt"].to_numpy(dtype=float)
        p = group["pf_tvt"].to_numpy(dtype=float)
        per_well.append(
            {
                "well_id": well_id,
                "rows": int(len(group)),
                "baseline_rmse": rmse(y, b),
                "pf_rmse": rmse(y, p),
                "best_alpha_prediction_rmse": rmse(
                    y,
                    b + float(best_alpha["alpha"]) * (p - b),
                ),
                "mean_pf_std": float(group["pf_std"].mean()),
            }
        )

    metrics = {
        "method": "lightweight_particle_filter",
        "validation_type": "sampled_all_training_wells",
        "train_wells": int(data["well_id"].nunique()),
        "validation_rows": int(len(sampled)),
        "particles": PARTICLES,
        "pf_seeds": list(PF_SEEDS),
        "validation_sample_per_well": VALIDATION_SAMPLE_PER_WELL,
        "baseline_rmse": rmse(target, baseline),
        "pf_rmse": rmse(target, pf),
        "best_baseline_pf_alpha": best_alpha,
        "alpha_sweep": alpha_rows,
        "per_well": per_well,
    }
    if current_best is not None:
        metrics["current_best_comparison"] = current_best
    return metrics


def sample_training_rows(
    data: pd.DataFrame,
    indices: np.ndarray,
    rows_per_well: int,
) -> np.ndarray:
    subset = data.iloc[indices]
    sampled_indices = []

    for _, local_indices in subset.groupby("well_id", sort=False).indices.items():
        absolute_indices = subset.index.to_numpy()[local_indices]
        if len(absolute_indices) > rows_per_well:
            positions = (
                np.linspace(0, len(absolute_indices) - 1, rows_per_well)
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


def validate_pf_feature_model(data: pd.DataFrame) -> dict:
    all_indices = np.arange(len(data))
    valid_eval_idx = sample_training_rows(
        data,
        all_indices,
        VALIDATION_SAMPLE_PER_WELL,
    )
    eval_data = data.loc[valid_eval_idx].reset_index(drop=True)
    groups = eval_data["well_id"].to_numpy()

    base_oof = np.zeros(len(eval_data), dtype=np.float32)
    pf_oof = np.zeros(len(eval_data), dtype=np.float32)
    fold_rows = []

    splitter = GroupKFold(n_splits=5)
    for fold, (train_group_idx, valid_idx) in enumerate(
        splitter.split(eval_data[FEATURE_COLUMNS], eval_data["residual"], groups),
        start=1,
    ):
        train_wells = set(eval_data.iloc[train_group_idx]["well_id"])
        train_idx = data.index[data["well_id"].isin(train_wells)].to_numpy()
        sampled_train_idx = sample_training_rows(
            data,
            train_idx,
            TRAIN_SAMPLE_PER_WELL,
        )

        base_model = make_model(seed=42)
        base_model.fit(
            data.loc[sampled_train_idx, FEATURE_COLUMNS],
            data.loc[sampled_train_idx, "residual"],
        )
        pf_model = make_model(seed=43)
        pf_model.fit(
            data.loc[sampled_train_idx, PF_FEATURE_COLUMNS],
            data.loc[sampled_train_idx, "residual"],
        )

        valid_frame = eval_data.iloc[valid_idx]
        base_raw = base_model.predict(valid_frame[FEATURE_COLUMNS])
        pf_raw = pf_model.predict(valid_frame[PF_FEATURE_COLUMNS])
        base_oof[valid_idx] = np.clip(base_raw, -RESIDUAL_CLIP, RESIDUAL_CLIP)
        pf_oof[valid_idx] = np.clip(pf_raw, -RESIDUAL_CLIP, RESIDUAL_CLIP)

        baseline = valid_frame["baseline_tvt"].to_numpy(dtype=float)
        target = valid_frame["tvt"].to_numpy(dtype=float)
        base_pred = baseline + RESIDUAL_BLEND * base_oof[valid_idx]
        pf_pred = baseline + RESIDUAL_BLEND * pf_oof[valid_idx]
        fold_rows.append(
            {
                "fold": fold,
                "validation_rows": int(len(valid_idx)),
                "train_sample_rows": int(len(sampled_train_idx)),
                "base_rmse": rmse(target, base_pred),
                "pf_feature_rmse": rmse(target, pf_pred),
            }
        )
        print(
            f"Fold {fold}: base={fold_rows[-1]['base_rmse']:.5f}, "
            f"pf_features={fold_rows[-1]['pf_feature_rmse']:.5f}",
            flush=True,
        )

    baseline = eval_data["baseline_tvt"].to_numpy(dtype=float)
    target = eval_data["tvt"].to_numpy(dtype=float)
    base_pred = baseline + RESIDUAL_BLEND * base_oof
    pf_pred = baseline + RESIDUAL_BLEND * pf_oof

    blend_rows = []
    for pf_weight in np.round(np.arange(0.0, 1.01, 0.05), 2):
        pred = (1.0 - pf_weight) * base_pred + pf_weight * pf_pred
        blend_rows.append({"pf_feature_weight": float(pf_weight), "rmse": rmse(target, pred)})

    best_blend = min(blend_rows, key=lambda row: row["rmse"])
    return {
        "validation_type": "grouped_by_well_sampled_rows",
        "validation_rows": int(len(eval_data)),
        "train_sample_per_well": TRAIN_SAMPLE_PER_WELL,
        "residual_blend": RESIDUAL_BLEND,
        "residual_clip": RESIDUAL_CLIP,
        "base_feature_rmse": rmse(target, base_pred),
        "pf_feature_rmse": rmse(target, pf_pred),
        "absolute_rmse_improvement_vs_base_features": rmse(target, base_pred)
        - rmse(target, pf_pred),
        "relative_rmse_improvement_pct_vs_base_features": (
            (rmse(target, base_pred) - rmse(target, pf_pred))
            / rmse(target, base_pred)
            * 100.0
        ),
        "best_base_pf_feature_blend": best_blend,
        "blend_sweep": blend_rows,
        "folds": fold_rows,
    }


def build_raw_pf_test_submission() -> pd.DataFrame:
    submission = pd.read_csv(SUBMISSION_TEMPLATE)
    parsed = submission["id"].map(split_submission_id)
    submission["well_id"] = parsed.map(lambda value: value[0])
    submission["row_index"] = parsed.map(lambda value: value[1])

    frames = []
    for well_id in sorted(submission["well_id"].unique()):
        frames.append(prediction_frame_for_well(well_id, "test", include_target=False))

    predictions = pd.concat(frames, ignore_index=True)
    output = submission.merge(predictions, on=["well_id", "row_index"], how="left")
    if output["pf_tvt"].isna().any():
        missing = int(output["pf_tvt"].isna().sum())
        raise ValueError(f"Submission has {missing} missing predictions.")
    output["tvt"] = output["pf_tvt"].astype(float)
    return output[["id", "tvt"]]


def train_final_pf_feature_model(data: pd.DataFrame) -> HistGradientBoostingRegressor:
    all_indices = np.arange(len(data))
    sampled_indices = sample_training_rows(
        data,
        all_indices,
        TRAIN_SAMPLE_PER_WELL,
    )
    model = make_model(seed=43)
    model.fit(
        data.loc[sampled_indices, PF_FEATURE_COLUMNS],
        data.loc[sampled_indices, "residual"],
    )
    return model


def build_pf_feature_test_submission(
    model: HistGradientBoostingRegressor,
) -> pd.DataFrame:
    submission = pd.read_csv(SUBMISSION_TEMPLATE)
    parsed = submission["id"].map(split_submission_id)
    submission["well_id"] = parsed.map(lambda value: value[0])
    submission["row_index"] = parsed.map(lambda value: value[1])

    base_frames = []
    pf_frames = []
    for well_id in sorted(submission["well_id"].unique()):
        base_frames.append(well_features(TEST_DIR / f"{well_id}__horizontal_well.csv", False))
        pf_frames.append(prediction_frame_for_well(well_id, "test", include_target=False))

    test_features = add_pf_features(
        pd.concat(base_frames, ignore_index=True),
        pd.concat(pf_frames, ignore_index=True),
    )
    raw_residual = model.predict(test_features[PF_FEATURE_COLUMNS])
    blended_residual = RESIDUAL_BLEND * np.clip(
        raw_residual,
        -RESIDUAL_CLIP,
        RESIDUAL_CLIP,
    )
    test_features["tvt_prediction"] = test_features["baseline_tvt"] + blended_residual

    output = submission.merge(
        test_features[["well_id", "row_index", "tvt_prediction"]],
        on=["well_id", "row_index"],
        how="left",
    )
    if output["tvt_prediction"].isna().any():
        missing = int(output["tvt_prediction"].isna().sum())
        raise ValueError(f"Submission has {missing} missing predictions.")
    output["tvt"] = output["tvt_prediction"].astype(float)
    return output[["id", "tvt"]]


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    train_wells = sorted(
        path.name.split("__", 1)[0]
        for path in TRAIN_DIR.glob("*__horizontal_well.csv")
    )
    frames = []
    for i, well_id in enumerate(train_wells, start=1):
        if i % 50 == 0:
            print(f"Particle filter validation wells: {i}/{len(train_wells)}", flush=True)
        frames.append(prediction_frame_for_well(well_id, "train", include_target=True))
    validation_predictions = pd.concat(frames, ignore_index=True)
    metrics = evaluate_particle_filter(validation_predictions)
    print("Loading base residual features for PF feature validation...", flush=True)
    base_features = load_training_features()
    feature_data = add_pf_features(base_features, validation_predictions)
    metrics["pf_feature_model"] = validate_pf_feature_model(feature_data)
    model = train_final_pf_feature_model(feature_data)
    raw_pf_submission = build_raw_pf_test_submission()
    pf_feature_submission = build_pf_feature_test_submission(model)

    metrics_path = REPORTS_DIR / "lightweight-particle-filter-metrics.json"
    predictions_path = REPORTS_DIR / "lightweight-particle-filter-validation.csv"
    raw_submission_path = SUBMISSIONS_DIR / "lightweight_particle_filter_submission.csv"
    pf_feature_submission_path = (
        SUBMISSIONS_DIR / "lightweight_particle_filter_feature_submission.csv"
    )
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    validation_predictions.to_csv(predictions_path, index=False)
    raw_pf_submission.to_csv(raw_submission_path, index=False)
    pf_feature_submission.to_csv(pf_feature_submission_path, index=False)

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote validation predictions: {predictions_path}")
    print(f"Wrote raw PF submission: {raw_submission_path}")
    print(f"Wrote PF feature submission: {pf_feature_submission_path}")
    print(f"Baseline sampled RMSE: {metrics['baseline_rmse']:.5f}")
    print(f"Particle filter sampled RMSE: {metrics['pf_rmse']:.5f}")
    best = metrics["best_baseline_pf_alpha"]
    print(f"Best baseline/PF alpha: {best['alpha']:.2f} -> {best['rmse']:.5f}")
    if "current_best_comparison" in metrics:
        cb = metrics["current_best_comparison"]
        print(f"Current best sampled RMSE: {cb['current_best_rmse']:.5f}")
        print(
            "Best current/PF blend: "
            f"{cb['best_blend']['pf_weight']:.2f} -> {cb['best_blend']['rmse']:.5f}"
        )


if __name__ == "__main__":
    main()
