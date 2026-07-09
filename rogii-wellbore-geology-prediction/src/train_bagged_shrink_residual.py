from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from train_residual_correction import (
    FEATURE_COLUMNS,
    REPORTS_DIR,
    RESIDUAL_CLIP,
    SUBMISSION_TEMPLATE,
    SUBMISSIONS_DIR,
    TEST_DIR,
    load_training_features,
    rmse,
    split_submission_id,
    well_features,
)


TRAIN_SAMPLE_PER_WELL = 900
VALIDATION_SAMPLE_PER_WELL = 800
ENSEMBLE_SEEDS = (17, 43, 89, 131, 197)
DEFAULT_BLEND = 0.60
OUTPUT_BLEND = 0.60


def sample_rows_per_well(
    data: pd.DataFrame,
    indices: np.ndarray,
    rows_per_well: int,
    seed: int | None = None,
) -> np.ndarray:
    subset = data.iloc[indices]
    sampled_indices = []
    rng = np.random.default_rng(seed)

    for _, local_indices in subset.groupby("well_id", sort=False).indices.items():
        absolute_indices = subset.index.to_numpy()[local_indices]
        if len(absolute_indices) > rows_per_well:
            if seed is None:
                positions = (
                    np.linspace(0, len(absolute_indices) - 1, rows_per_well)
                    .round()
                    .astype(int)
                )
                absolute_indices = absolute_indices[positions]
            else:
                absolute_indices = np.sort(
                    rng.choice(absolute_indices, size=rows_per_well, replace=False)
                )
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


def predict_ensemble(
    models: list[HistGradientBoostingRegressor],
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    raw_predictions = np.vstack(
        [model.predict(frame[FEATURE_COLUMNS]) for model in models]
    )
    clipped_predictions = np.clip(raw_predictions, -RESIDUAL_CLIP, RESIDUAL_CLIP)
    return clipped_predictions.mean(axis=0), clipped_predictions.std(axis=0)


def dynamic_alpha(
    residual_std: np.ndarray,
    base_alpha: float,
    shrink_strength: float,
    scale: float,
    minimum_alpha: float,
) -> np.ndarray:
    if scale <= 0.0:
        return np.full(len(residual_std), base_alpha, dtype=np.float32)
    shrink = 1.0 / (1.0 + shrink_strength * residual_std / scale)
    alpha = base_alpha * shrink
    return np.clip(alpha, minimum_alpha, base_alpha).astype(np.float32)


def evaluate_alpha_grid(
    baseline: np.ndarray,
    target: np.ndarray,
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
) -> tuple[list[dict], dict]:
    rows = []

    for alpha in np.round(np.arange(0.0, 1.01, 0.1), 2):
        prediction = baseline + alpha * residual_mean
        rows.append(
            {
                "mode": "fixed",
                "alpha": float(alpha),
                "rmse": rmse(target, prediction),
            }
        )

    finite_std = residual_std[np.isfinite(residual_std)]
    scales = [
        float(np.quantile(finite_std, quantile))
        for quantile in (0.50, 0.75, 0.90)
        if len(finite_std)
    ]
    shrink_strengths = (0.25, 0.50, 0.75, 1.00, 1.50, 2.00)
    base_alphas = (0.50, 0.60, 0.70, 0.80)
    minimum_alphas = (0.05, 0.10, 0.20, 0.30)

    for base_alpha in base_alphas:
        for shrink_strength in shrink_strengths:
            for scale in scales:
                for minimum_alpha in minimum_alphas:
                    alpha = dynamic_alpha(
                        residual_std=residual_std,
                        base_alpha=base_alpha,
                        shrink_strength=shrink_strength,
                        scale=scale,
                        minimum_alpha=minimum_alpha,
                    )
                    prediction = baseline + alpha * residual_mean
                    rows.append(
                        {
                            "mode": "dynamic",
                            "base_alpha": float(base_alpha),
                            "shrink_strength": float(shrink_strength),
                            "std_scale": float(scale),
                            "minimum_alpha": float(minimum_alpha),
                            "mean_alpha": float(alpha.mean()),
                            "p10_alpha": float(np.quantile(alpha, 0.10)),
                            "p90_alpha": float(np.quantile(alpha, 0.90)),
                            "rmse": rmse(target, prediction),
                        }
                    )

    best = min(rows, key=lambda row: row["rmse"])
    return rows, best


def validate_bagged_shrinkage(data: pd.DataFrame) -> dict:
    all_indices = np.arange(len(data))
    valid_eval_idx = sample_rows_per_well(
        data,
        all_indices,
        VALIDATION_SAMPLE_PER_WELL,
        seed=None,
    )
    eval_data = data.loc[valid_eval_idx].reset_index(drop=True)

    groups = eval_data["well_id"].to_numpy()
    single_residual_oof = np.zeros(len(eval_data), dtype=np.float32)
    residual_mean_oof = np.zeros(len(eval_data), dtype=np.float32)
    residual_std_oof = np.zeros(len(eval_data), dtype=np.float32)
    fold_rows = []

    splitter = GroupKFold(n_splits=5)
    for fold, (train_group_idx, valid_idx) in enumerate(
        splitter.split(eval_data[FEATURE_COLUMNS], eval_data["residual"], groups),
        start=1,
    ):
        train_wells = set(eval_data.iloc[train_group_idx]["well_id"])
        train_idx = data.index[data["well_id"].isin(train_wells)].to_numpy()

        single_train_idx = sample_rows_per_well(
            data,
            train_idx,
            TRAIN_SAMPLE_PER_WELL,
            seed=None,
        )
        single_model = make_model(42)
        single_model.fit(
            data.loc[single_train_idx, FEATURE_COLUMNS],
            data.loc[single_train_idx, "residual"],
        )

        models = []
        train_sample_sizes = []
        for seed in ENSEMBLE_SEEDS:
            sampled_train_idx = sample_rows_per_well(
                data,
                train_idx,
                TRAIN_SAMPLE_PER_WELL,
                seed=seed + fold * 1000,
            )
            model = make_model(seed)
            model.fit(
                data.loc[sampled_train_idx, FEATURE_COLUMNS],
                data.loc[sampled_train_idx, "residual"],
            )
            models.append(model)
            train_sample_sizes.append(int(len(sampled_train_idx)))

        valid_frame = eval_data.iloc[valid_idx]
        single_raw = single_model.predict(valid_frame[FEATURE_COLUMNS])
        single_residual_oof[valid_idx] = np.clip(
            single_raw,
            -RESIDUAL_CLIP,
            RESIDUAL_CLIP,
        )
        residual_mean, residual_std = predict_ensemble(models, valid_frame)
        residual_mean_oof[valid_idx] = residual_mean
        residual_std_oof[valid_idx] = residual_std

        baseline = valid_frame["baseline_tvt"].to_numpy()
        target = valid_frame["tvt"].to_numpy()
        single_prediction = baseline + DEFAULT_BLEND * single_residual_oof[valid_idx]
        fixed_prediction = baseline + DEFAULT_BLEND * residual_mean
        fold_rows.append(
            {
                "fold": fold,
                "validation_rows": int(len(valid_idx)),
                "single_train_sample_rows": int(len(single_train_idx)),
                "train_sample_rows_per_model": train_sample_sizes,
                "baseline_rmse": rmse(target, baseline),
                "single_fixed_blend_rmse": rmse(target, single_prediction),
                "fixed_ensemble_rmse": rmse(target, fixed_prediction),
                "mean_residual_std": float(residual_std.mean()),
                "p90_residual_std": float(np.quantile(residual_std, 0.90)),
            }
        )
        print(
            f"Fold {fold}: single={fold_rows[-1]['single_fixed_blend_rmse']:.5f}, "
            f"bagged={fold_rows[-1]['fixed_ensemble_rmse']:.5f}",
            flush=True,
        )

    baseline = eval_data["baseline_tvt"].to_numpy()
    target = eval_data["tvt"].to_numpy()
    single_prediction = baseline + DEFAULT_BLEND * single_residual_oof
    fixed_prediction = baseline + DEFAULT_BLEND * residual_mean_oof
    alpha_grid, best = evaluate_alpha_grid(
        baseline,
        target,
        residual_mean_oof,
        residual_std_oof,
    )

    return {
        "method": "bagged_shrink_residual_hgb",
        "base_method": "residual_correction_hgb",
        "validation_type": "grouped_by_well_sampled_rows",
        "train_wells": int(data["well_id"].nunique()),
        "validation_rows": int(len(eval_data)),
        "train_sample_per_well_per_model": TRAIN_SAMPLE_PER_WELL,
        "validation_sample_per_well": VALIDATION_SAMPLE_PER_WELL,
        "ensemble_seeds": list(ENSEMBLE_SEEDS),
        "residual_clip": RESIDUAL_CLIP,
        "baseline_rmse": rmse(target, baseline),
        "single_model_blend": DEFAULT_BLEND,
        "single_model_rmse": rmse(target, single_prediction),
        "fixed_ensemble_blend": DEFAULT_BLEND,
        "fixed_ensemble_rmse": rmse(target, fixed_prediction),
        "absolute_rmse_improvement_vs_single_model": rmse(target, single_prediction)
        - rmse(target, fixed_prediction),
        "relative_rmse_improvement_pct_vs_single_model": (
            (rmse(target, single_prediction) - rmse(target, fixed_prediction))
            / rmse(target, single_prediction)
            * 100.0
        ),
        "best_alpha_rule": best,
        "best_alpha_rmse": float(best["rmse"]),
        "absolute_rmse_improvement_vs_fixed_ensemble": rmse(target, fixed_prediction)
        - float(best["rmse"]),
        "relative_rmse_improvement_pct_vs_fixed_ensemble": (
            (rmse(target, fixed_prediction) - float(best["rmse"]))
            / rmse(target, fixed_prediction)
            * 100.0
        ),
        "mean_residual_std": float(residual_std_oof.mean()),
        "p90_residual_std": float(np.quantile(residual_std_oof, 0.90)),
        "folds": fold_rows,
        "alpha_grid": alpha_grid,
    }


def train_final_models(data: pd.DataFrame) -> list[HistGradientBoostingRegressor]:
    all_indices = np.arange(len(data))
    models = []

    for seed in ENSEMBLE_SEEDS:
        sampled_indices = sample_rows_per_well(
            data,
            all_indices,
            TRAIN_SAMPLE_PER_WELL,
            seed=seed,
        )
        model = make_model(seed)
        model.fit(data.loc[sampled_indices, FEATURE_COLUMNS], data.loc[sampled_indices, "residual"])
        models.append(model)

    return models


def build_submission(
    models: list[HistGradientBoostingRegressor],
    alpha_rule: dict,
) -> pd.DataFrame:
    submission = pd.read_csv(SUBMISSION_TEMPLATE)
    parsed = submission["id"].map(split_submission_id)
    submission["well_id"] = parsed.map(lambda value: value[0])
    submission["row_index"] = parsed.map(lambda value: value[1])

    prediction_frames = []
    for well_id in sorted(submission["well_id"].unique()):
        path = TEST_DIR / f"{well_id}__horizontal_well.csv"
        features = well_features(path, include_target=False)
        residual_mean, residual_std = predict_ensemble(models, features)
        if alpha_rule["mode"] == "dynamic":
            alpha = dynamic_alpha(
                residual_std=residual_std,
                base_alpha=float(alpha_rule["base_alpha"]),
                shrink_strength=float(alpha_rule["shrink_strength"]),
                scale=float(alpha_rule["std_scale"]),
                minimum_alpha=float(alpha_rule["minimum_alpha"]),
            )
            blended_residual = alpha * residual_mean
        else:
            blended_residual = OUTPUT_BLEND * residual_mean
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
    metrics = validate_bagged_shrinkage(training_features)
    models = train_final_models(training_features)
    submission = build_submission(models, metrics["best_alpha_rule"])

    metrics_path = REPORTS_DIR / "bagged-shrink-residual-metrics.json"
    submission_path = SUBMISSIONS_DIR / "bagged_shrink_residual_submission.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    submission.to_csv(submission_path, index=False)

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote submission: {submission_path}")
    print(f"Baseline sampled RMSE: {metrics['baseline_rmse']:.5f}")
    print(f"Fixed ensemble RMSE: {metrics['fixed_ensemble_rmse']:.5f}")
    print(f"Best alpha rule RMSE: {metrics['best_alpha_rmse']:.5f}")
    print(
        "Dynamic improvement vs fixed ensemble: "
        f"{metrics['relative_rmse_improvement_pct_vs_fixed_ensemble']:.3f}%"
    )


if __name__ == "__main__":
    main()
