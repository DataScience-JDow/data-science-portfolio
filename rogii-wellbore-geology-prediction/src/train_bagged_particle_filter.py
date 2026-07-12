from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from train_bagged_shrink_residual import sample_rows_per_well
from train_lightweight_particle_filter import (
    PF_FEATURE_COLUMNS,
    RESIDUAL_CLIP,
    add_pf_features,
    prediction_frame_for_well,
)
from train_residual_correction import (
    REPORTS_DIR,
    SUBMISSION_TEMPLATE,
    SUBMISSIONS_DIR,
    TEST_DIR,
    load_training_features,
    rmse,
    split_submission_id,
    well_features,
)


PF_VALIDATION_PATH = REPORTS_DIR / "lightweight-particle-filter-validation.csv"
TRAIN_SAMPLE_PER_WELL = 900
VALIDATION_SAMPLE_PER_WELL = 800
ENSEMBLE_SEEDS = (17, 43, 89, 131, 197)
DEFAULT_BLEND = 0.60


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
    raw = np.vstack([model.predict(frame[PF_FEATURE_COLUMNS]) for model in models])
    clipped = np.clip(raw, -RESIDUAL_CLIP, RESIDUAL_CLIP)
    return clipped.mean(axis=0), clipped.std(axis=0)


def alpha_sweep(
    baseline: np.ndarray,
    target: np.ndarray,
    residual: np.ndarray,
) -> tuple[list[dict], dict]:
    rows = []
    for alpha in np.round(np.arange(0.40, 0.91, 0.02), 2):
        prediction = baseline + alpha * residual
        rows.append({"alpha": float(alpha), "rmse": rmse(target, prediction)})
    return rows, min(rows, key=lambda row: row["rmse"])


def validate_bagged_pf(data: pd.DataFrame) -> dict:
    all_indices = np.arange(len(data))
    eval_indices = sample_rows_per_well(
        data,
        all_indices,
        VALIDATION_SAMPLE_PER_WELL,
        seed=None,
    )
    eval_data = data.loc[eval_indices].reset_index(drop=True)
    groups = eval_data["well_id"].to_numpy()

    single_oof = np.zeros(len(eval_data), dtype=np.float32)
    bagged_oof = np.zeros(len(eval_data), dtype=np.float32)
    ensemble_std_oof = np.zeros(len(eval_data), dtype=np.float32)
    fold_rows = []

    splitter = GroupKFold(n_splits=5)
    for fold, (train_group_idx, valid_idx) in enumerate(
        splitter.split(eval_data[PF_FEATURE_COLUMNS], eval_data["residual"], groups),
        start=1,
    ):
        train_wells = set(eval_data.iloc[train_group_idx]["well_id"])
        train_indices = data.index[data["well_id"].isin(train_wells)].to_numpy()

        single_indices = sample_rows_per_well(
            data,
            train_indices,
            TRAIN_SAMPLE_PER_WELL,
            seed=None,
        )
        single_model = make_model(43)
        single_model.fit(
            data.loc[single_indices, PF_FEATURE_COLUMNS],
            data.loc[single_indices, "residual"],
        )

        models = []
        sample_sizes = []
        for seed in ENSEMBLE_SEEDS:
            sampled_indices = sample_rows_per_well(
                data,
                train_indices,
                TRAIN_SAMPLE_PER_WELL,
                seed=seed + fold * 1000,
            )
            model = make_model(seed)
            model.fit(
                data.loc[sampled_indices, PF_FEATURE_COLUMNS],
                data.loc[sampled_indices, "residual"],
            )
            models.append(model)
            sample_sizes.append(int(len(sampled_indices)))

        valid_frame = eval_data.iloc[valid_idx]
        single_oof[valid_idx] = np.clip(
            single_model.predict(valid_frame[PF_FEATURE_COLUMNS]),
            -RESIDUAL_CLIP,
            RESIDUAL_CLIP,
        )
        bagged_mean, bagged_std = predict_ensemble(models, valid_frame)
        bagged_oof[valid_idx] = bagged_mean
        ensemble_std_oof[valid_idx] = bagged_std

        baseline = valid_frame["baseline_tvt"].to_numpy(dtype=float)
        target = valid_frame["tvt"].to_numpy(dtype=float)
        fold_rows.append(
            {
                "fold": fold,
                "validation_rows": int(len(valid_idx)),
                "train_sample_rows_per_model": sample_sizes,
                "single_rmse": rmse(target, baseline + DEFAULT_BLEND * single_oof[valid_idx]),
                "bagged_rmse": rmse(target, baseline + DEFAULT_BLEND * bagged_mean),
                "mean_ensemble_std": float(bagged_std.mean()),
                "p90_ensemble_std": float(np.quantile(bagged_std, 0.90)),
            }
        )
        print(
            f"Fold {fold}: single={fold_rows[-1]['single_rmse']:.5f}, "
            f"bagged={fold_rows[-1]['bagged_rmse']:.5f}",
            flush=True,
        )

    baseline = eval_data["baseline_tvt"].to_numpy(dtype=float)
    target = eval_data["tvt"].to_numpy(dtype=float)
    single_prediction = baseline + DEFAULT_BLEND * single_oof
    bagged_prediction = baseline + DEFAULT_BLEND * bagged_oof
    sweep, best = alpha_sweep(baseline, target, bagged_oof)

    return {
        "method": "bagged_particle_filter_residual_hgb",
        "validation_type": "grouped_by_well_sampled_rows",
        "validation_rows": int(len(eval_data)),
        "train_wells": int(data["well_id"].nunique()),
        "train_sample_per_well": TRAIN_SAMPLE_PER_WELL,
        "validation_sample_per_well": VALIDATION_SAMPLE_PER_WELL,
        "ensemble_seeds": list(ENSEMBLE_SEEDS),
        "default_blend": DEFAULT_BLEND,
        "single_model_rmse": rmse(target, single_prediction),
        "bagged_default_blend_rmse": rmse(target, bagged_prediction),
        "absolute_improvement_vs_single": rmse(target, single_prediction)
        - rmse(target, bagged_prediction),
        "relative_improvement_pct_vs_single": (
            (rmse(target, single_prediction) - rmse(target, bagged_prediction))
            / rmse(target, single_prediction)
            * 100.0
        ),
        "best_alpha": best,
        "alpha_sweep": sweep,
        "mean_ensemble_std": float(ensemble_std_oof.mean()),
        "p90_ensemble_std": float(np.quantile(ensemble_std_oof, 0.90)),
        "folds": fold_rows,
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
        model.fit(
            data.loc[sampled_indices, PF_FEATURE_COLUMNS],
            data.loc[sampled_indices, "residual"],
        )
        models.append(model)
    return models


def build_test_features() -> pd.DataFrame:
    submission = pd.read_csv(SUBMISSION_TEMPLATE)
    parsed = submission["id"].map(split_submission_id)
    test_wells = sorted(parsed.map(lambda value: value[0]).unique())
    base_frames = []
    pf_frames = []
    for well_id in test_wells:
        base_frames.append(well_features(TEST_DIR / f"{well_id}__horizontal_well.csv", False))
        pf_frames.append(prediction_frame_for_well(well_id, "test", include_target=False))
    return add_pf_features(
        pd.concat(base_frames, ignore_index=True),
        pd.concat(pf_frames, ignore_index=True),
    )


def build_submission(
    models: list[HistGradientBoostingRegressor],
    test_features: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    submission = pd.read_csv(SUBMISSION_TEMPLATE)
    parsed = submission["id"].map(split_submission_id)
    submission["well_id"] = parsed.map(lambda value: value[0])
    submission["row_index"] = parsed.map(lambda value: value[1])

    residual_mean, _ = predict_ensemble(models, test_features)
    predictions = test_features[["well_id", "row_index", "baseline_tvt"]].copy()
    predictions["tvt_prediction"] = predictions["baseline_tvt"] + alpha * residual_mean
    output = submission.merge(
        predictions[["well_id", "row_index", "tvt_prediction"]],
        on=["well_id", "row_index"],
        how="left",
    )
    if output["tvt_prediction"].isna().any():
        raise ValueError(f"Submission has {int(output['tvt_prediction'].isna().sum())} missing predictions.")
    output["tvt"] = output["tvt_prediction"].astype(float)
    return output[["id", "tvt"]]


def main() -> None:
    if not PF_VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Run train_lightweight_particle_filter.py first; missing {PF_VALIDATION_PATH}."
        )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading residual and particle-filter features...", flush=True)
    base_features = load_training_features()
    pf_predictions = pd.read_csv(PF_VALIDATION_PATH)
    data = add_pf_features(base_features, pf_predictions)

    metrics = validate_bagged_pf(data)
    models = train_final_models(data)
    test_features = build_test_features()
    submission = build_submission(models, test_features, float(metrics["best_alpha"]["alpha"]))

    metrics_path = REPORTS_DIR / "bagged-particle-filter-metrics.json"
    submission_path = SUBMISSIONS_DIR / "bagged_particle_filter_submission.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    submission.to_csv(submission_path, index=False)
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote submission: {submission_path}")
    print(f"Single PF-feature RMSE: {metrics['single_model_rmse']:.5f}")
    print(f"Bagged PF-feature RMSE: {metrics['bagged_default_blend_rmse']:.5f}")
    print(
        f"Best alpha: {metrics['best_alpha']['alpha']:.2f} -> "
        f"{metrics['best_alpha']['rmse']:.5f}"
    )


if __name__ == "__main__":
    main()
