from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold

from train_bagged_particle_filter import (
    PF_VALIDATION_PATH,
    TRAIN_SAMPLE_PER_WELL,
    VALIDATION_SAMPLE_PER_WELL,
    alpha_sweep,
    make_model,
)
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
    TRAIN_DIR,
    load_training_features,
    rmse,
    split_submission_id,
    well_features,
)


FORMATION_COLUMNS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
CONTROLS_PER_WELL = 30
NEIGHBORS_TO_QUERY = 48
NEIGHBORS_TO_AVERAGE = 12
DISTANCE_FLOOR = 250.0
OFFSET_FEATURE_COLUMNS = PF_FEATURE_COLUMNS + [
    *(f"offset_depth_{column.lower()}" for column in FORMATION_COLUMNS),
    *(f"offset_delta_{column.lower()}" for column in FORMATION_COLUMNS),
    "offset_depth_mean",
    "offset_delta_mean",
    "offset_neighbor_distance",
]


def target_geometry_for_well(path) -> pd.DataFrame:
    well_id = path.name.split("__", 1)[0]
    frame = pd.read_csv(path, usecols=["X", "Y", "TVT_input"])
    target_mask = frame["TVT_input"].isna().to_numpy()
    target_indices = np.flatnonzero(target_mask)
    last_known_index = np.flatnonzero(~target_mask)[-1]
    return pd.DataFrame(
        {
            "well_id": well_id,
            "row_index": target_indices,
            "query_x": frame.loc[target_indices, "X"].to_numpy(dtype=np.float32),
            "query_y": frame.loc[target_indices, "Y"].to_numpy(dtype=np.float32),
            "start_x": np.float32(frame.loc[last_known_index, "X"]),
            "start_y": np.float32(frame.loc[last_known_index, "Y"]),
        }
    )


def add_target_geometry(data: pd.DataFrame, base_dir) -> pd.DataFrame:
    geometry = pd.concat(
        [
            target_geometry_for_well(base_dir / f"{well_id}__horizontal_well.csv")
            for well_id in sorted(data["well_id"].unique())
        ],
        ignore_index=True,
    )
    merged = data.merge(geometry, on=["well_id", "row_index"], how="left")
    if merged["query_x"].isna().any():
        raise ValueError(f"Missing geometry for {int(merged['query_x'].isna().sum())} rows.")
    return merged


def build_control_points(well_ids: set[str] | None = None) -> pd.DataFrame:
    frames = []
    paths = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
    for path in paths:
        well_id = path.name.split("__", 1)[0]
        if well_ids is not None and well_id not in well_ids:
            continue
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


def validate_offset_model(data: pd.DataFrame) -> dict:
    eval_indices = sample_rows_per_well(
        data,
        np.arange(len(data)),
        VALIDATION_SAMPLE_PER_WELL,
        seed=None,
    )
    eval_data = data.loc[eval_indices].reset_index(drop=True)
    groups = eval_data["well_id"].to_numpy()
    pf_oof = np.zeros(len(eval_data), dtype=np.float32)
    offset_oof = np.zeros(len(eval_data), dtype=np.float32)
    fold_rows = []

    splitter = GroupKFold(n_splits=5)
    for fold, (train_group_idx, valid_idx) in enumerate(
        splitter.split(eval_data[PF_FEATURE_COLUMNS], eval_data["residual"], groups),
        start=1,
    ):
        train_wells = set(eval_data.iloc[train_group_idx]["well_id"])
        train_indices = data.index[data["well_id"].isin(train_wells)].to_numpy()
        sampled_indices = sample_rows_per_well(
            data,
            train_indices,
            TRAIN_SAMPLE_PER_WELL,
            seed=None,
        )
        controls = build_control_points(train_wells)
        offset_train = add_offset_features(data.loc[sampled_indices], controls)
        valid = eval_data.iloc[valid_idx]
        offset_valid = add_offset_features(valid, controls)

        pf_model = make_model(43)
        pf_model.fit(
            data.loc[sampled_indices, PF_FEATURE_COLUMNS],
            data.loc[sampled_indices, "residual"],
        )
        offset_model = make_model(59)
        offset_model.fit(
            offset_train[OFFSET_FEATURE_COLUMNS],
            offset_train["residual"],
        )
        pf_oof[valid_idx] = np.clip(
            pf_model.predict(valid[PF_FEATURE_COLUMNS]),
            -RESIDUAL_CLIP,
            RESIDUAL_CLIP,
        )
        offset_oof[valid_idx] = np.clip(
            offset_model.predict(offset_valid[OFFSET_FEATURE_COLUMNS]),
            -RESIDUAL_CLIP,
            RESIDUAL_CLIP,
        )
        baseline = valid["baseline_tvt"].to_numpy(dtype=float)
        target = valid["tvt"].to_numpy(dtype=float)
        fold_rows.append(
            {
                "fold": fold,
                "validation_rows": int(len(valid_idx)),
                "controls": int(len(controls)),
                "pf_rmse_alpha_0_60": rmse(target, baseline + 0.60 * pf_oof[valid_idx]),
                "offset_rmse_alpha_0_60": rmse(
                    target,
                    baseline + 0.60 * offset_oof[valid_idx],
                ),
                "mean_neighbor_distance": float(offset_valid["offset_neighbor_distance"].mean()),
            }
        )
        print(
            f"Fold {fold}: PF={fold_rows[-1]['pf_rmse_alpha_0_60']:.5f}, "
            f"offset={fold_rows[-1]['offset_rmse_alpha_0_60']:.5f}",
            flush=True,
        )

    baseline = eval_data["baseline_tvt"].to_numpy(dtype=float)
    target = eval_data["tvt"].to_numpy(dtype=float)
    pf_sweep, pf_best = alpha_sweep(baseline, target, pf_oof)
    offset_sweep, offset_best = alpha_sweep(baseline, target, offset_oof)
    blend_rows = []
    for offset_weight in np.round(np.arange(0.0, 1.01, 0.10), 2):
        residual = (1.0 - offset_weight) * pf_oof + offset_weight * offset_oof
        _, best = alpha_sweep(baseline, target, residual)
        blend_rows.append(
            {
                "offset_weight": float(offset_weight),
                "alpha": best["alpha"],
                "rmse": best["rmse"],
            }
        )
    return {
        "method": "offset_formation_particle_filter_residual_hgb",
        "validation_type": "grouped_by_well_with_heldout_offset_surfaces",
        "validation_rows": int(len(eval_data)),
        "train_wells": int(data["well_id"].nunique()),
        "controls_per_well": CONTROLS_PER_WELL,
        "neighbors_to_average": NEIGHBORS_TO_AVERAGE,
        "pf_best_alpha": pf_best,
        "offset_best_alpha": offset_best,
        "best_pf_offset_blend": min(blend_rows, key=lambda row: row["rmse"]),
        "pf_alpha_sweep": pf_sweep,
        "offset_alpha_sweep": offset_sweep,
        "blend_sweep": blend_rows,
        "folds": fold_rows,
    }


def build_test_data() -> pd.DataFrame:
    submission = pd.read_csv(SUBMISSION_TEMPLATE)
    parsed = submission["id"].map(split_submission_id)
    wells = sorted(parsed.map(lambda value: value[0]).unique())
    base = pd.concat(
        [well_features(TEST_DIR / f"{well_id}__horizontal_well.csv", False) for well_id in wells],
        ignore_index=True,
    )
    pf = pd.concat(
        [prediction_frame_for_well(well_id, "test", include_target=False) for well_id in wells],
        ignore_index=True,
    )
    return add_target_geometry(add_pf_features(base, pf), TEST_DIR)


def train_final_model(data: pd.DataFrame, controls: pd.DataFrame):
    sampled_indices = sample_rows_per_well(
        data,
        np.arange(len(data)),
        TRAIN_SAMPLE_PER_WELL,
        seed=None,
    )
    offset_train = add_offset_features(data.loc[sampled_indices], controls)
    model = make_model(59)
    model.fit(
        offset_train[OFFSET_FEATURE_COLUMNS],
        offset_train["residual"],
    )
    return model


def build_submission(model, test_data: pd.DataFrame, alpha: float) -> pd.DataFrame:
    residual = np.clip(
        model.predict(test_data[OFFSET_FEATURE_COLUMNS]),
        -RESIDUAL_CLIP,
        RESIDUAL_CLIP,
    )
    predictions = test_data[["well_id", "row_index", "baseline_tvt"]].copy()
    predictions["tvt_prediction"] = predictions["baseline_tvt"] + alpha * residual

    submission = pd.read_csv(SUBMISSION_TEMPLATE)
    parsed = submission["id"].map(split_submission_id)
    submission["well_id"] = parsed.map(lambda value: value[0])
    submission["row_index"] = parsed.map(lambda value: value[1])
    output = submission.merge(
        predictions[["well_id", "row_index", "tvt_prediction"]],
        on=["well_id", "row_index"],
        how="left",
    )
    if output["tvt_prediction"].isna().any():
        raise ValueError(f"Submission has {int(output['tvt_prediction'].isna().sum())} missing rows.")
    output["tvt"] = output["tvt_prediction"].astype(float)
    return output[["id", "tvt"]]


def main() -> None:
    if not PF_VALIDATION_PATH.exists():
        raise FileNotFoundError(f"Missing cached PF validation predictions: {PF_VALIDATION_PATH}")
    print("Loading PF features and target geometry...", flush=True)
    data = add_target_geometry(
        add_pf_features(load_training_features(), pd.read_csv(PF_VALIDATION_PATH)),
        TRAIN_DIR,
    )
    metrics = validate_offset_model(data)
    controls = build_control_points()
    model = train_final_model(data, controls)
    test_data = add_offset_features(build_test_data(), controls)
    submission = build_submission(
        model,
        test_data,
        float(metrics["offset_best_alpha"]["alpha"]),
    )
    metrics_path = REPORTS_DIR / "offset-formation-particle-filter-metrics.json"
    submission_path = SUBMISSIONS_DIR / "offset_formation_particle_filter_submission.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    submission.to_csv(submission_path, index=False)
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote submission: {submission_path}")
    print(f"PF best: {metrics['pf_best_alpha']}")
    print(f"Offset best: {metrics['offset_best_alpha']}")
    print(f"Best blend: {metrics['best_pf_offset_blend']}")


if __name__ == "__main__":
    main()
