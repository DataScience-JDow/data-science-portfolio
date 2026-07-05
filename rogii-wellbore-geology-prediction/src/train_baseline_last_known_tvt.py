from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
TRAIN_DIR = RAW_DIR / "train"
TEST_DIR = RAW_DIR / "test"
SUBMISSION_TEMPLATE = RAW_DIR / "sample_submission.csv"
SUBMISSIONS_DIR = PROJECT_DIR / "submissions"
REPORTS_DIR = PROJECT_DIR / "reports"


def split_submission_id(value: str) -> tuple[str, int]:
    well_id, row_index = value.rsplit("_", 1)
    return well_id, int(row_index)


def target_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["TVT_input"].isna()


def last_known_tvt(frame: pd.DataFrame) -> float:
    known = frame["TVT_input"].dropna()
    if known.empty:
        raise ValueError("Cannot predict a well with no known TVT_input values.")
    return float(known.iloc[-1])


def validate_on_train() -> dict:
    rows = []
    squared_error_sum = 0.0
    prediction_count = 0

    for path in sorted(TRAIN_DIR.glob("*__horizontal_well.csv")):
        well_id = path.name.split("__", 1)[0]
        frame = pd.read_csv(path, usecols=["TVT", "TVT_input"])
        mask = target_mask(frame)

        if not mask.any():
            continue

        prediction = last_known_tvt(frame)
        truth = frame.loc[mask, "TVT"].to_numpy(dtype=float)
        errors = truth - prediction
        squared_error = float(np.dot(errors, errors))
        rmse = float(np.sqrt(np.mean(errors * errors)))

        squared_error_sum += squared_error
        prediction_count += len(errors)
        rows.append(
            {
                "well_id": well_id,
                "known_rows": int((~mask).sum()),
                "target_rows": int(mask.sum()),
                "last_known_tvt": prediction,
                "rmse": rmse,
                "mean_error": float(np.mean(errors)),
                "max_abs_error": float(np.max(np.abs(errors))),
            }
        )

    if prediction_count == 0:
        raise ValueError("No training target rows found for validation.")

    per_well = pd.DataFrame(rows)
    return {
        "method": "baseline_last_known_tvt",
        "train_wells": int(len(per_well)),
        "validation_rows": int(prediction_count),
        "overall_rmse": float(np.sqrt(squared_error_sum / prediction_count)),
        "per_well_rmse_mean": float(per_well["rmse"].mean()),
        "per_well_rmse_median": float(per_well["rmse"].median()),
        "per_well_rmse_p90": float(per_well["rmse"].quantile(0.90)),
        "per_well_rmse_p95": float(per_well["rmse"].quantile(0.95)),
        "per_well_rmse_max": float(per_well["rmse"].max()),
        "worst_wells": per_well.sort_values("rmse", ascending=False)
        .head(10)
        .to_dict(orient="records"),
    }


def build_submission() -> pd.DataFrame:
    submission = pd.read_csv(SUBMISSION_TEMPLATE)
    parsed = submission["id"].map(split_submission_id)
    submission["well_id"] = parsed.map(lambda value: value[0])
    submission["row_index"] = parsed.map(lambda value: value[1])

    predictions = {}
    for well_id in sorted(submission["well_id"].unique()):
        path = TEST_DIR / f"{well_id}__horizontal_well.csv"
        frame = pd.read_csv(path, usecols=["TVT_input"])
        prediction = last_known_tvt(frame)
        predictions[well_id] = prediction

    submission["tvt"] = submission["well_id"].map(predictions).astype(float)
    return submission[["id", "tvt"]]


def main() -> None:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = validate_on_train()
    submission = build_submission()

    output_path = SUBMISSIONS_DIR / "last_known_tvt_submission.csv"
    metrics_path = REPORTS_DIR / "baseline-metrics.json"

    submission.to_csv(output_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"Wrote submission: {output_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Local validation RMSE: {metrics['overall_rmse']:.5f}")


if __name__ == "__main__":
    main()
