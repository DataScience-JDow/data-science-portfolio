from pathlib import Path

import pandas as pd


LOCAL_INPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

def find_raw_dir() -> Path:
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = sorted(kaggle_input.glob("**/sample_submission.csv"))
        if matches:
            return matches[0].parent
    return LOCAL_INPUT_DIR


RAW_DIR = find_raw_dir()
TEST_DIR = RAW_DIR / "test"
SUBMISSION_TEMPLATE = RAW_DIR / "sample_submission.csv"
OUTPUT_PATH = Path("/kaggle/working/submission.csv")

if not OUTPUT_PATH.parent.exists():
    OUTPUT_PATH = Path(__file__).resolve().parents[1] / "submissions" / "submission.csv"


def split_submission_id(value: str) -> tuple[str, int]:
    well_id, row_index = value.rsplit("_", 1)
    return well_id, int(row_index)


def last_known_tvt(frame: pd.DataFrame) -> float:
    known = frame["TVT_input"].dropna()
    if known.empty:
        raise ValueError("Cannot predict a well with no known TVT_input values.")
    return float(known.iloc[-1])


def main() -> None:
    submission = pd.read_csv(SUBMISSION_TEMPLATE)
    parsed = submission["id"].map(split_submission_id)
    submission["well_id"] = parsed.map(lambda value: value[0])

    predictions = {}
    for well_id in sorted(submission["well_id"].unique()):
        path = TEST_DIR / f"{well_id}__horizontal_well.csv"
        frame = pd.read_csv(path, usecols=["TVT_input"])
        predictions[well_id] = last_known_tvt(frame)

    submission["tvt"] = submission["well_id"].map(predictions).astype(float)
    output = submission[["id", "tvt"]]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH} with {len(output):,} rows")


if __name__ == "__main__":
    main()
