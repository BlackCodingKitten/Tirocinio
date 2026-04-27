from __future__ import annotations

import argparse
import re
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd


VIDEO_METRICS_SHEET = "video_metrics"
OUTPUT_STEM = "intersection_sets"


def is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def to_raw_github_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc != "github.com":
        return url

    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 5 and parts[2] == "blob":
        user, repo, _, branch = parts[:4]
        file_path = "/".join(parts[4:])
        return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{file_path}"

    return url


def read_excel_source(source: str | Path, sheet_name: str) -> pd.DataFrame:
    source_str = str(source)

    if is_url(source_str):
        raw_url = to_raw_github_url(source_str)
        request = Request(raw_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request) as response:
            content = response.read()
        return pd.read_excel(BytesIO(content), sheet_name=sheet_name)

    return pd.read_excel(source_str, sheet_name=sheet_name)


def normalize_experiment(value: Any) -> str:
    text = str(value).strip()
    if text.upper() == "1A":
        return "1A"
    return text


def normalize_video_id(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"\.(mp4|avi|mov|mkv|webm)$", "", text, flags=re.IGNORECASE)
    return text


def first_valid_classification(values: pd.Series) -> str:
    for value in values:
        if pd.notna(value):
            text = str(value).strip()
            if text and text.lower() not in {"nan", "none"}:
                return text
    return "NON_DISPONIBILE"


def load_video_metrics(input_xlsx: str | Path) -> pd.DataFrame:
    df = read_excel_source(input_xlsx, VIDEO_METRICS_SHEET)

    required = {"video_id", "experiment", "accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti in {VIDEO_METRICS_SHEET}: {sorted(missing)}")

    df = df.copy()
    df["video_id"] = df["video_id"].map(normalize_video_id)
    df["experiment"] = df["experiment"].map(normalize_experiment)
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")

    if "classification" not in df.columns:
        df["classification"] = pd.NA

    df = df.dropna(subset=["video_id", "experiment", "accuracy"])
    return df


def build_accuracy_table(video_metrics: pd.DataFrame) -> pd.DataFrame:
    accuracy_table = (
        video_metrics
        .pivot_table(
            index="video_id",
            columns="experiment",
            values="accuracy",
            aggfunc="mean",
        )
        .rename(columns=str)
        .reset_index()
    )

    required_experiments = {"1A", "2", "3", "4"}
    missing = required_experiments - set(accuracy_table.columns)
    if missing:
        raise ValueError(f"Esperimenti mancanti in {VIDEO_METRICS_SHEET}: {sorted(missing)}")

    classification_table = (
        video_metrics
        .groupby("video_id", as_index=False)["classification"]
        .agg(first_valid_classification)
    )

    result = accuracy_table.merge(classification_table, on="video_id", how="left")

    result = result.rename(
        columns={
            "1A": "accuracy_1A",
            "2": "accuracy_2",
            "3": "accuracy_3",
            "4": "accuracy_4",
        }
    )

    result["delta_1A_to_2"] = result["accuracy_2"] - result["accuracy_1A"]
    result["delta_3_to_4"] = result["accuracy_4"] - result["accuracy_3"]

    return result[
        [
            "video_id",
            "classification",
            "accuracy_1A",
            "accuracy_2",
            "accuracy_3",
            "accuracy_4",
            "delta_1A_to_2",
            "delta_3_to_4",
        ]
    ]


def build_sets(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    improved_3_to_4 = (
        table[table["delta_3_to_4"] > 0]
        .sort_values(["delta_3_to_4", "video_id"], ascending=[False, True])
        .reset_index(drop=True)
    )

    improved_1a_to_2 = (
        table[table["delta_1A_to_2"] > 0]
        .sort_values(["delta_1A_to_2", "video_id"], ascending=[False, True])
        .reset_index(drop=True)
    )

    intersection = (
        table[(table["delta_3_to_4"] > 0) & (table["delta_1A_to_2"] > 0)]
        .sort_values(["delta_1A_to_2", "delta_3_to_4", "video_id"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    long_df = pd.concat(
        [
            improved_3_to_4.assign(set_name="improved_3_to_4"),
            improved_1a_to_2.assign(set_name="improved_1A_to_2"),
            intersection.assign(set_name="intersection_improved_both"),
        ],
        ignore_index=True,
    )

    ordered_columns = [
        "set_name",
        "video_id",
        "classification",
        "accuracy_1A",
        "accuracy_2",
        "accuracy_3",
        "accuracy_4",
        "delta_1A_to_2",
        "delta_3_to_4",
    ]

    return (
        improved_3_to_4[ordered_columns[1:]],
        improved_1a_to_2[ordered_columns[1:]],
        intersection[ordered_columns[1:]],
        long_df[ordered_columns],
    )


def save_outputs(
    improved_3_to_4: pd.DataFrame,
    improved_1a_to_2: pd.DataFrame,
    intersection: pd.DataFrame,
    long_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = output_dir / f"{OUTPUT_STEM}.pkl"
    xlsx_path = output_dir / f"{OUTPUT_STEM}.xlsx"

    long_df.to_pickle(pkl_path)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        long_df.to_excel(writer, sheet_name="all_sets_long", index=False)
        improved_3_to_4.to_excel(writer, sheet_name="improved_3_to_4", index=False)
        improved_1a_to_2.to_excel(writer, sheet_name="improved_1A_to_2", index=False)
        intersection.to_excel(writer, sheet_name="intersection", index=False)

    print(f"Video migliorati 3→4: {len(improved_3_to_4)}")
    print(f"Video migliorati 1A→2: {len(improved_1a_to_2)}")
    print(f"Intersezione: {len(intersection)}")
    print(f"DataFrame pandas salvato in: {pkl_path.resolve()}")
    print(f"Excel salvato in: {xlsx_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-xlsx",
        default="https://github.com/BlackCodingKitten/Tirocinio/blob/main/1A_2_3_4_merge.xlsx",
    )
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    video_metrics = load_video_metrics(args.input_xlsx)
    table = build_accuracy_table(video_metrics)
    improved_3_to_4, improved_1a_to_2, intersection, long_df = build_sets(table)

    save_outputs(
        improved_3_to_4=improved_3_to_4,
        improved_1a_to_2=improved_1a_to_2,
        intersection=intersection,
        long_df=long_df,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
