from __future__ import annotations

import argparse
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_JSON = "Data/TranscriptionData/final_classification/manual_revision.json"


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


def read_binary_source(source: str | Path) -> bytes:
    source_str = str(source)

    if is_url(source_str):
        raw_url = to_raw_github_url(source_str)
        request = Request(raw_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request) as response:
            return response.read()

    return Path(source).read_bytes()


def normalize_video_id(value: Any) -> str:
    text = str(value).strip()
    text = Path(text).name
    text = re.sub(r"\.(mp4|avi|mov|mkv|webm)$", "", text, flags=re.IGNORECASE)
    return text.lower()


def load_classification_map(json_source: str | Path) -> dict[str, str]:
    data = json.loads(read_binary_source(json_source).decode("utf-8"))

    if not isinstance(data, dict):
        raise ValueError("Il JSON deve essere un dizionario con chiavi tipo Video1.mp4.")

    mapping: dict[str, str] = {}

    for raw_video_id, payload in data.items():
        normalized_video_id = normalize_video_id(raw_video_id)

        if isinstance(payload, dict):
            classification = payload.get("classification")
        else:
            classification = None

        if classification is not None:
            mapping[normalized_video_id] = str(classification).strip()

    return mapping


def insert_classification_after_video_id(df: pd.DataFrame) -> pd.DataFrame:
    if "classification" not in df.columns:
        columns = list(df.columns)
        video_idx = columns.index("video_id")
        columns.insert(video_idx + 1, "classification")
        df["classification"] = pd.NA
        return df[columns]

    columns = list(df.columns)
    columns.remove("classification")
    video_idx = columns.index("video_id")
    columns.insert(video_idx + 1, "classification")
    return df[columns]


def enrich_sheet(df: pd.DataFrame, classification_map: dict[str, str]) -> pd.DataFrame:
    if "video_id" not in df.columns:
        return df

    df = df.copy()
    df = insert_classification_after_video_id(df)

    normalized_ids = df["video_id"].map(normalize_video_id)
    mapped = normalized_ids.map(classification_map)

    df["classification"] = mapped.combine_first(df["classification"])

    return df


def enrich_workbook(input_xlsx: str | Path, json_source: str | Path, output_xlsx: str | Path) -> None:
    xlsx_bytes = read_binary_source(input_xlsx)
    sheets = pd.read_excel(BytesIO(xlsx_bytes), sheet_name=None)
    classification_map = load_classification_map(json_source)

    enriched_sheets = {
        sheet_name: enrich_sheet(df, classification_map)
        for sheet_name, df in sheets.items()
    }

    output_xlsx = Path(output_xlsx)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        for sheet_name, df in enriched_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Classificazioni trovate nel JSON: {len(classification_map)}")
    print(f"Workbook arricchito salvato in: {output_xlsx.resolve()}")


def default_output_name(input_xlsx: str | Path) -> str:
    input_str = str(input_xlsx)

    if is_url(input_str):
        name = Path(urlparse(input_str).path).name
    else:
        name = Path(input_xlsx).name

    stem = Path(name).stem
    return f"{stem}_with_classification.xlsx"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-xlsx", required=True)
    parser.add_argument("--classification-json", default=DEFAULT_JSON)
    parser.add_argument("--output-xlsx", default=None)
    args = parser.parse_args()

    output_xlsx = args.output_xlsx or default_output_name(args.input_xlsx)

    enrich_workbook(
        input_xlsx=args.input_xlsx,
        json_source=args.classification_json,
        output_xlsx=output_xlsx,
    )


if __name__ == "__main__":
    main()
