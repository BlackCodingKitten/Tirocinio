from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd
from openai import OpenAI


DEFAULT_MANUAL_REVISION_JSON = (
    "Data/TranscriptionData/final_classification/manual_revision.json"
)

DEFAULT_DATASET_JSON = (
    "Data/Dataset/maia_ita_mc_by_video_category_pool.json"
)

DEFAULT_OUTPUT_XLSX = "semantic_similarity_dialogue_videos.xlsx"
DEFAULT_CACHE_JSONL = "semantic_similarity_dialogue_videos_cache.jsonl"


SYSTEM_PROMPT = """You are an expert evaluator of semantic similarity between two pieces of text."""


USER_PROMPT_TEMPLATE = """Your task is to assess how much the content of an audio transcription matches the content of a video caption.

The input texts are in Italian.

The transcription may be noisy or incomplete.

Focus ONLY on semantic meaning, not on wording or style.

Consider:

- Events and actions described

- Objects and entities mentioned

- Overall situation or context

Score definition:

- 1.0 : The transcription and the caption convey the same information

- 0.5 : They partially overlap, but important differences exist

- 0.0 : They describe completely different content

Ignore:

- Differences in wording

- Grammatical correctness

Return ONLY a number between 0 and 1 (e.g., 0.73). Do not output the reasoning.

Transcription:

{transcription}

Caption:

{caption}"""


VIDEO_EXTENSION_RE = re.compile(r"\.(mp4|avi|mov|mkv|webm)$", flags=re.IGNORECASE)


def is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def to_raw_github_url(url: str) -> str:
    """
    Converte un URL GitHub /blob/ in raw.githubusercontent.com.
    Se l'URL non è GitHub o è già raw, lo lascia invariato.
    """
    parsed = urlparse(url)

    if parsed.netloc == "raw.githubusercontent.com":
        return url

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


def load_json(source: str | Path) -> Any:
    return json.loads(read_binary_source(source).decode("utf-8"))


def normalize_video_id(value: Any) -> str:
    """
    Normalizza Video1, Video1.mp4, /path/Video1.mp4 nella stessa chiave: video1.
    """
    text = str(value).strip()
    text = Path(text).name
    text = VIDEO_EXTENSION_RE.sub("", text)
    return text.lower()


def canonical_video_id(value: Any) -> str:
    """
    Versione leggibile da mettere negli output.
    """
    text = str(value).strip()
    text = Path(text).name
    text = VIDEO_EXTENSION_RE.sub("", text)
    return text


def load_dialogue_video_ids_from_intersection_xlsx(xlsx_path: str | Path) -> pd.DataFrame:
    """
    Legge tutti i fogli del file XLSX delle intersezioni e prende i video unici
    in cui la colonna classification contiene 'dialogue'.

    Funziona anche se il file contiene più fogli come:
    - all_sets_long
    - improved_3_to_4
    - improved_1A_to_2
    - intersection
    """
    xlsx_bytes = read_binary_source(xlsx_path)
    sheets = pd.read_excel(BytesIO(xlsx_bytes), sheet_name=None)

    collected: list[pd.DataFrame] = []

    for sheet_name, df in sheets.items():
        if "video_id" not in df.columns or "classification" not in df.columns:
            continue

        tmp = df[["video_id", "classification"]].copy()
        tmp["source_sheet"] = sheet_name
        tmp["classification"] = tmp["classification"].astype("string")
        tmp = tmp[
            tmp["classification"]
            .fillna("")
            .str.contains("dialogue", case=False, regex=False)
        ]

        collected.append(tmp)

    if not collected:
        return pd.DataFrame(columns=["video_id", "video_key", "classification"])

    all_rows = pd.concat(collected, ignore_index=True)
    all_rows["video_id"] = all_rows["video_id"].map(canonical_video_id)
    all_rows["video_key"] = all_rows["video_id"].map(normalize_video_id)

    # senza duplicati: tengo la prima classification disponibile per ogni video
    unique_rows = (
        all_rows
        .drop_duplicates(subset=["video_key"])
        [["video_id", "video_key", "classification"]]
        .sort_values("video_id")
        .reset_index(drop=True)
    )

    return unique_rows


def extract_generated_transcriptions(manual_revision_data: Any) -> dict[str, str]:
    """
    Estrae generated_transcription da JSON anche se la struttura varia leggermente.

    Casi supportati:
    1) {
         "Video1.mp4": {"generated_transcription": "..."}
       }

    2) [
         {"video_id": "Video1.mp4", "generated_transcription": "..."}
       ]

    3) Strutture annidate: la funzione prova anche una visita ricorsiva.
    """
    mapping: dict[str, str] = {}

    def visit(obj: Any, possible_video_id: Any | None = None) -> None:
        if isinstance(obj, dict):
            current_video_id = (
                obj.get("video_id")
                or obj.get("video")
                or obj.get("file")
                or obj.get("filename")
                or possible_video_id
            )

            if "generated_transcription" in obj and current_video_id is not None:
                value = obj.get("generated_transcription")
                if value is not None:
                    text = str(value).strip()
                    if text:
                        mapping[normalize_video_id(current_video_id)] = text

            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    visit(value, possible_video_id=key)

        elif isinstance(obj, list):
            for item in obj:
                visit(item, possible_video_id=possible_video_id)

    visit(manual_revision_data)
    return mapping


def extract_target_captions(dataset_data: dict[str, Any]) -> pd.DataFrame:
    """
    Dal dataset MAIA estrae una riga per ogni caption target.

    Input atteso:
    {
      "Video1": {
        "CausaleEsplicita_A": {
          "pool_pos_1": {
            "0": "...",
            "1": "...",
            "target": 0
          }
        }
      }
    }

    Output:
    video_id, video_key, question_category, pool_key, target, caption
    """
    rows: list[dict[str, Any]] = []

    if not isinstance(dataset_data, dict):
        raise ValueError("Il dataset JSON deve essere un dizionario indicizzato per video.")

    for raw_video_id, video_payload in dataset_data.items():
        if not isinstance(video_payload, dict):
            continue

        readable_video_id = canonical_video_id(raw_video_id)
        video_key = normalize_video_id(raw_video_id)

        for question_category, category_payload in video_payload.items():
            if not isinstance(category_payload, dict):
                continue

            for pool_key, pool_payload in category_payload.items():
                if not isinstance(pool_payload, dict):
                    continue

                if "target" not in pool_payload:
                    continue

                target = pool_payload["target"]
                target_key = str(target)

                if target_key not in pool_payload:
                    continue

                caption = str(pool_payload[target_key]).strip()

                if not caption:
                    continue

                rows.append(
                    {
                        "video_id": readable_video_id,
                        "video_key": video_key,
                        "question_category": question_category,
                        "pool_key": pool_key,
                        "target": int(target),
                        "caption": caption,
                    }
                )

    return pd.DataFrame(rows)


def build_eval_rows(
    dialogue_videos: pd.DataFrame,
    transcription_map: dict[str, str],
    caption_table: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    dialogue_video_keys = set(dialogue_videos["video_key"])

    filtered_captions = caption_table[
        caption_table["video_key"].isin(dialogue_video_keys)
    ].copy()

    video_metadata = {
        row["video_key"]: {
            "intersection_video_id": row["video_id"],
            "classification": row["classification"],
        }
        for _, row in dialogue_videos.iterrows()
    }

    for _, caption_row in filtered_captions.iterrows():
        video_key = caption_row["video_key"]
        transcription = transcription_map.get(video_key)

        if transcription is None or not str(transcription).strip():
            rows.append(
                {
                    "video_id": video_metadata[video_key]["intersection_video_id"],
                    "classification": video_metadata[video_key]["classification"],
                    "question_category": caption_row["question_category"],
                    "pool_key": caption_row["pool_key"],
                    "target": caption_row["target"],
                    "caption": caption_row["caption"],
                    "generated_transcription": pd.NA,
                    "missing_transcription": True,
                }
            )
            continue

        rows.append(
            {
                "video_id": video_metadata[video_key]["intersection_video_id"],
                "classification": video_metadata[video_key]["classification"],
                "question_category": caption_row["question_category"],
                "pool_key": caption_row["pool_key"],
                "target": caption_row["target"],
                "caption": caption_row["caption"],
                "generated_transcription": transcription,
                "missing_transcription": False,
            }
        )

    return pd.DataFrame(rows)


def parse_score(raw_text: str) -> float:
    """
    Estrae un numero tra 0 e 1 dalla risposta del modello.
    Accetta sia 0.73 sia 0,73.
    """
    text = raw_text.strip().replace(",", ".")
    match = re.search(r"[-+]?\d*\.?\d+", text)

    if not match:
        raise ValueError(f"Impossibile estrarre uno score numerico da: {raw_text!r}")

    score = float(match.group(0))

    if score < 0.0 or score > 1.0:
        raise ValueError(f"Score fuori range [0, 1]: {score} da risposta {raw_text!r}")

    return score


def make_cache_key(
    video_id: str,
    question_category: str,
    pool_key: str,
    run_index: int,
    transcription: str,
    caption: str,
    model: str,
) -> str:
    payload = {
        "video_id": video_id,
        "question_category": question_category,
        "pool_key": pool_key,
        "run_index": run_index,
        "transcription": transcription,
        "caption": caption,
        "model": model,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_template": USER_PROMPT_TEMPLATE,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache(cache_jsonl: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(cache_jsonl)

    if not path.exists():
        return {}

    cache: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            key = record.get("cache_key")

            if key:
                cache[key] = record

    return cache


def append_cache_record(cache_jsonl: str | Path, record: dict[str, Any]) -> None:
    path = Path(cache_jsonl)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def call_gpt_similarity_score(
    client: OpenAI,
    model: str,
    transcription: str,
    caption: str,
    temperature: float | None,
    max_output_tokens: int,
) -> tuple[float, str]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        transcription=transcription,
        caption=caption,
    )

    kwargs: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "max_output_tokens": max_output_tokens,
    }

    if temperature is not None:
        kwargs["temperature"] = temperature

    response = client.responses.create(**kwargs)

    raw_text = response.output_text.strip()
    score = parse_score(raw_text)

    return score, raw_text


def evaluate_rows_with_gpt(
    rows: pd.DataFrame,
    model: str,
    runs_per_pair: int,
    temperature: float | None,
    max_output_tokens: int,
    cache_jsonl: str | Path,
    sleep_seconds: float,
) -> pd.DataFrame:
    client = OpenAI()
    cache = load_cache(cache_jsonl)

    evaluated_rows: list[dict[str, Any]] = []

    total_rows = len(rows)

    for row_number, (_, row) in enumerate(rows.iterrows(), start=1):
        output_row = row.to_dict()

        if row.get("missing_transcription", False):
            for run_index in range(1, runs_per_pair + 1):
                output_row[f"score_{run_index}"] = pd.NA
                output_row[f"raw_output_{run_index}"] = pd.NA

            output_row["mean_score"] = pd.NA
            evaluated_rows.append(output_row)
            continue

        transcription = str(row["generated_transcription"])
        caption = str(row["caption"])

        scores: list[float] = []

        print(
            f"[{row_number}/{total_rows}] "
            f"{row['video_id']} | {row['question_category']} | {row['pool_key']}"
        )

        for run_index in range(1, runs_per_pair + 1):
            cache_key = make_cache_key(
                video_id=str(row["video_id"]),
                question_category=str(row["question_category"]),
                pool_key=str(row["pool_key"]),
                run_index=run_index,
                transcription=transcription,
                caption=caption,
                model=model,
            )

            if cache_key in cache:
                cached = cache[cache_key]
                score = float(cached["score"])
                raw_text = str(cached["raw_output"])
            else:
                score, raw_text = call_gpt_similarity_score(
                    client=client,
                    model=model,
                    transcription=transcription,
                    caption=caption,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )

                cached = {
                    "cache_key": cache_key,
                    "video_id": row["video_id"],
                    "question_category": row["question_category"],
                    "pool_key": row["pool_key"],
                    "run_index": run_index,
                    "model": model,
                    "score": score,
                    "raw_output": raw_text,
                }

                append_cache_record(cache_jsonl, cached)
                cache[cache_key] = cached

                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

            output_row[f"score_{run_index}"] = score
            output_row[f"raw_output_{run_index}"] = raw_text
            scores.append(score)

        output_row["mean_score"] = sum(scores) / len(scores) if scores else pd.NA
        evaluated_rows.append(output_row)

    result = pd.DataFrame(evaluated_rows)

    score_columns = [f"score_{i}" for i in range(1, runs_per_pair + 1)]
    raw_columns = [f"raw_output_{i}" for i in range(1, runs_per_pair + 1)]

    ordered_columns = [
        "video_id",
        "classification",
        "question_category",
        "pool_key",
        "target",
        "caption",
        "generated_transcription",
        "missing_transcription",
        *score_columns,
        "mean_score",
        *raw_columns,
    ]

    return result[ordered_columns]


def save_results(output_xlsx: str | Path, results: pd.DataFrame) -> None:
    output_xlsx = Path(output_xlsx)

    video_summary = (
        results
        .groupby(["video_id", "classification"], as_index=False)
        .agg(
            n_captions=("caption", "count"),
            n_missing_transcriptions=("missing_transcription", "sum"),
            mean_similarity=("mean_score", "mean"),
            min_similarity=("mean_score", "min"),
            max_similarity=("mean_score", "max"),
        )
    )

    category_summary = (
        results
        .groupby(["question_category"], as_index=False)
        .agg(
            n_items=("caption", "count"),
            mean_similarity=("mean_score", "mean"),
            min_similarity=("mean_score", "min"),
            max_similarity=("mean_score", "max"),
        )
    )

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="caption_scores", index=False)
        video_summary.to_excel(writer, sheet_name="video_summary", index=False)
        category_summary.to_excel(writer, sheet_name="category_summary", index=False)

    print(f"File XLSX salvato in: {output_xlsx.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filtra i video dialogue dal file intersection_sets.xlsx, "
            "estrae generated_transcription e caption target, "
            "chiama GPT 3 volte per ogni coppia trascrizione-caption, "
            "calcola la media e salva un XLSX."
        )
    )

    parser.add_argument(
        "--intersection-xlsx",
        required=True,
        help="File XLSX delle intersezioni, es. intersection_sets.xlsx.",
    )
    parser.add_argument(
        "--manual-revision-json",
        default=DEFAULT_MANUAL_REVISION_JSON,
        help="Path o URL del manual_revision.json.",
    )
    parser.add_argument(
        "--dataset-json",
        default=DEFAULT_DATASET_JSON,
        help="Path o URL del maia_ita_mc_by_video_category_pool.json.",
    )
    parser.add_argument(
        "--output-xlsx",
        default=DEFAULT_OUTPUT_XLSX,
        help="File XLSX finale.",
    )
    parser.add_argument(
        "--cache-jsonl",
        default=DEFAULT_CACHE_JSONL,
        help="Cache JSONL per non rifare chiamate GPT già completate.",
    )
    # parser.add_argument(
    #     "--model",
    #     default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
    #     help="Modello OpenAI da usare. Puoi anche impostare OPENAI_MODEL.",
    # )
    parser.add_argument(
    "--model",
    default=os.environ.get("OPENAI_MODEL", "gpt-5.4-nano"),
    help="Modello OpenAI da usare. Puoi anche impostare OPENAI_MODEL.",
)
    parser.add_argument(
        "--runs-per-pair",
        type=int,
        default=3,
        help="Numero di chiamate GPT per ogni coppia trascrizione-caption.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperatura del modello. Usa -1 per non passarla alla API.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=20,
        help="Numero massimo di token generati.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Pausa tra chiamate API, utile per rate limit.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limita il numero di coppie trascrizione-caption, utile per test.",
    )

    args = parser.parse_args()

    if args.runs_per_pair <= 0:
        raise ValueError("--runs-per-pair deve essere maggiore di 0.")

    temperature = None if args.temperature < 0 else args.temperature

    dialogue_videos = load_dialogue_video_ids_from_intersection_xlsx(
        args.intersection_xlsx
    )

    print(f"Video dialogue unici trovati: {len(dialogue_videos)}")

    manual_revision_data = load_json(args.manual_revision_json)
    transcription_map = extract_generated_transcriptions(manual_revision_data)

    print(f"Trascrizioni caricate: {len(transcription_map)}")

    dataset_data = load_json(args.dataset_json)
    caption_table = extract_target_captions(dataset_data)

    print(f"Caption target caricate: {len(caption_table)}")

    rows = build_eval_rows(
        dialogue_videos=dialogue_videos,
        transcription_map=transcription_map,
        caption_table=caption_table,
    )

    if args.max_rows is not None:
        rows = rows.head(args.max_rows).copy()

    print(f"Coppie trascrizione-caption da valutare: {len(rows)}")

    if rows.empty:
        raise ValueError("Nessuna coppia trascrizione-caption da valutare.")

    results = evaluate_rows_with_gpt(
        rows=rows,
        model=args.model,
        runs_per_pair=args.runs_per_pair,
        temperature=temperature,
        max_output_tokens=args.max_output_tokens,
        cache_jsonl=args.cache_jsonl,
        sleep_seconds=args.sleep_seconds,
    )

    save_results(args.output_xlsx, results)


if __name__ == "__main__":
    main()
