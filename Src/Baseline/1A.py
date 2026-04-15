from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import openpyxl
import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Select the GPU to use.
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


JsonDict = Dict[str, Any]
EMPTY_RAW_OUTPUT_TOKEN = "[[EMPTY_OUTPUT]]"


@dataclass(frozen=True)
class Example:
    """
    Single multiple-choice example extracted from the MAIA JSON file.
    """
    video_id: str
    question_category: str
    normalized_question_category: str
    pool_key: str
    choice_0: str
    choice_1: str
    target: int


@dataclass(frozen=True)
class PredictionRecord:
    """
    Flat prediction record used both for JSON export and metric computation.
    raw_model_output is always forced to be a non-empty string.
    """
    video_id: str
    question_category: str
    normalized_question_category: str
    pool_key: str
    choice_0: str
    choice_1: str
    target: int
    transcript: str
    prompt: str
    raw_model_output: str
    predicted_label: Optional[int]
    is_correct: bool


def load_model(
    model_name: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "cuda:5",
    attn_implementation: str = "flash_attention_2",
) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    print(f"[INFO] Loading model: {model_name}")
    print("[INFO] Initializing model weights...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )

    print("[INFO] Model weights loaded.")
    print("[INFO] Loading processor...")

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True, padding_side="left")
    tokenizer = processor.tokenizer

    print("[INFO] Processor loaded successfully.")
    print("[INFO] Model setup completed.")
    return model, processor, tokenizer


def load_json_file(json_path: str | Path) -> JsonDict:
    """
    Load a JSON file from disk.
    """
    path = Path(json_path)
    print(f"[INFO] Loading JSON file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] JSON file loaded successfully: {path}")
    return data


def save_json_file(data: JsonDict, json_path: str | Path) -> None:
    """
    Save a JSON file to disk using UTF-8 and readable indentation.
    """
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving JSON output to: {path}")
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] JSON output saved successfully: {path}")


def save_text_file(text: str, text_path: str | Path) -> None:
    """
    Save a plain text file to disk.
    """
    path = Path(text_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving text report to: {path}")
    with path.open("w", encoding="utf-8") as f:
        f.write(text)
    print(f"[INFO] Text report saved successfully: {path}")


def save_dataframe_csv(df: pd.DataFrame, csv_path: str | Path) -> None:
    """
    Save a pandas DataFrame as CSV.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving DataFrame CSV to: {path}")
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[INFO] DataFrame CSV saved successfully: {path}")


def natural_video_sort_key(video_id: str) -> Tuple[int, str]:
    """
    Sort video IDs like Video2.mp4 before Video10.mp4.
    """
    match = re.search(r"(\d+)", video_id)
    if match:
        return int(match.group(1)), video_id
    return 10**9, video_id


def natural_pool_sort_key(pool_key: str) -> Tuple[int, str]:
    """
    Sort keys like pool_pos_2 before pool_pos_10.
    """
    match = re.search(r"(\d+)", pool_key)
    if match:
        return int(match.group(1)), pool_key
    return 10**9, pool_key


def normalize_question_category(category: str) -> str:
    """
    Normalize question_category by removing a trailing A/B suffix when it is
    expressed as a final token such as:
    - Controfattuale_A
    - Controfattuale B
    - Controfattuale-B

    If no suffix is found, the original category is returned unchanged.
    """
    stripped = category.strip()
    match = re.match(r"^(.*?)(?:[_\-\s]+[AaBb])$", stripped)
    if match:
        return match.group(1).strip("_- ")
    return stripped


def extract_examples(dataset_json: JsonDict) -> List[Example]:
    """
    Extract all examples from the nested dataset JSON.

    Expected structure:
    {
      "Video1.mp4": {
        "Controfattuale_A": {
          "pool_pos_1": {
            "0": "...",
            "1": "...",
            "target": 0
          }
        }
      }
    }
    """
    print("[INFO] Extracting examples from dataset JSON...")
    examples: List[Example] = []

    for video_id, categories in dataset_json.items():
        if not isinstance(categories, dict):
            continue

        for question_category, pools in categories.items():
            if not isinstance(pools, dict):
                continue

            normalized_category = normalize_question_category(question_category)

            for pool_key, payload in pools.items():
                if not isinstance(payload, dict):
                    continue

                choice_0 = str(payload["0"])
                choice_1 = str(payload["1"])
                target = int(payload["target"])

                examples.append(
                    Example(
                        video_id=video_id,
                        question_category=question_category,
                        normalized_question_category=normalized_category,
                        pool_key=pool_key,
                        choice_0=choice_0,
                        choice_1=choice_1,
                        target=target,
                    )
                )

    print(f"[INFO] Extracted {len(examples)} raw examples. Sorting them now...")

    examples.sort(
        key=lambda ex: (
            natural_video_sort_key(ex.video_id),
            ex.normalized_question_category,
            ex.question_category,
            natural_pool_sort_key(ex.pool_key),
        )
    )

    print("[INFO] Example extraction and sorting completed.")
    return examples


def get_video_metadata(final_results: JsonDict, video_id: str) -> JsonDict:
    """
    Return the metadata block for a given video from final_results.json.
    """
    video_data = final_results.get(video_id, {})
    return video_data if isinstance(video_data, dict) else {}


def get_transcript_for_video(final_results: JsonDict, video_id: str) -> str:
    """
    Retrieve the transcript for a given video if it is available and relevant.

    The previous logic is preserved:
    if the classification contains the substring 'dialogue' the transcription is
    returned. Otherwise, a standard fallback string is returned.
    """
    video_data = get_video_metadata(final_results, video_id)
    classification = str(video_data.get("classification", "")).lower()
    transcript = str(video_data.get("generated_transcription", "")).strip()

    if "dialogue" in classification:
        return transcript

    return "Trascrizione non disponibile, l'audio contiene musica o rumore."


def build_prompt(choice_0: str, choice_1: str, transcript: str) -> str:
    """
    Build the text prompt for the model.

    The active prompt is intentionally kept unchanged.
    """
    return (
        f"Scegli la descrizione corretta rispetto al contenuto del video dato:\n"
        f"0. {choice_0}\n"
        f"1. {choice_1}\n\n"
        f"Rispondi solo con {0} o {1}"
    )


def build_messages(prompts: Sequence[str]) -> List[List[Dict[str, Any]]]:
    """
    Convert a list of prompts into chat-formatted messages.
    """
    print(f"[INFO] Building chat messages for {len(prompts)} prompts...")
    messages: List[List[Dict[str, Any]]] = []

    for prompt in prompts:
        messages.append(
            [
                {
                    "role": "system",
                    "content": "You are a precise assistant."
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                },
            ]
        )

    print("[INFO] Chat messages built successfully.")
    return messages


def move_inputs_to_model_device(
    inputs: Dict[str, Any],
    model: Qwen2_5_VLForConditionalGeneration,
) -> Dict[str, Any]:
    """
    Move tensor inputs to the device used by the first model parameter.
    This is the safest simple strategy for generation with device_map='auto'.
    """
    print("[INFO] Moving batch inputs to model device...")
    model_device = next(model.parameters()).device
    moved_inputs = {
        key: value.to(model_device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    print(f"[INFO] Inputs moved to device: {model_device}")
    return moved_inputs


def generate_answers_batch(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    prompts: Sequence[str],
    max_new_tokens: int = 16,
) -> List[str]:
    """
    Run batched generation for a list of prompts.
    """
    print(f"[INFO] Generating answers for a batch of {len(prompts)} prompts...")
    messages = build_messages(prompts)

    print("[INFO] Applying chat templates...")
    rendered_texts = [
        processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        for message in messages
    ]

    print("[INFO] Tokenizing batch inputs...")
    inputs = processor(
        text=rendered_texts,
        padding=True,
        return_tensors="pt",
    )
    inputs = move_inputs_to_model_device(inputs, model)

    print("[INFO] Running model.generate()...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    print("[INFO] Decoding generated outputs...")

    trimmed_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_texts = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    print("[INFO] Batch generation completed.")
    return [text.strip() for text in output_texts]


def normalize_raw_output(raw_output: Optional[str]) -> str:
    """
    Force raw_model_output to always be a non-empty string so that evaluation
    remains meaningful and explicit.
    """
    if raw_output is None:
        return EMPTY_RAW_OUTPUT_TOKEN

    cleaned = str(raw_output).strip()
    return cleaned if cleaned else EMPTY_RAW_OUTPUT_TOKEN


def parse_binary_answer(raw_output: str) -> Optional[int]:
    """
    Extract the first valid binary answer from the model output.

    Returns:
    - 0 or 1 if a valid answer is found
    - None otherwise
    """
    cleaned = raw_output.strip()

    if "0" in cleaned:
        return 0
    if "1" in cleaned:
        return 1
    else:
        return None


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Safe floating-point division.
    """
    return numerator / denominator if denominator != 0 else 0.0


def compute_confusion_counts(records: Sequence[PredictionRecord]) -> Dict[str, int]:
    """
    Compute a binary 2x2 confusion matrix over valid predictions only.

                    pred_0  pred_1
        actual_0      c00     c01
        actual_1      c10     c11
    """
    c00 = sum(
        record.target == 0 and record.predicted_label == 0
        for record in records
    )
    c01 = sum(
        record.target == 0 and record.predicted_label == 1
        for record in records
    )
    c10 = sum(
        record.target == 1 and record.predicted_label == 0
        for record in records
    )
    c11 = sum(
        record.target == 1 and record.predicted_label == 1
        for record in records
    )

    return {
        "actual_0_pred_0": c00,
        "actual_0_pred_1": c01,
        "actual_1_pred_0": c10,
        "actual_1_pred_1": c11,
    }


def compute_label_metrics(
    records: Sequence[PredictionRecord],
    label: int,
) -> Dict[str, float]:
    """
    Compute class-specific metrics for one label without any averaging.
    Invalid predictions count as misses for the true class in recall.
    """
    tp = sum(
        record.predicted_label == label and record.target == label
        for record in records
    )
    fp = sum(
        record.predicted_label == label and record.target != label
        for record in records
    )
    fn = sum(
        record.target == label and record.predicted_label != label
        for record in records
    )

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    support = sum(record.target == label for record in records)
    predicted_as_label = sum(record.predicted_label == label for record in records)

    return {
        "support": float(support),
        "predicted_as_label": float(predicted_as_label),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_metrics(records: Sequence[PredictionRecord]) -> JsonDict:
    """
    Compute metrics for the provided subset of records.

    If the subset is global, the metrics are global.
    If the subset contains one video, the metrics are per-video.
    If the subset contains one category, the metrics are per-category.

    No macro-average or micro-average is used.
    """
    print(f"[INFO] Computing metrics for {len(records)} records...")

    n_samples = len(records)
    empty_raw_outputs = sum(
        record.raw_model_output == EMPTY_RAW_OUTPUT_TOKEN
        for record in records
    )
    invalid_predictions = sum(
        record.predicted_label not in (0, 1)
        for record in records
    )
    valid_predictions = n_samples - invalid_predictions
    correct_predictions = sum(record.is_correct for record in records)

    label_0 = compute_label_metrics(records, label=0)
    label_1 = compute_label_metrics(records, label=1)
    confusion_counts = compute_confusion_counts(records)

    print("[INFO] Metrics computed successfully.")
    return {
        "n_samples": n_samples,
        "valid_predictions": valid_predictions,
        "invalid_predictions": invalid_predictions,
        "empty_raw_outputs": empty_raw_outputs,

        "Accuracy": safe_divide(correct_predictions, n_samples),
        "Accuracy_valid_only": safe_divide(
            confusion_counts["actual_0_pred_0"] + confusion_counts["actual_1_pred_1"],
            valid_predictions,
        ),

        "Support_0": int(label_0["support"]),
        "Predicted_as_0": int(label_0["predicted_as_label"]),
        "Precision_0": label_0["precision"],
        "Recall_0": label_0["recall"],
        "F1_0": label_0["f1"],

        "Support_1": int(label_1["support"]),
        "Predicted_as_1": int(label_1["predicted_as_label"]),
        "Precision_1": label_1["precision"],
        "Recall_1": label_1["recall"],
        "F1_1": label_1["f1"],

        **confusion_counts,
    }


def compute_all_metrics(records: Sequence[PredictionRecord]) -> JsonDict:
    """
    Compute:
    - global metrics
    - metrics by normalized question category
    - metrics by video
    """
    print("[INFO] Computing aggregated metrics...")
    by_class: Dict[str, List[PredictionRecord]] = {}
    by_video: Dict[str, List[PredictionRecord]] = {}

    for record in records:
        by_class.setdefault(record.normalized_question_category, []).append(record)
        by_video.setdefault(record.video_id, []).append(record)

    print(f"[INFO] Found {len(by_class)} normalized question categories.")
    print(f"[INFO] Found {len(by_video)} videos with predictions.")

    class_metrics = {
        class_name: compute_metrics(class_records)
        for class_name, class_records in sorted(by_class.items())
    }
    video_metrics = {
        video_id: compute_metrics(video_records)
        for video_id, video_records in sorted(
            by_video.items(),
            key=lambda item: natural_video_sort_key(item[0]),
        )
    }

    print("[INFO] Aggregated metrics completed.")
    return {
        "global": compute_metrics(records),
        "by_normalized_question_category": class_metrics,
        "by_video": video_metrics,
    }


def metric_row_from_summary(
    entity_name: str,
    metrics: JsonDict,
    entity_column_name: str,
) -> Dict[str, Any]:
    """
    Convert an aggregate metric summary dictionary into a single flat row
    suitable for a pandas DataFrame.
    """
    return {
        entity_column_name: entity_name,
        "n_samples": metrics["n_samples"],
        "valid_predictions": metrics["valid_predictions"],
        "invalid_predictions": metrics["invalid_predictions"],
        "empty_raw_outputs": metrics["empty_raw_outputs"],

        "Accuracy": metrics["Accuracy"],
        "Accuracy_valid_only": metrics["Accuracy_valid_only"],

        "Support_0": metrics["Support_0"],
        "Predicted_as_0": metrics["Predicted_as_0"],
        "Precision_0": metrics["Precision_0"],
        "Recall_0": metrics["Recall_0"],
        "F1_0": metrics["F1_0"],

        "Support_1": metrics["Support_1"],
        "Predicted_as_1": metrics["Predicted_as_1"],
        "Precision_1": metrics["Precision_1"],
        "Recall_1": metrics["Recall_1"],
        "F1_1": metrics["F1_1"],

        "actual_0_pred_0": metrics["actual_0_pred_0"],
        "actual_0_pred_1": metrics["actual_0_pred_1"],
        "actual_1_pred_0": metrics["actual_1_pred_0"],
        "actual_1_pred_1": metrics["actual_1_pred_1"],
    }


def build_global_metrics_dataframe(metrics_summary: JsonDict) -> pd.DataFrame:
    """
    One-row DataFrame with the global evaluation metrics.

    Meaning:
    - This summarizes the overall behaviour of the model on the full dataset.
    """
    return pd.DataFrame(
        [
            metric_row_from_summary(
                entity_name="GLOBAL",
                metrics=metrics_summary["global"],
                entity_column_name="scope",
            )
        ]
    )


def build_category_metrics_dataframe(metrics_summary: JsonDict) -> pd.DataFrame:
    """
    DataFrame with one row per normalized_question_category.

    Meaning:
    - This lets you compare model quality across semantic question families
      after collapsing suffix variants such as _A / _B into the same category.
    """
    rows = [
        metric_row_from_summary(
            entity_name=category_name,
            metrics=category_metrics,
            entity_column_name="normalized_question_category",
        )
        for category_name, category_metrics
        in metrics_summary["by_normalized_question_category"].items()
    ]
    return pd.DataFrame(rows)


def build_video_metrics_dataframe(metrics_summary: JsonDict) -> pd.DataFrame:
    """
    DataFrame with one row per video.
    """
    rows = [
        metric_row_from_summary(
            entity_name=video_id,
            metrics=video_metrics,
            entity_column_name="video_id",
        )
        for video_id, video_metrics in metrics_summary["by_video"].items()
    ]
    return pd.DataFrame(rows)


def build_confusion_matrix_dataframe(metrics: JsonDict) -> pd.DataFrame:
    """
    Build a 2x2 confusion matrix DataFrame from a metric summary.
    """
    return pd.DataFrame(
        [
            [metrics["actual_0_pred_0"], metrics["actual_0_pred_1"]],
            [metrics["actual_1_pred_0"], metrics["actual_1_pred_1"]],
        ],
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )


def build_category_confusion_dataframe(metrics_summary: JsonDict) -> pd.DataFrame:
    """
    Flat confusion counts by normalized_question_category.
    """
    rows: List[Dict[str, Any]] = []

    for category_name, metrics in metrics_summary["by_normalized_question_category"].items():
        rows.append(
            {
                "normalized_question_category": category_name,
                "actual_0_pred_0": metrics["actual_0_pred_0"],
                "actual_0_pred_1": metrics["actual_0_pred_1"],
                "actual_1_pred_0": metrics["actual_1_pred_0"],
                "actual_1_pred_1": metrics["actual_1_pred_1"],
            }
        )

    return pd.DataFrame(rows)


def dataframe_to_string(df: pd.DataFrame, index: bool = False) -> str:
    """
    Convert a pandas DataFrame to a readable string with stable formatting.
    """
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 240,
        "display.max_colwidth", 240,
    ):
        return df.to_string(index=index, float_format=lambda x: f"{x:.4f}")


def build_metrics_report(metrics_summary: JsonDict) -> str:
    """
    Build a readable TXT report focused only on the meaningful aggregate metrics.
    """
    print("[INFO] Building metrics text report...")

    global_df = build_global_metrics_dataframe(metrics_summary)
    category_df = build_category_metrics_dataframe(metrics_summary)
    video_df = build_video_metrics_dataframe(metrics_summary)
    global_conf_df = build_confusion_matrix_dataframe(metrics_summary["global"])
    category_conf_df = build_category_confusion_dataframe(metrics_summary)

    report_parts: List[str] = [
        "MODEL EVALUATION REPORT",
        "=======================",
        "",
        "DEFINITIONS",
        "-----------",
        "- Global metrics summarize the overall model performance on the full dataset.",
        "- Per-video metrics are computed only on the examples of that specific video.",
        "- Per-category metrics are computed only on the examples of that specific normalized question category.",
        "- Accuracy is computed on the entire current subset.",
        "- Accuracy_valid_only is computed only on valid parsed predictions.",
        "- Precision, recall and F1 are reported separately for label 0 and label 1.",
        "- No macro-average or micro-average is used.",
        "- Invalid predictions are outputs that could not be parsed as 0 or 1.",
        f"- Empty raw outputs are normalized to the explicit token: {EMPTY_RAW_OUTPUT_TOKEN}",
        "- Confusion matrices include only valid predictions.",
        "- Recall penalizes invalid predictions because an invalid output is treated as a miss for the true class.",
        "",
        "GLOBAL PERFORMANCE",
        "------------------",
        dataframe_to_string(global_df, index=False),
        "",
        "GLOBAL CONFUSION MATRIX",
        "-----------------------",
        dataframe_to_string(global_conf_df, index=True),
        "",
        "PER NORMALIZED QUESTION CATEGORY",
        "--------------------------------",
        dataframe_to_string(category_df, index=False),
        "",
        "PER NORMALIZED QUESTION CATEGORY CONFUSION",
        "------------------------------------------",
        dataframe_to_string(category_conf_df, index=False),
        "",
        "PER VIDEO PERFORMANCE",
        "---------------------",
        dataframe_to_string(video_df, index=False),
    ]

    print("[INFO] Metrics text report built successfully.")
    return "\n".join(report_parts)


def plot_global_confusion_matrix(metrics_summary: JsonDict, output_path: str | Path) -> None:
    """
    Save a PNG heatmap for the global confusion matrix.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conf_df = build_confusion_matrix_dataframe(metrics_summary["global"])
    values = conf_df.values

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(values)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(conf_df.columns)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(conf_df.index)
    ax.set_title("Global confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, str(values[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Global confusion matrix plot saved to: {output_path}")


def plot_metric_by_category(
    df: pd.DataFrame,
    metric_column: str,
    title: str,
    ylabel: str,
    output_path: str | Path,
) -> None:
    """
    Save a bar chart for a metric by normalized question category.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print(f"[INFO] Skipping plot {title}: empty DataFrame.")
        return

    plot_df = df.sort_values(metric_column, ascending=False)

    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 0.8), 5))
    ax.bar(plot_df["normalized_question_category"], plot_df[metric_column])

    ax.set_title(title)
    ax.set_xlabel("Normalized question category")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)

    values = plot_df[metric_column].tolist()
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Plot saved to: {output_path}")


def save_evaluation_artifacts(
    metrics_summary: JsonDict,
    output_dir: str | Path,
) -> None:
    """
    Save evaluation artifacts:
    - Excel workbook with multiple sheets
    - CSV files for the main DataFrames
    - PNG charts for quick visual inspection
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    global_df = build_global_metrics_dataframe(metrics_summary)
    category_df = build_category_metrics_dataframe(metrics_summary)
    video_df = build_video_metrics_dataframe(metrics_summary)
    global_conf_df = build_confusion_matrix_dataframe(metrics_summary["global"])
    category_conf_df = build_category_confusion_dataframe(metrics_summary)

    workbook_path = output_dir / "1A_evaluation_summary.xlsx"
    global_csv_path = output_dir / "global_metrics.csv"
    category_csv_path = output_dir / "normalized_question_category_metrics.csv"
    video_csv_path = output_dir / "video_metrics.csv"
    category_conf_csv_path = output_dir / "normalized_question_category_confusion.csv"

    with pd.ExcelWriter(workbook_path) as writer:
        global_df.to_excel(writer, sheet_name="global_metrics", index=False)
        category_df.to_excel(writer, sheet_name="category_metrics", index=False)
        video_df.to_excel(writer, sheet_name="video_metrics", index=False)
        global_conf_df.to_excel(writer, sheet_name="global_confusion")
        category_conf_df.to_excel(writer, sheet_name="category_confusion", index=False)

    save_dataframe_csv(global_df, global_csv_path)
    save_dataframe_csv(category_df, category_csv_path)
    save_dataframe_csv(video_df, video_csv_path)
    save_dataframe_csv(category_conf_df, category_conf_csv_path)

    plot_global_confusion_matrix(
        metrics_summary,
        plots_dir / "global_confusion_matrix.png",
    )
    plot_metric_by_category(
        category_df,
        metric_column="Accuracy",
        title="Accuracy by normalized question category",
        ylabel="Accuracy",
        output_path=plots_dir / "Accuracy_by_category.png",
    )
    plot_metric_by_category(
        category_df,
        metric_column="F1_0",
        title="F1 for label 0 by normalized question category",
        ylabel="F1 (label 0)",
        output_path=plots_dir / "F1_0_by_category.png",
    )
    plot_metric_by_category(
        category_df,
        metric_column="F1_1",
        title="F1 for label 1 by normalized question category",
        ylabel="F1 (label 1)",
        output_path=plots_dir / "F1_1_by_category.png",
    )
    plot_metric_by_category(
        category_df,
        metric_column="invalid_predictions",
        title="Invalid predictions by normalized question category",
        ylabel="Invalid predictions",
        output_path=plots_dir / "invalid_predictions_by_category.png",
    )

    print(f"[INFO] Evaluation workbook saved to: {workbook_path}")
    print(f"[INFO] Evaluation CSVs saved in: {output_dir}")
    print(f"[INFO] Evaluation plots saved in: {plots_dir}")


def build_results_json(
    records: Sequence[PredictionRecord],
    final_results: JsonDict,
    metrics_summary: JsonDict,
) -> JsonDict:
    """
    Build the final per-video JSON file.
    """
    print("[INFO] Building final results JSON structure...")
    results: JsonDict = {}

    for record in records:
        video_metadata = get_video_metadata(final_results, record.video_id)

        if record.video_id not in results:
            results[record.video_id] = {
                "classification": video_metadata.get("classification"),
                "score": video_metadata.get("score"),
                "score_meaning": video_metadata.get("score_meaning"),
                "selected_model": video_metadata.get("selected_model"),
                "generated_transcription": video_metadata.get("generated_transcription"),
                "metrics": metrics_summary["by_video"].get(record.video_id, {}),
                "questions": {},
            }

        video_questions = results[record.video_id]["questions"]
        if record.question_category not in video_questions:
            video_questions[record.question_category] = {}

        video_questions[record.question_category][record.pool_key] = {
            "normalized_question_category": record.normalized_question_category,
            "0": record.choice_0,
            "1": record.choice_1,
            "target": record.target,
            "prompt": record.prompt,
            "raw_model_output": record.raw_model_output,
            "predicted_label": record.predicted_label,
            "is_correct": record.is_correct,
        }

    print("[INFO] Final results JSON structure built successfully.")
    return results


def run_inference(
    examples: Sequence[Example],
    final_results: JsonDict,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    batch_size: int = 16,
) -> List[PredictionRecord]:
    """
    Run batched inference on all extracted examples.
    """
    print("[INFO] Starting batched inference...")
    records: List[PredictionRecord] = []
    total_examples = len(examples)
    total_batches = (total_examples + batch_size - 1) // batch_size

    print(f"[INFO] Total examples to process: {total_examples}")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Total batches: {total_batches}")

    for start_idx in range(0, total_examples, batch_size):
        batch_examples = examples[start_idx:start_idx + batch_size]
        batch_number = (start_idx // batch_size) + 1

        print(
            f"[INFO] Processing batch {batch_number}/{total_batches} "
            f"(examples {start_idx + 1}-{start_idx + len(batch_examples)} of {total_examples})..."
        )

        prompts: List[str] = []
        transcripts: List[str] = []

        for example in batch_examples:
            transcript = get_transcript_for_video(final_results, example.video_id)
            prompt = build_prompt(
                choice_0=example.choice_0,
                choice_1=example.choice_1,
                transcript=transcript,
            )
            transcripts.append(transcript)
            prompts.append(prompt)

        print(f"[INFO] Built {len(prompts)} prompts for current batch.")

        raw_outputs = generate_answers_batch(
            model=model,
            processor=processor,
            prompts=prompts,
        )

        print(f"[INFO] Received {len(raw_outputs)} raw outputs for current batch.")

        for example, transcript, prompt, raw_output in zip(
            batch_examples,
            transcripts,
            prompts,
            raw_outputs,
        ):
            safe_raw_output = normalize_raw_output(raw_output)
            predicted_label = parse_binary_answer(safe_raw_output)
            is_correct = predicted_label == example.target

            records.append(
                PredictionRecord(
                    video_id=example.video_id,
                    question_category=example.question_category,
                    normalized_question_category=example.normalized_question_category,
                    pool_key=example.pool_key,
                    choice_0=example.choice_0,
                    choice_1=example.choice_1,
                    target=example.target,
                    transcript=transcript,
                    prompt=prompt,
                    raw_model_output=safe_raw_output,
                    predicted_label=predicted_label,
                    is_correct=is_correct,
                )
            )

        processed = min(start_idx + len(batch_examples), total_examples)
        print(f"[INFO] Processed {processed}/{total_examples} examples so far.")

    print("[INFO] Inference completed for all examples.")
    return records


def main() -> None:
    print("[INFO] Script started.")
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Input files
    final_results_path = Path("Data/TranscriptionData/final_classification/final_results.json")
    dataset_path = Path("Data/Dataset/maia_ita_mc_by_video_category_pool.json")

    # Output files
    predictions_output_path = Path("Data/ModelResponse/1A/qwen_mc_1A_predictions_by_video.json")
    metrics_report_output_path = Path("Data/ModelResponse/1A/qwen_mc_1A_metrics_report.txt")
    evaluation_output_dir = Path("Data/ModelResponse/1A/evaluation")

    # Inference configuration
    batch_size = 4

    print("[INFO] Configuration loaded.")
    print(f"[INFO] final_results_path: {final_results_path}")
    print(f"[INFO] dataset_path: {dataset_path}")
    print(f"[INFO] predictions_output_path: {predictions_output_path}")
    print(f"[INFO] metrics_report_output_path: {metrics_report_output_path}")
    print(f"[INFO] evaluation_output_dir: {evaluation_output_dir}")
    print(f"[INFO] batch_size: {batch_size}")

    print("[INFO] Loading input files...")
    final_results = load_json_file(final_results_path)
    dataset_json = load_json_file(dataset_path)
    print("[INFO] Input files loaded successfully.")

    examples = extract_examples(dataset_json)
    if not examples:
        raise ValueError("No examples were found in the dataset JSON.")

    print(f"[INFO] Total extracted examples: {len(examples)}")

    print("[INFO] Loading model and processor...")
    model, processor, tokenizer = load_model(model_name)
    print("[INFO] Model and processor are ready.")

    print("[INFO] Starting inference phase...")
    records = run_inference(
        examples=examples,
        final_results=final_results,
        model=model,
        processor=processor,
        batch_size=batch_size,
    )
    print(f"[INFO] Inference phase completed. Total prediction records: {len(records)}")

    print("[INFO] Starting metrics computation...")
    metrics_summary = compute_all_metrics(records)
    print("[INFO] Metrics computation completed.")

    print("[INFO] Building final predictions JSON...")
    predictions_json = build_results_json(
        records=records,
        final_results=final_results,
        metrics_summary=metrics_summary,
    )
    print("[INFO] Final predictions JSON built successfully.")

    print("[INFO] Saving output files...")
    save_json_file(predictions_json, predictions_output_path)

    metrics_report = build_metrics_report(metrics_summary)
    save_text_file(metrics_report, metrics_report_output_path)

    save_evaluation_artifacts(metrics_summary, evaluation_output_dir)

    print(f"[INFO] Predictions saved to: {predictions_output_path}")
    print(f"[INFO] Metrics report saved to: {metrics_report_output_path}")
    print(f"[INFO] Evaluation artifacts saved to: {evaluation_output_dir}")
    print("[INFO] Script finished successfully.")


if __name__ == "__main__":
    main()
    # Data/TranscriptionData/final_classification/final_results.json
