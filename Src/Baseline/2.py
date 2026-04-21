from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Select the GPU to use.
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

JsonDict = Dict[str, Any]
EMPTY_RAW_OUTPUT_TOKEN = "[[EMPTY_OUTPUT]]"


@dataclass(frozen=True)
class Example:
    """
    Single multiple-choice example extracted from the dataset JSON.
    """
    video_id: str
    question_category: str
    normalized_question_category: str
    pool_key: str
    answer_0: str
    answer_1: str
    target: int


@dataclass(frozen=True)
class PredictionRecord:
    """
    Flat prediction record used for JSON export and metric computation.
    """
    video_id: str
    question_category: str
    normalized_question_category: str
    pool_key: str
    answer_0: str
    answer_1: str
    target: int
    transcript: str
    prompt: str
    raw_model_output: str
    predicted_label: Optional[int]
    is_correct: bool


def load_model(
    model_name: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "cuda:2",
    attn_implementation: str = "flash_attention_2",
) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    """
    Load the model and processor.
    """
    print(f"[INFO] Loading model: {model_name}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )

    processor = AutoProcessor.from_pretrained(
        model_name,
        # use_fast=True,
        padding_side="left",
    )

    print("[INFO] Model and processor loaded successfully.")
    return model, processor


def load_json_file(json_path: str | Path) -> JsonDict:
    """
    Load a JSON file from disk.
    """
    path = Path(json_path)
    print(f"[INFO] Loading JSON file: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json_file(data: JsonDict, json_path: str | Path) -> None:
    """
    Save a JSON file to disk using UTF-8 and readable indentation.
    """
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving JSON output to: {path}")
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_text_file(text: str, text_path: str | Path) -> None:
    """
    Save a plain text file to disk.
    """
    path = Path(text_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving text report to: {path}")
    with path.open("w", encoding="utf-8") as file:
        file.write(text)


def save_dataframe_csv(df: pd.DataFrame, csv_path: str | Path) -> None:
    """
    Save a pandas DataFrame as CSV.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving CSV to: {path}")
    df.to_csv(path, index=False, encoding="utf-8")


def natural_numeric_sort_key(value: str) -> Tuple[int, str]:
    """
    Sort strings such as Video2.mp4 before Video10.mp4.
    """
    match = re.search(r"(\d+)", value)
    if match:
        return int(match.group(1)), value
    return 10**9, value


def normalize_question_category(category: str) -> str:
    """
    Remove a trailing A/B suffix from category names when present.
    """
    stripped = category.strip()
    match = re.match(r"^(.*?)(?:[_\-\s]+[AaBb])$", stripped)
    if match:
        return match.group(1).strip("_- ")
    return stripped


def extract_examples(dataset_json: JsonDict) -> List[Example]:
    """
    Extract all multiple-choice examples from the dataset JSON.
    """
    print("[INFO] Extracting examples from dataset...")
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

                examples.append(
                    Example(
                        video_id=video_id,
                        question_category=question_category,
                        normalized_question_category=normalized_category,
                        pool_key=pool_key,
                        answer_0=str(payload["0"]),
                        answer_1=str(payload["1"]),
                        target=int(payload["target"]),
                    )
                )

    examples.sort(
        key=lambda example: (
            natural_numeric_sort_key(example.video_id),
            example.normalized_question_category,
            example.question_category,
            natural_numeric_sort_key(example.pool_key),
        )
    )

    print(f"[INFO] Extracted {len(examples)} examples.")
    return examples


def load_transcript_map(transcription_json: JsonDict) -> Dict[str, str]:
    """
    Build a simple video_id -> transcript mapping.

    If a transcription is missing or empty, a fallback text is used.
    """
    print("[INFO] Building transcript map...")
    transcript_map: Dict[str, str] = {}

    for video_id, payload in transcription_json.items():
        if not isinstance(payload, dict):
            transcript_map[video_id] = "L'audio contiene solo musica o rumore"
            continue

        transcript = str(
            payload.get("generated_transcription", "L'audio contiene solo musica o rumore")
        ).strip()

        if not transcript:
            transcript = "L'audio contiene solo musica o rumore"

        transcript_map[video_id] = transcript

    print(f"[INFO] Transcript map built for {len(transcript_map)} videos.")
    return transcript_map


def build_prompt(answer_0: str, answer_1: str, transcript: str) -> str:
    """
    Build the text prompt for experiment 2.
    """
    return (
        f"Dato il video, considera anche la trascrizione del suo audio: {transcript}.\n"
        "Scegli la descrizione corretta rispetto al contenuto del video:\n"
        f"0:{answer_0}\n"
        f"1:{answer_1}\n"
        "Rispondi solo con 0 o 1."
    )


def build_messages(prompts: Sequence[str]) -> List[List[Dict[str, Any]]]:
    """
    Convert prompts into chat-formatted messages.
    """
    print(f"[INFO] Building chat messages for {len(prompts)} prompts...")
    messages: List[List[Dict[str, Any]]] = []

    for prompt in prompts:
        messages.append(
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a precise binary assistant. Answer only with 0 or 1.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                },
            ]
        )

    return messages


def move_inputs_to_model_device(
    inputs: Dict[str, Any],
    model: Qwen2_5_VLForConditionalGeneration,
) -> Dict[str, Any]:
    """
    Move tensor inputs to the device used by the first model parameter.
    """
    model_device = next(model.parameters()).device
    return {
        key: value.to(model_device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def generate_answers_batch(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    prompts: Sequence[str],
    max_new_tokens: int = 16,
) -> List[str]:
    """
    Run batched generation for a list of prompts.
    """
    messages = build_messages(prompts)

    rendered_texts = [
        processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        for message in messages
    ]

    inputs = processor(
        text=rendered_texts,
        padding=True,
        return_tensors="pt",
    )
    inputs = move_inputs_to_model_device(inputs, model)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    trimmed_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_texts = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return [text.strip() for text in output_texts]


def normalize_raw_output(raw_output: Optional[str]) -> str:
    """
    Force raw_model_output to always be a non-empty string.
    """
    if raw_output is None:
        return EMPTY_RAW_OUTPUT_TOKEN

    cleaned = str(raw_output).strip()
    return cleaned if cleaned else EMPTY_RAW_OUTPUT_TOKEN


def parse_binary_answer(raw_output: str) -> Optional[int]:
    """
    Extract the first valid standalone binary answer from the model output.
    """
    match = re.search(r"\b([01])\b", raw_output)
    if match:
        return int(match.group(1))
    return None


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Safe floating-point division.
    """
    return numerator / denominator if denominator != 0 else 0.0


def compute_confusion_counts(records: Sequence[PredictionRecord]) -> Dict[str, int]:
    """
    Compute confusion counts over valid predictions only.
    """
    true_0_pred_0 = sum(record.target == 0 and record.predicted_label == 0 for record in records)
    true_0_pred_1 = sum(record.target == 0 and record.predicted_label == 1 for record in records)
    true_1_pred_0 = sum(record.target == 1 and record.predicted_label == 0 for record in records)
    true_1_pred_1 = sum(record.target == 1 and record.predicted_label == 1 for record in records)

    return {
        "true_0_pred_0": true_0_pred_0,
        "true_0_pred_1": true_0_pred_1,
        "true_1_pred_0": true_1_pred_0,
        "true_1_pred_1": true_1_pred_1,
    }


def compute_label_metrics(records: Sequence[PredictionRecord], label: int) -> Dict[str, float]:
    """
    Compute precision, recall and F1 for a single label.
    """
    true_positive = sum(
        record.predicted_label == label and record.target == label
        for record in records
    )
    false_positive = sum(
        record.predicted_label == label and record.target != label
        for record in records
    )
    false_negative = sum(
        record.target == label and record.predicted_label != label
        for record in records
    )

    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    f1_score = safe_divide(2 * precision * recall, precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def compute_metrics(records: Sequence[PredictionRecord]) -> JsonDict:
    """
    Compute a compact metric summary for a subset of records.
    """
    total_examples = len(records)
    empty_answers = sum(record.raw_model_output == EMPTY_RAW_OUTPUT_TOKEN for record in records)
    invalid_answers = sum(record.predicted_label not in (0, 1) for record in records)
    valid_answers = total_examples - invalid_answers
    correct_answers = sum(record.is_correct for record in records)

    label_0_metrics = compute_label_metrics(records, label=0)
    label_1_metrics = compute_label_metrics(records, label=1)
    confusion = compute_confusion_counts(records)

    return {
        "total_examples": total_examples,
        "valid_answers": valid_answers,
        "invalid_answers": invalid_answers,
        "empty_answers": empty_answers,
        "overall_accuracy": safe_divide(correct_answers, total_examples),
        "valid_only_accuracy": safe_divide(
            confusion["true_0_pred_0"] + confusion["true_1_pred_1"],
            valid_answers,
        ),
        "precision_label_0": label_0_metrics["precision"],
        "recall_label_0": label_0_metrics["recall"],
        "f1_label_0": label_0_metrics["f1_score"],
        "precision_label_1": label_1_metrics["precision"],
        "recall_label_1": label_1_metrics["recall"],
        "f1_label_1": label_1_metrics["f1_score"],
        **confusion,
    }


def compute_all_metrics(records: Sequence[PredictionRecord]) -> JsonDict:
    """
    Compute global metrics, metrics by normalized question category,
    and metrics by video.
    """
    print("[INFO] Computing aggregated metrics...")
    by_category: Dict[str, List[PredictionRecord]] = {}
    by_video: Dict[str, List[PredictionRecord]] = {}

    for record in records:
        by_category.setdefault(record.normalized_question_category, []).append(record)
        by_video.setdefault(record.video_id, []).append(record)

    category_metrics = {
        category_name: compute_metrics(category_records)
        for category_name, category_records in sorted(by_category.items())
    }

    video_metrics = {
        video_id: compute_metrics(video_records)
        for video_id, video_records in sorted(
            by_video.items(),
            key=lambda item: natural_numeric_sort_key(item[0]),
        )
    }

    return {
        "global": compute_metrics(records),
        "by_normalized_question_category": category_metrics,
        "by_video": video_metrics,
    }


def metric_row_from_summary(
    entity_name: str,
    metrics: JsonDict,
    entity_column_name: str,
) -> Dict[str, Any]:
    """
    Convert a metric summary into a flat DataFrame row.
    """
    return {
        entity_column_name: entity_name,
        "total_examples": metrics["total_examples"],
        "valid_answers": metrics["valid_answers"],
        "invalid_answers": metrics["invalid_answers"],
        "empty_answers": metrics["empty_answers"],
        "overall_accuracy": metrics["overall_accuracy"],
        "valid_only_accuracy": metrics["valid_only_accuracy"],
        "precision_label_0": metrics["precision_label_0"],
        "recall_label_0": metrics["recall_label_0"],
        "f1_label_0": metrics["f1_label_0"],
        "precision_label_1": metrics["precision_label_1"],
        "recall_label_1": metrics["recall_label_1"],
        "f1_label_1": metrics["f1_label_1"],
        "true_0_pred_0": metrics["true_0_pred_0"],
        "true_0_pred_1": metrics["true_0_pred_1"],
        "true_1_pred_0": metrics["true_1_pred_0"],
        "true_1_pred_1": metrics["true_1_pred_1"],
    }


def build_global_metrics_dataframe(metrics_summary: JsonDict) -> pd.DataFrame:
    """
    One-row DataFrame with the global evaluation metrics.
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
    DataFrame with one row per normalized question category.
    """
    rows = [
        metric_row_from_summary(
            entity_name=category_name,
            metrics=category_metrics,
            entity_column_name="normalized_question_category",
        )
        for category_name, category_metrics in metrics_summary["by_normalized_question_category"].items()
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
    Build a 2x2 confusion matrix DataFrame.
    """
    return pd.DataFrame(
        [
            [metrics["true_0_pred_0"], metrics["true_0_pred_1"]],
            [metrics["true_1_pred_0"], metrics["true_1_pred_1"]],
        ],
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    )


def dataframe_to_string(df: pd.DataFrame, index: bool = False) -> str:
    """
    Convert a pandas DataFrame to a readable string.
    """
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 240,
        "display.max_colwidth", 240,
    ):
        return df.to_string(index=index, float_format=lambda value: f"{value:.4f}")


def build_metrics_report(metrics_summary: JsonDict) -> str:
    """
    Build a compact TXT report for experiment 2.
    """
    global_df = build_global_metrics_dataframe(metrics_summary)
    category_df = build_category_metrics_dataframe(metrics_summary)
    video_df = build_video_metrics_dataframe(metrics_summary)
    global_conf_df = build_confusion_matrix_dataframe(metrics_summary["global"])

    report_parts: List[str] = [
        "EXPERIMENT 2 REPORT",
        "===================",
        "",
        "GLOBAL METRICS",
        "--------------",
        dataframe_to_string(global_df, index=False),
        "",
        "GLOBAL CONFUSION MATRIX",
        "-----------------------",
        dataframe_to_string(global_conf_df, index=True),
        "",
        "METRICS BY NORMALIZED QUESTION CATEGORY",
        "--------------------------------------",
        dataframe_to_string(category_df, index=False),
        "",
        "METRICS BY VIDEO",
        "----------------",
        dataframe_to_string(video_df, index=False),
    ]

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
    image = ax.imshow(values)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(conf_df.columns)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(conf_df.index)
    ax.set_title("Experiment 2 - Global Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            ax.text(column_index, row_index, str(values[row_index, column_index]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


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
        return

    plot_df = df.sort_values(metric_column, ascending=False)

    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 0.8), 5))
    ax.bar(plot_df["normalized_question_category"], plot_df[metric_column])

    ax.set_title(title)
    ax.set_xlabel("Normalized question category")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)

    for index, value in enumerate(plot_df[metric_column].tolist()):
        ax.text(index, value, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_evaluation_artifacts(metrics_summary: JsonDict, output_dir: str | Path) -> None:
    """
    Save compact evaluation artifacts for experiment 2.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    global_df = build_global_metrics_dataframe(metrics_summary)
    category_df = build_category_metrics_dataframe(metrics_summary)
    video_df = build_video_metrics_dataframe(metrics_summary)
    global_conf_df = build_confusion_matrix_dataframe(metrics_summary["global"])

    workbook_path = output_dir / "2_evaluation_summary.xlsx"
    global_csv_path = output_dir / "global_metrics.csv"
    category_csv_path = output_dir / "normalized_question_category_metrics.csv"
    video_csv_path = output_dir / "video_metrics.csv"

    with pd.ExcelWriter(workbook_path) as writer:
        global_df.to_excel(writer, sheet_name="global_metrics", index=False)
        category_df.to_excel(writer, sheet_name="category_metrics", index=False)
        video_df.to_excel(writer, sheet_name="video_metrics", index=False)
        global_conf_df.to_excel(writer, sheet_name="global_confusion")

    save_dataframe_csv(global_df, global_csv_path)
    save_dataframe_csv(category_df, category_csv_path)
    save_dataframe_csv(video_df, video_csv_path)

    plot_global_confusion_matrix(
        metrics_summary,
        plots_dir / "experiment_2_global_confusion_matrix.png",
    )
    plot_metric_by_category(
        category_df,
        metric_column="overall_accuracy",
        title="Experiment 2 - Accuracy by Question Category",
        ylabel="Accuracy",
        output_path=plots_dir / "experiment_2_accuracy_by_category.png",
    )
    plot_metric_by_category(
        category_df,
        metric_column="f1_label_0",
        title="Experiment 2 - F1 Score for Label 0 by Question Category",
        ylabel="F1 label 0",
        output_path=plots_dir / "experiment_2_f1_label_0_by_category.png",
    )
    plot_metric_by_category(
        category_df,
        metric_column="f1_label_1",
        title="Experiment 2 - F1 Score for Label 1 by Question Category",
        ylabel="F1 label 1",
        output_path=plots_dir / "experiment_2_f1_label_1_by_category.png",
    )
    plot_metric_by_category(
        category_df,
        metric_column="invalid_answers",
        title="Experiment 2 - Invalid Answers by Question Category",
        ylabel="Invalid answers",
        output_path=plots_dir / "experiment_2_invalid_answers_by_category.png",
    )


def build_results_json(
    records: Sequence[PredictionRecord],
    transcript_map: Dict[str, str],
    metrics_summary: JsonDict,
) -> JsonDict:
    """
    Build the final per-video JSON output.
    """
    print("[INFO] Building final predictions JSON...")
    results: JsonDict = {}

    for record in records:
        if record.video_id not in results:
            results[record.video_id] = {
                "transcript": transcript_map.get(record.video_id, "Trascrizione non disponibile."),
                "video_metrics": metrics_summary["by_video"].get(record.video_id, {}),
                "questions": {},
            }

        video_questions = results[record.video_id]["questions"]
        if record.question_category not in video_questions:
            video_questions[record.question_category] = {}

        video_questions[record.question_category][record.pool_key] = {
            "normalized_question_category": record.normalized_question_category,
            "0": record.answer_0,
            "1": record.answer_1,
            "target": record.target,
            "prompt": record.prompt,
            "raw_model_output": record.raw_model_output,
            "predicted_label": record.predicted_label,
            "is_correct": record.is_correct,
        }

    results["metrics"] = metrics_summary
    return results


def run_inference(
    examples: Sequence[Example],
    transcript_map: Dict[str, str],
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    batch_size: int = 16,
) -> List[PredictionRecord]:
    """
    Run batched inference on all extracted examples.
    """
    print("[INFO] Starting inference...")
    records: List[PredictionRecord] = []
    total_examples = len(examples)
    total_batches = (total_examples + batch_size - 1) // batch_size

    for start_index in range(0, total_examples, batch_size):
        batch_examples = examples[start_index:start_index + batch_size]
        batch_number = (start_index // batch_size) + 1

        print(
            f"[INFO] Processing batch {batch_number}/{total_batches} "
            f"({start_index + 1}-{start_index + len(batch_examples)} / {total_examples})"
        )

        prompts: List[str] = []
        transcripts: List[str] = []

        for example in batch_examples:
            transcript = transcript_map.get(example.video_id, "Trascrizione non disponibile.")
            prompt = build_prompt(
                answer_0=example.answer_0,
                answer_1=example.answer_1,
                transcript=transcript,
            )
            transcripts.append(transcript)
            prompts.append(prompt)

        raw_outputs = generate_answers_batch(
            model=model,
            processor=processor,
            prompts=prompts,
        )

        for example, transcript, prompt, raw_output in zip(batch_examples, transcripts, prompts, raw_outputs):
            safe_raw_output = normalize_raw_output(raw_output)
            predicted_label = parse_binary_answer(safe_raw_output)
            is_correct = predicted_label == example.target

            records.append(
                PredictionRecord(
                    video_id=example.video_id,
                    question_category=example.question_category,
                    normalized_question_category=example.normalized_question_category,
                    pool_key=example.pool_key,
                    answer_0=example.answer_0,
                    answer_1=example.answer_1,
                    target=example.target,
                    transcript=transcript,
                    prompt=prompt,
                    raw_model_output=safe_raw_output,
                    predicted_label=predicted_label,
                    is_correct=is_correct,
                )
            )

    print("[INFO] Inference completed.")
    return records


def main() -> None:
    print("[INFO] Script started.")
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Input files
    transcripts_path = Path("Data/TranscriptionData/final_classification/manual_revision.json")
    dataset_path = Path("Data/Dataset/maia_ita_mc_by_video_category_pool.json")

    # Output files
    predictions_output_path = Path("Data/ModelResponse/2/qwen_mc_2_predictions_by_video.json")
    metrics_report_output_path = Path("Data/ModelResponse/2/qwen_mc_2_metrics_report.txt")
    evaluation_output_dir = Path("Data/ModelResponse/2/evaluation")

    # Inference configuration
    batch_size = 16

    transcription_json = load_json_file(transcripts_path)
    dataset_json = load_json_file(dataset_path)

    transcript_map = load_transcript_map(transcription_json)
    examples = extract_examples(dataset_json)
    if not examples:
        raise ValueError("No examples were found in the dataset JSON.")

    model, processor = load_model(model_name)

    records = run_inference(
        examples=examples,
        transcript_map=transcript_map,
        model=model,
        processor=processor,
        batch_size=batch_size,
    )

    metrics_summary = compute_all_metrics(records)
    predictions_json = build_results_json(
        records=records,
        transcript_map=transcript_map,
        metrics_summary=metrics_summary,
    )

    save_json_file(predictions_json, predictions_output_path)

    metrics_report = build_metrics_report(metrics_summary)
    save_text_file(metrics_report, metrics_report_output_path)

    save_evaluation_artifacts(metrics_summary, evaluation_output_dir)

    print("[INFO] Script finished successfully.")


if __name__ == "__main__":
    main()