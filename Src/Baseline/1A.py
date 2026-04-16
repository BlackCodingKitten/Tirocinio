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
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


JsonDict = Dict[str, Any]
EMPTY_RAW_OUTPUT_TOKEN = "[[EMPTY_OUTPUT]]"
EXPERIMENT_NAME = "1A"


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
    Flat prediction record used for JSON export and metric computation.
    raw_model_output is always forced to be a non-empty string.
    """

    video_id: str
    question_category: str
    normalized_question_category: str
    pool_key: str
    choice_0: str
    choice_1: str
    target: int
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
    """
    Load the Qwen model and its processor.
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
        data = json.load(file)
    print(f"[INFO] JSON file loaded successfully: {path}")
    return data


def save_json_file(data: JsonDict, json_path: str | Path) -> None:
    """
    Save a JSON file to disk using UTF-8 and readable indentation.
    """
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving JSON output to: {path}")
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    print(f"[INFO] JSON output saved successfully: {path}")


def save_text_file(text: str, text_path: str | Path) -> None:
    """
    Save a plain text file to disk.
    """
    path = Path(text_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving text report to: {path}")
    with path.open("w", encoding="utf-8") as file:
        file.write(text)
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
    Normalize question_category by removing a trailing A/B suffix when present.
    """
    stripped = category.strip()
    match = re.match(r"^(.*?)(?:[_\-\s]+[AaBb])$", stripped)
    if match:
        return match.group(1).strip("_- ")
    return stripped


def extract_examples(dataset_json: JsonDict) -> List[Example]:
    """
    Extract all examples from the nested dataset JSON.
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

                examples.append(
                    Example(
                        video_id=video_id,
                        question_category=question_category,
                        normalized_question_category=normalized_category,
                        pool_key=pool_key,
                        choice_0=str(payload["0"]),
                        choice_1=str(payload["1"]),
                        target=int(payload["target"]),
                    )
                )

    examples.sort(
        key=lambda example: (
            natural_video_sort_key(example.video_id),
            example.normalized_question_category,
            example.question_category,
            natural_pool_sort_key(example.pool_key),
        )
    )

    print(f"[INFO] Extracted and sorted {len(examples)} examples.")
    return examples


def build_prompt(choice_0: str, choice_1: str) -> str:
    """
    Build the prompt for experiment 1A.
    """
    return (
        "Scegli la descrizione corretta rispetto al contenuto del video:\n"
        f"0:{choice_0}\n"
        f"1:{choice_1}\n"
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
    max_new_tokens: int = 8,
) -> List[str]:
    """
    Run batched generation for a list of prompts.
    """
    print(f"[INFO] Generating answers for a batch of {len(prompts)} prompts...")
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
    Extract the first valid binary answer from the model output.
    """
    cleaned = raw_output.strip()

    if cleaned in {"0", "1"}:
        return int(cleaned)

    match = re.search(r"\b([01])\b", cleaned)
    if match:
        return int(match.group(1))

    if cleaned and cleaned[0] in {"0", "1"}:
        return int(cleaned[0])

    return None


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Safe floating-point division.
    """
    return numerator / denominator if denominator != 0 else 0.0


def compute_binary_label_metrics(
    records: Sequence[PredictionRecord],
    label: int,
) -> Dict[str, float]:
    """
    Compute precision, recall and F1 for a specific binary label.
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
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_metrics(records: Sequence[PredictionRecord]) -> JsonDict:
    """
    Compute a compact set of metrics for a subset of records.
    """
    print(f"[INFO] Computing metrics for {len(records)} records...")

    sample_count = len(records)
    invalid_prediction_count = sum(record.predicted_label not in (0, 1) for record in records)
    valid_prediction_count = sample_count - invalid_prediction_count
    correct_prediction_count = sum(record.is_correct for record in records)

    true_negative = sum(record.target == 0 and record.predicted_label == 0 for record in records)
    false_positive = sum(record.target == 0 and record.predicted_label == 1 for record in records)
    false_negative = sum(record.target == 1 and record.predicted_label == 0 for record in records)
    true_positive = sum(record.target == 1 and record.predicted_label == 1 for record in records)

    label_0_metrics = compute_binary_label_metrics(records, label=0)
    label_1_metrics = compute_binary_label_metrics(records, label=1)

    return {
        "sample_count": sample_count,
        "valid_prediction_count": valid_prediction_count,
        "invalid_prediction_count": invalid_prediction_count,
        "invalid_prediction_rate": safe_divide(invalid_prediction_count, sample_count),
        "overall_accuracy": safe_divide(correct_prediction_count, sample_count),
        "valid_accuracy": safe_divide(true_negative + true_positive, valid_prediction_count),
        "precision_label_0": label_0_metrics["precision"],
        "recall_label_0": label_0_metrics["recall"],
        "f1_label_0": label_0_metrics["f1"],
        "precision_label_1": label_1_metrics["precision"],
        "recall_label_1": label_1_metrics["recall"],
        "f1_label_1": label_1_metrics["f1"],
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }


def compute_all_metrics(records: Sequence[PredictionRecord]) -> JsonDict:
    """
    Compute global metrics, metrics by normalized question category and by video.
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
            key=lambda item: natural_video_sort_key(item[0]),
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
    Convert a metric summary into a flat row for a DataFrame.
    """
    return {
        entity_column_name: entity_name,
        "sample_count": metrics["sample_count"],
        "valid_prediction_count": metrics["valid_prediction_count"],
        "invalid_prediction_count": metrics["invalid_prediction_count"],
        "invalid_prediction_rate": metrics["invalid_prediction_rate"],
        "overall_accuracy": metrics["overall_accuracy"],
        "valid_accuracy": metrics["valid_accuracy"],
        "precision_label_0": metrics["precision_label_0"],
        "recall_label_0": metrics["recall_label_0"],
        "f1_label_0": metrics["f1_label_0"],
        "precision_label_1": metrics["precision_label_1"],
        "recall_label_1": metrics["recall_label_1"],
        "f1_label_1": metrics["f1_label_1"],
        "true_negative": metrics["true_negative"],
        "false_positive": metrics["false_positive"],
        "false_negative": metrics["false_negative"],
        "true_positive": metrics["true_positive"],
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
    Build a 2x2 confusion matrix DataFrame from a metric summary.
    """
    return pd.DataFrame(
        [
            [metrics["true_negative"], metrics["false_positive"]],
            [metrics["false_negative"], metrics["true_positive"]],
        ],
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )


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
        return df.to_string(index=index, float_format=lambda value: f"{value:.4f}")


def build_metrics_report(metrics_summary: JsonDict) -> str:
    """
    Build a readable TXT report for experiment 1A.
    """
    print("[INFO] Building metrics text report...")

    global_df = build_global_metrics_dataframe(metrics_summary)
    category_df = build_category_metrics_dataframe(metrics_summary)
    video_df = build_video_metrics_dataframe(metrics_summary)
    global_conf_df = build_confusion_matrix_dataframe(metrics_summary["global"])

    report_parts: List[str] = [
        f"EXPERIMENT {EXPERIMENT_NAME} EVALUATION REPORT",
        "=" * 32,
        "",
        "METRIC GUIDE",
        "------------",
        "- overall_accuracy: accuracy computed on all samples.",
        "- valid_accuracy: accuracy computed only on outputs successfully parsed as 0 or 1.",
        "- invalid_prediction_rate: share of outputs that could not be parsed as 0 or 1.",
        "- precision/recall/F1 are reported separately for label 0 and label 1.",
        "- The confusion matrix follows the convention actual rows / predicted columns.",
        "",
        f"EXPERIMENT {EXPERIMENT_NAME} - GLOBAL METRICS",
        "--------------------------------",
        dataframe_to_string(global_df, index=False),
        "",
        f"EXPERIMENT {EXPERIMENT_NAME} - GLOBAL CONFUSION MATRIX",
        "-----------------------------------------",
        dataframe_to_string(global_conf_df, index=True),
        "",
        f"EXPERIMENT {EXPERIMENT_NAME} - METRICS BY NORMALIZED QUESTION CATEGORY",
        "-----------------------------------------------------------",
        dataframe_to_string(category_df, index=False),
        "",
        f"EXPERIMENT {EXPERIMENT_NAME} - METRICS BY VIDEO",
        "------------------------------------",
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
    ax.set_title(f"Experiment {EXPERIMENT_NAME} - Global confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            ax.text(column_index, row_index, str(values[row_index, column_index]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
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

    for index, value in enumerate(plot_df[metric_column].tolist()):
        ax.text(index, value, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Plot saved to: {output_path}")


def save_evaluation_artifacts(
    metrics_summary: JsonDict,
    output_dir: str | Path,
) -> None:
    """
    Save the main evaluation artifacts for experiment 1A.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    global_df = build_global_metrics_dataframe(metrics_summary)
    category_df = build_category_metrics_dataframe(metrics_summary)
    video_df = build_video_metrics_dataframe(metrics_summary)
    global_conf_df = build_confusion_matrix_dataframe(metrics_summary["global"])

    workbook_path = output_dir / f"{EXPERIMENT_NAME}_evaluation_summary.xlsx"
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
        plots_dir / "global_confusion_matrix.png",
    )
    plot_metric_by_category(
        category_df,
        metric_column="overall_accuracy",
        title=f"Experiment {EXPERIMENT_NAME} - Overall accuracy by normalized question category",
        ylabel="Overall accuracy",
        output_path=plots_dir / "overall_accuracy_by_category.png",
    )
    plot_metric_by_category(
        category_df,
        metric_column="invalid_prediction_rate",
        title=f"Experiment {EXPERIMENT_NAME} - Invalid prediction rate by normalized question category",
        ylabel="Invalid prediction rate",
        output_path=plots_dir / "invalid_prediction_rate_by_category.png",
    )
    plot_metric_by_category(
        category_df,
        metric_column="f1_label_1",
        title=f"Experiment {EXPERIMENT_NAME} - F1 score for label 1 by normalized question category",
        ylabel="F1 label 1",
        output_path=plots_dir / "f1_label_1_by_category.png",
    )

    print(f"[INFO] Evaluation workbook saved to: {workbook_path}")
    print(f"[INFO] Evaluation CSVs saved in: {output_dir}")
    print(f"[INFO] Evaluation plots saved in: {plots_dir}")


def build_results_json(
    records: Sequence[PredictionRecord],
    metrics_summary: JsonDict,
) -> JsonDict:
    """
    Build the final per-video JSON output.
    """
    print("[INFO] Building final results JSON structure...")
    results: JsonDict = {}

    for record in records:
        if record.video_id not in results:
            results[record.video_id] = {
                "experiment": EXPERIMENT_NAME,
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

    return results


def run_inference(
    examples: Sequence[Example],
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

    for start_index in range(0, total_examples, batch_size):
        batch_examples = examples[start_index:start_index + batch_size]
        batch_number = (start_index // batch_size) + 1

        print(
            f"[INFO] Processing batch {batch_number}/{total_batches} "
            f"(examples {start_index + 1}-{start_index + len(batch_examples)} of {total_examples})..."
        )

        prompts = [
            build_prompt(choice_0=example.choice_0, choice_1=example.choice_1)
            for example in batch_examples
        ]

        raw_outputs = generate_answers_batch(
            model=model,
            processor=processor,
            prompts=prompts,
        )

        for example, prompt, raw_output in zip(batch_examples, prompts, raw_outputs):
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
                    prompt=prompt,
                    raw_model_output=safe_raw_output,
                    predicted_label=predicted_label,
                    is_correct=is_correct,
                )
            )

        processed_examples = min(start_index + len(batch_examples), total_examples)
        print(f"[INFO] Processed {processed_examples}/{total_examples} examples so far.")

    print("[INFO] Inference completed for all examples.")
    return records


def main() -> None:
    """
    Main entry point for experiment 1A.
    """
    print("[INFO] Script started.")
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    dataset_path = Path("Data/Dataset/maia_ita_mc_by_video_category_pool.json")

    predictions_output_path = Path("Data/ModelResponse/1A/qwen_mc_1A_predictions_by_video.json")
    metrics_report_output_path = Path("Data/ModelResponse/1A/qwen_mc_1A_metrics_report.txt")
    evaluation_output_dir = Path("Data/ModelResponse/1A/evaluation")

    batch_size = 4

    print("[INFO] Configuration loaded.")
    print(f"[INFO] dataset_path: {dataset_path}")
    print(f"[INFO] predictions_output_path: {predictions_output_path}")
    print(f"[INFO] metrics_report_output_path: {metrics_report_output_path}")
    print(f"[INFO] evaluation_output_dir: {evaluation_output_dir}")
    print(f"[INFO] batch_size: {batch_size}")

    dataset_json = load_json_file(dataset_path)
    examples = extract_examples(dataset_json)
    if not examples:
        raise ValueError("No examples were found in the dataset JSON.")

    print(f"[INFO] Total extracted examples: {len(examples)}")

    model, processor = load_model(model_name)

    records = run_inference(
        examples=examples,
        model=model,
        processor=processor,
        batch_size=batch_size,
    )
    print(f"[INFO] Inference phase completed. Total prediction records: {len(records)}")

    metrics_summary = compute_all_metrics(records)
    predictions_json = build_results_json(records=records, metrics_summary=metrics_summary)

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
