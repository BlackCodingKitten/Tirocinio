
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoProcessor, PreTrainedTokenizerBase, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Select the GPU to use.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

JsonDict = Dict[str, Any]
EMPTY_RAW_OUTPUT_TOKEN = "[[EMPTY_OUTPUT]]"
PROMPT_TEMPLATE_VERSION = "video_plus_transcript_strict_v1"


@dataclass(frozen=True)
class Example:
    video_id: str
    question_category: str
    normalized_question_category: str
    pool_key: str
    choice_0: str
    choice_1: str
    target: int


@dataclass(frozen=True)
class PredictionRecord:
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
    device_map: str = "cuda:1",
    attn_implementation: str = "flash_attention_2",
) -> tuple[
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    PreTrainedTokenizerBase,
]:
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
    tokenizer = cast(PreTrainedTokenizerBase, processor.tokenizer)

    print("[INFO] Model loaded successfully.")
    return model, processor, tokenizer


def get_model_input_device(model: Qwen2_5_VLForConditionalGeneration) -> torch.device:
    try:
        first_parameter = next(model.parameters())
        return first_parameter.device
    except StopIteration:
        return torch.device("cpu")


def load_json_file(json_path: str | Path) -> JsonDict:
    path = Path(json_path)
    print(f"[INFO] Loading JSON file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = cast(JsonDict, json.load(f))
    print(f"[INFO] JSON file loaded successfully: {path}")
    return data


def save_json_file(data: JsonDict, json_path: str | Path) -> None:
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving JSON output to: {path}")
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] JSON output saved successfully: {path}")


def save_text_file(text: str, text_path: str | Path) -> None:
    path = Path(text_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving text report to: {path}")
    with path.open("w", encoding="utf-8") as f:
        f.write(text)
    print(f"[INFO] Text report saved successfully: {path}")


def save_dataframe_csv(df: pd.DataFrame, csv_path: str | Path) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving DataFrame CSV to: {path}")
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[INFO] DataFrame CSV saved successfully: {path}")


def natural_video_sort_key(video_id: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)", video_id)
    if match:
        return int(match.group(1)), video_id
    return 10**9, video_id


def natural_pool_sort_key(pool_key: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)", pool_key)
    if match:
        return int(match.group(1)), pool_key
    return 10**9, pool_key


def normalize_question_category(category: str) -> str:
    stripped = category.strip()
    match = re.match(r"^(.*?)(?:[_\-\s]+[AaBb])$", stripped)
    if match:
        return match.group(1).strip("_- ")
    return stripped


def extract_examples(dataset_json: JsonDict) -> List[Example]:
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

    examples.sort(
        key=lambda ex: (
            natural_video_sort_key(ex.video_id),
            ex.normalized_question_category,
            ex.question_category,
            natural_pool_sort_key(ex.pool_key),
        )
    )

    print(f"[INFO] Extracted and sorted {len(examples)} examples.")
    return examples


def get_video_metadata(final_results: JsonDict, video_id: str) -> JsonDict:
    video_data = final_results.get(video_id, {})
    return video_data if isinstance(video_data, dict) else {}


def get_transcript_for_video(final_results: JsonDict, video_id: str) -> str:
    video_data = get_video_metadata(final_results, video_id)
    transcript = str(video_data.get("generated_transcription", "")).strip()
    return transcript if transcript else "Trascrizione non disponibile."


def resolve_video_path(video_root_dir: str | Path, video_id: str) -> Path:
    video_name = str(video_id).strip()
    if not video_name.lower().endswith(".mp4"):
        video_name = f"{video_name}.mp4"

    video_path = (Path(video_root_dir) / video_name).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    return video_path


def path_to_file_uri(path: str | Path) -> str:
    return Path(path).resolve().as_uri()


def build_prompt(choice_0: str, choice_1: str, transcript: str) -> str:
    transcription = transcript.strip() if transcript.strip() else "Trascrizione non disponibile."
    prompt = (
        f"Dato il video, considera anche la trascrizione del suo audio: {transcription}.\n"
        "Scegli la descrizione corretta rispetto al contenuto del video:\n"
        f"0:{choice_0}\n"
        f"1:{choice_1}\n"
        "Rispondi solo con 0 o 1."
    )

    if "{transcription}" in prompt or "{answer_0}" in prompt or "{answer_1}" in prompt:
        raise RuntimeError(f"[BUG] Prompt template sbagliato generato: {prompt}")

    return prompt


def build_message_tv(
    prompt_text: str,
    video_path: str | Path,
    max_pixels: int = 360 * 420,
    fps: float = 2.0,
) -> List[Dict[str, Any]]:
    return [
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
            "content": [
                {
                    "type": "video",
                    "video": path_to_file_uri(video_path),
                    "max_pixels": max_pixels,
                    "fps": fps,
                },
                {
                    "type": "text",
                    "text": prompt_text,
                },
            ],
        },
    ]


def build_messages(
    prompts: Sequence[str],
    video_paths: Sequence[Path],
    max_pixels: int = 360 * 420,
    fps: float = 2.0,
) -> List[List[Dict[str, Any]]]:
    if len(prompts) != len(video_paths):
        raise ValueError("prompts and video_paths must have the same length.")

    return [
        build_message_tv(
            prompt_text=prompt,
            video_path=video_path,
            max_pixels=max_pixels,
            fps=fps,
        )
        for prompt, video_path in zip(prompts, video_paths)
    ]


def message_contains_video(message: Sequence[Dict[str, Any]]) -> bool:
    for turn in message:
        content = turn.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "video":
                    return True
    return False


def process_visual_batch_strict(
    messages: Sequence[List[Dict[str, Any]]],
) -> Tuple[Any, Any, Dict[str, Any]]:
    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
        )
        return image_inputs, video_inputs, video_kwargs
    except TypeError:
        image_inputs, video_inputs = process_vision_info(messages)
        return image_inputs, video_inputs, {}


def move_inputs_to_device(batch_inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch_inputs.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def assert_video_features_present(
    batch_inputs: Dict[str, Any],
    video_paths: Sequence[Path],
) -> None:
    required_keys = ("pixel_values_videos", "video_grid_thw")
    missing = [key for key in required_keys if key not in batch_inputs]
    if missing:
        available = ", ".join(sorted(batch_inputs.keys()))
        raise RuntimeError(
            "[FATAL] Video features are missing from processor output. "
            f"Missing keys: {missing}. Available keys: {available}. "
            "The model is NOT seeing the videos."
        )

    pixel_values_videos = batch_inputs["pixel_values_videos"]
    video_grid_thw = batch_inputs["video_grid_thw"]

    if not isinstance(pixel_values_videos, torch.Tensor) or pixel_values_videos.numel() == 0:
        raise RuntimeError("[FATAL] pixel_values_videos is empty or invalid.")

    if not isinstance(video_grid_thw, torch.Tensor) or video_grid_thw.numel() == 0:
        raise RuntimeError("[FATAL] video_grid_thw is empty or invalid.")

    print("[DEBUG] Video tensors detected correctly.")
    print(f"[DEBUG] pixel_values_videos.shape = {tuple(pixel_values_videos.shape)}")
    print(f"[DEBUG] video_grid_thw.shape      = {tuple(video_grid_thw.shape)}")
    print(f"[DEBUG] videos in batch           = {len(video_paths)}")


def log_prompt_debug_info(prompts: Sequence[str]) -> None:
    if prompts:
        print("[DEBUG] First prompt in current batch:")
        print(prompts[0])


def generate_answers_batch(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    prompts: Sequence[str],
    video_paths: Sequence[Path],
    max_new_tokens: int = 4,
    max_pixels: int = 360 * 420,
    fps: float = 2.0,
    debug_visual_inputs: bool = False,
) -> List[str]:
    if len(prompts) != len(video_paths):
        raise ValueError("prompts and video_paths must have the same length.")

    messages = build_messages(
        prompts=prompts,
        video_paths=video_paths,
        max_pixels=max_pixels,
        fps=fps,
    )

    rendered_texts = [
        processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        for message in messages
    ]

    image_inputs, video_inputs, video_kwargs = process_visual_batch_strict(messages)

    processor_kwargs: Dict[str, Any] = {
        "text": rendered_texts,
        "padding": True,
        "return_tensors": "pt",
    }

    if image_inputs is not None:
        processor_kwargs["images"] = image_inputs

    if video_inputs is not None:
        processor_kwargs["videos"] = video_inputs
        processor_kwargs["fps"] = fps

    if video_kwargs:
        processor_kwargs.update(video_kwargs)

    inputs = processor(**processor_kwargs)

    if any(message_contains_video(message) for message in messages):
        assert_video_features_present(inputs, video_paths)

    target_device = get_model_input_device(model)
    inputs = move_inputs_to_device(inputs, target_device)

    if debug_visual_inputs:
        log_prompt_debug_info(prompts)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    trimmed_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_texts = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return [text.strip() for text in output_texts]


def normalize_raw_output(raw_output: Optional[str]) -> str:
    if raw_output is None:
        return EMPTY_RAW_OUTPUT_TOKEN

    cleaned = str(raw_output).strip()
    return cleaned if cleaned else EMPTY_RAW_OUTPUT_TOKEN


def parse_binary_answer(raw_output: str) -> Optional[int]:
    cleaned = raw_output.strip()

    if cleaned in {"0", "1"}:
        return int(cleaned)

    matches = re.findall(r"\b([01])\b", cleaned)
    unique_matches = set(matches)

    if len(unique_matches) == 1:
        return int(matches[0])

    return None


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else 0.0


def compute_label_metrics(records: Sequence[PredictionRecord], label: int) -> Dict[str, float]:
    tp = sum(record.predicted_label == label and record.target == label for record in records)
    fp = sum(record.predicted_label == label and record.target != label for record in records)
    fn = sum(record.target == label and record.predicted_label != label for record in records)

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": float(sum(record.target == label for record in records)),
    }


def compute_confusion_counts(records: Sequence[PredictionRecord]) -> Dict[str, int]:
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


def compute_metrics(records: Sequence[PredictionRecord]) -> JsonDict:
    print(f"[INFO] Computing metrics for {len(records)} records...")

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
        "support_label_0": int(label_0_metrics["support"]),
        "precision_label_0": label_0_metrics["precision"],
        "recall_label_0": label_0_metrics["recall"],
        "f1_label_0": label_0_metrics["f1"],
        "support_label_1": int(label_1_metrics["support"]),
        "precision_label_1": label_1_metrics["precision"],
        "recall_label_1": label_1_metrics["recall"],
        "f1_label_1": label_1_metrics["f1"],
        **confusion,
    }


def compute_all_metrics(records: Sequence[PredictionRecord]) -> JsonDict:
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
    return {
        entity_column_name: entity_name,
        "total_examples": metrics["total_examples"],
        "valid_answers": metrics["valid_answers"],
        "invalid_answers": metrics["invalid_answers"],
        "empty_answers": metrics["empty_answers"],
        "overall_accuracy": metrics["overall_accuracy"],
        "valid_only_accuracy": metrics["valid_only_accuracy"],
        "support_label_0": metrics["support_label_0"],
        "precision_label_0": metrics["precision_label_0"],
        "recall_label_0": metrics["recall_label_0"],
        "f1_label_0": metrics["f1_label_0"],
        "support_label_1": metrics["support_label_1"],
        "precision_label_1": metrics["precision_label_1"],
        "recall_label_1": metrics["recall_label_1"],
        "f1_label_1": metrics["f1_label_1"],
        "true_0_pred_0": metrics["true_0_pred_0"],
        "true_0_pred_1": metrics["true_0_pred_1"],
        "true_1_pred_0": metrics["true_1_pred_0"],
        "true_1_pred_1": metrics["true_1_pred_1"],
    }


def build_global_metrics_dataframe(metrics_summary: JsonDict) -> pd.DataFrame:
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
    return pd.DataFrame(
        [
            [metrics["true_0_pred_0"], metrics["true_0_pred_1"]],
            [metrics["true_1_pred_0"], metrics["true_1_pred_1"]],
        ],
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    )


def build_category_confusion_dataframe(metrics_summary: JsonDict) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for category_name, metrics in metrics_summary["by_normalized_question_category"].items():
        rows.append(
            {
                "normalized_question_category": category_name,
                "true_0_pred_0": metrics["true_0_pred_0"],
                "true_0_pred_1": metrics["true_0_pred_1"],
                "true_1_pred_0": metrics["true_1_pred_0"],
                "true_1_pred_1": metrics["true_1_pred_1"],
            }
        )

    return pd.DataFrame(rows)


def dataframe_to_string(df: pd.DataFrame, index: bool = False) -> str:
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 240,
        "display.max_colwidth", 240,
    ):
        return df.to_string(index=index, float_format=lambda x: f"{x:.4f}")


def build_metrics_report(metrics_summary: JsonDict) -> str:
    global_df = build_global_metrics_dataframe(metrics_summary)
    category_df = build_category_metrics_dataframe(metrics_summary)
    video_df = build_video_metrics_dataframe(metrics_summary)
    global_conf_df = build_confusion_matrix_dataframe(metrics_summary["global"])
    category_conf_df = build_category_confusion_dataframe(metrics_summary)

    report_parts: List[str] = [
        "MODEL EVALUATION REPORT - EXPERIMENT 4",
        "======================================",
        "",
        f"PROMPT_TEMPLATE_VERSION: {PROMPT_TEMPLATE_VERSION}",
        f"EMPTY_RAW_OUTPUT_TOKEN: {EMPTY_RAW_OUTPUT_TOKEN}",
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

    return "\n".join(report_parts)


def plot_global_confusion_matrix(metrics_summary: JsonDict, output_path: str | Path) -> None:
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
    ax.set_title("Experiment 4 - Global Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, str(values[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
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

    values = plot_df[metric_column].tolist()
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_evaluation_artifacts(
    metrics_summary: JsonDict,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    global_df = build_global_metrics_dataframe(metrics_summary)
    category_df = build_category_metrics_dataframe(metrics_summary)
    video_df = build_video_metrics_dataframe(metrics_summary)
    global_conf_df = build_confusion_matrix_dataframe(metrics_summary["global"])
    category_conf_df = build_category_confusion_dataframe(metrics_summary)

    workbook_path = output_dir / "4_evaluation_summary.xlsx"
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
        metric_column="overall_accuracy",
        title="Experiment 4 - Accuracy by Question Category",
        ylabel="Overall accuracy",
        output_path=plots_dir / "accuracy_by_category.png",
    )
    plot_metric_by_category(
        category_df,
        metric_column="f1_label_0",
        title="Experiment 4 - F1 Score for Label 0 by Question Category",
        ylabel="F1 label 0",
        output_path=plots_dir / "f1_label_0_by_category.png",
    )
    plot_metric_by_category(
        category_df,
        metric_column="f1_label_1",
        title="Experiment 4 - F1 Score for Label 1 by Question Category",
        ylabel="F1 label 1",
        output_path=plots_dir / "f1_label_1_by_category.png",
    )
    plot_metric_by_category(
        category_df,
        metric_column="invalid_answers",
        title="Experiment 4 - Invalid Answers by Question Category",
        ylabel="Invalid answers",
        output_path=plots_dir / "invalid_answers_by_category.png",
    )

    print(f"[INFO] Evaluation workbook saved to: {workbook_path}")


def build_results_json(
    records: Sequence[PredictionRecord],
    final_results: JsonDict,
    metrics_summary: JsonDict,
    fps: float,
    max_pixels: int,
) -> JsonDict:
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
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "inference_config": {
                    "fps": fps,
                    "max_pixels": max_pixels,
                },
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
            "transcript_used": record.transcript,
            "prompt": record.prompt,
            "raw_model_output": record.raw_model_output,
            "predicted_label": record.predicted_label,
            "is_correct": record.is_correct,
        }

    return results


def print_example_result(record: PredictionRecord) -> None:
    correctness = "CORRECT" if record.is_correct else "WRONG"
    parsed_label = record.predicted_label if record.predicted_label is not None else "INVALID"

    print(
        "[RESULT] "
        f"video={record.video_id} | "
        f"category={record.question_category} | "
        f"pool={record.pool_key}"
    )
    print(f"[RESULT] transcript     = {record.transcript}")
    print(f"[RESULT] raw_output     = {record.raw_model_output}")
    print(f"[RESULT] parsed_label   = {parsed_label}")
    print(f"[RESULT] target         = {record.target}")
    print(f"[RESULT] correctness    = {correctness}")
    print("-" * 100)


def run_inference(
    examples: Sequence[Example],
    final_results: JsonDict,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    video_root_dir: str | Path,
    batch_size: int = 8,
    max_pixels: int = 360 * 420,
    fps: float = 2.0,
    debug_visual_inputs: bool = True,
) -> List[PredictionRecord]:
    records: List[PredictionRecord] = []
    total_examples = len(examples)
    total_batches = (total_examples + batch_size - 1) // batch_size

    print(f"[INFO] Total examples: {total_examples}")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Total batches: {total_batches}")

    for start_idx in range(0, total_examples, batch_size):
        batch_examples = examples[start_idx:start_idx + batch_size]
        batch_number = (start_idx // batch_size) + 1

        print(
            f"[INFO] Processing batch {batch_number}/{total_batches} "
            f"(examples {start_idx + 1}-{start_idx + len(batch_examples)} of {total_examples})"
        )

        prompts: List[str] = []
        transcripts: List[str] = []
        video_paths: List[Path] = []

        for example in batch_examples:
            transcript = get_transcript_for_video(final_results, example.video_id)
            prompt = build_prompt(
                choice_0=example.choice_0,
                choice_1=example.choice_1,
                transcript=transcript,
            )
            video_path = resolve_video_path(video_root_dir, example.video_id)

            transcripts.append(transcript)
            prompts.append(prompt)
            video_paths.append(video_path)

        raw_outputs = generate_answers_batch(
            model=model,
            processor=processor,
            prompts=prompts,
            video_paths=video_paths,
            max_new_tokens=4,
            max_pixels=max_pixels,
            fps=fps,
            debug_visual_inputs=debug_visual_inputs,
        )

        for example, transcript, prompt, raw_output in zip(
            batch_examples,
            transcripts,
            prompts,
            raw_outputs,
        ):
            safe_raw_output = normalize_raw_output(raw_output)
            predicted_label = parse_binary_answer(safe_raw_output)
            is_correct = predicted_label == example.target

            record = PredictionRecord(
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
            records.append(record)
            print_example_result(record)

        print(f"[INFO] Processed {min(start_idx + len(batch_examples), total_examples)}/{total_examples}")

    return records


def main() -> None:
    print("[INFO] Script started.")
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    final_results_path = Path("Data/TranscriptionData/final_classification/final_results.json")
    dataset_path = Path("Data/Dataset/maia_ita_mc_by_video_category_pool.json")
    video_root_dir = Path("Data/Videos")

    predictions_output_path = Path("Data/ModelResponse/4/qwen_mc_video_transcript_predictions_by_video.json")
    metrics_report_output_path = Path("Data/ModelResponse/4/qwen_mc_video_transcript_metrics_report.txt")
    evaluation_output_dir = Path("Data/ModelResponse/4/evaluation")

    batch_size = 2
    max_pixels = 512 * 512
    fps = 2.0
    debug_visual_inputs = True

    if predictions_output_path.exists():
        predictions_output_path.unlink()
    if metrics_report_output_path.exists():
        metrics_report_output_path.unlink()

    final_results = load_json_file(final_results_path)
    dataset_json = load_json_file(dataset_path)

    examples = extract_examples(dataset_json)
    if not examples:
        raise ValueError("No examples were found in the dataset JSON.")

    model, processor, tokenizer = load_model(model_name)
    print(f"[INFO] Tokenizer loaded: {type(tokenizer).__name__}")
    print(f"[INFO] Prompt template version: {PROMPT_TEMPLATE_VERSION}")

    records = run_inference(
        examples=examples,
        final_results=final_results,
        model=model,
        processor=processor,
        video_root_dir=video_root_dir,
        batch_size=batch_size,
        max_pixels=max_pixels,
        fps=fps,
        debug_visual_inputs=debug_visual_inputs,
    )

    metrics_summary = compute_all_metrics(records)

    predictions_json = build_results_json(
        records=records,
        final_results=final_results,
        metrics_summary=metrics_summary,
        fps=fps,
        max_pixels=max_pixels,
    )

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
