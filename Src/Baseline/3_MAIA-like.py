from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, PreTrainedTokenizerBase, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

try:
    from decord import VideoReader, cpu
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "decord is required for this script. Install it with: pip install decord"
    ) from exc


# The physical GPU selected here becomes logical cuda:0 inside the process.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

JsonDict = Dict[str, Any]
EMPTY_RAW_OUTPUT_TOKEN = "[[EMPTY_OUTPUT]]"
PROMPT_TEMPLATE_VERSION = "maia_like_eval_mc_01_v1"


@dataclass(frozen=True)
class Example:
    """Single multiple-choice example extracted from the local MAIA JSON file."""

    video_id: str
    question_category: str
    normalized_question_category: str
    pool_key: str
    choice_0: str
    choice_1: str
    target: int


@dataclass(frozen=True)
class PredictionRecord:
    """Flat prediction record used both for export and metric computation."""

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


@dataclass(frozen=True)
class InferenceConfig:
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    dataset_path: str = "Data/Dataset/maia_ita_mc_by_video_category_pool.json"
    videos_dir: str = "Data/Videos"
    out_dir: str = "Data/ModelResponse/3_MAIA-like"
    batch_size: int = 4
    num_frames: int = 32
    max_image_dimension: int = 900
    max_new_tokens: int = 4
    temperature: float = 0.0
    top_p: float = 1.0
    torch_dtype_name: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    processor_min_pixels: int = 28 * 28
    processor_max_pixels: int = 1280 * 28 * 28


def resolve_torch_dtype(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    mapping: Dict[str, torch.dtype] = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[normalized]


def load_model(
    model_name: str,
    torch_dtype: torch.dtype,
    attn_implementation: str,
    min_pixels: int,
    max_pixels: int,
) -> tuple[
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    PreTrainedTokenizerBase,
]:
    print(f"[INFO] Loading model: {model_name}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="cuda:1",
        attn_implementation=attn_implementation,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_name,
        use_fast=True,
        padding_side="left",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    tokenizer = cast(PreTrainedTokenizerBase, processor.tokenizer)

    print("[INFO] Model ready.")
    return model, processor, tokenizer


def get_model_input_device(model: Qwen2_5_VLForConditionalGeneration) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def load_json_file(json_path: str | Path) -> JsonDict:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        return cast(JsonDict, json.load(f))


def save_json_file(data: JsonDict, json_path: str | Path) -> None:
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(records: Sequence[JsonDict], jsonl_path: str | Path) -> None:
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_text_file(text: str, text_path: str | Path) -> None:
    path = Path(text_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def save_dataframe_csv(df: pd.DataFrame, csv_path: str | Path) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


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
        key=lambda ex: (
            natural_video_sort_key(ex.video_id),
            ex.normalized_question_category,
            ex.question_category,
            natural_pool_sort_key(ex.pool_key),
        )
    )
    return examples


def chunks(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def list_mp4_videos(video_root_dir: str | Path) -> List[Path]:
    root = Path(video_root_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    videos = sorted(root.glob("*.mp4"), key=lambda p: natural_video_sort_key(p.stem))
    if not videos:
        raise ValueError(f"No .mp4 files found in: {root}")
    return videos


def build_video_path_index(video_paths: Sequence[Path]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in video_paths:
        index[path.stem] = path
        index[path.name] = path
    return index


def resolve_video_path(video_id: str, video_path_index: Dict[str, Path]) -> Path:
    video_key = str(video_id).strip()
    path = video_path_index.get(video_key) or video_path_index.get(f"{video_key}.mp4")
    if path is None:
        raise KeyError(f"No video path found for {video_id!r}")
    return path


def resize_max_dim(img: Image.Image, max_dim: int) -> Image.Image:
    width, height = img.size
    scale = min(1.0, float(max_dim) / float(max(width, height)))
    if scale == 1.0:
        return img

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return img.resize((new_width, new_height), Image.BICUBIC)


def sample_video_frames(
    video_path: str | Path,
    num_frames: int,
    max_image_dimension: int,
) -> List[Image.Image]:
    path = str(Path(video_path).resolve())
    vr = VideoReader(path, ctx=cpu(0))
    total_frames = len(vr)
    if total_frames == 0:
        raise ValueError(f"Video has 0 frames: {path}")

    sampled = max(1, min(num_frames, total_frames))
    frame_indices = np.linspace(0, total_frames - 1, num=sampled, dtype=np.int64)
    frame_array = vr.get_batch(frame_indices).asnumpy()

    output: List[Image.Image] = []
    for arr in frame_array:
        image = Image.fromarray(arr, mode="RGB")
        output.append(resize_max_dim(image, max_image_dimension))
    return output


def preprocess_videos(
    examples: Sequence[Example],
    video_path_index: Dict[str, Path],
    num_frames: int,
    max_image_dimension: int,
) -> Dict[str, List[Image.Image]]:
    processed_videos: Dict[str, List[Image.Image]] = {}

    print(f"[INFO] Preprocessing videos to sample {num_frames} frames each...")
    for example in examples:
        if example.video_id in processed_videos:
            continue

        video_path = resolve_video_path(example.video_id, video_path_index)
        processed_videos[example.video_id] = sample_video_frames(
            video_path=video_path,
            num_frames=num_frames,
            max_image_dimension=max_image_dimension,
        )

    print(f"[INFO] Preprocessed {len(processed_videos)} unique videos.")
    return processed_videos


def build_prompt(choice_0: str, choice_1: str) -> str:
    return (
        "Scegli la descrizione corretta rispetto al contenuto del video:\n"
        "0:{choice_0}"
        "1:{choice_1}"
        "Rispondi solo con 0 o 1."
    )


def build_message(frames: Sequence[Image.Image], prompt_text: str) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for frame in frames:
        content.append({"type": "image", "image": frame})
    content.append({"type": "text", "text": prompt_text})

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a precise visual assistant. Reply with a single digit only: 0 or 1.",
                }
            ],
        },
        {
            "role": "user",
            "content": content,
        },
    ]


def move_inputs_to_device(batch_inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch_inputs.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def generate_answers_batch(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    prompts: Sequence[str],
    image_batches: Sequence[Sequence[Image.Image]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    messages = [
        build_message(frames=frames, prompt_text=prompt)
        for prompt, frames in zip(prompts, image_batches)
    ]

    rendered_texts = [
        processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        for message in messages
    ]

    image_inputs, video_inputs = process_vision_info(messages)
    if video_inputs is not None:
        raise RuntimeError("Unexpected video_inputs in image-based MAIA-like evaluation.")

    inputs = processor(
        text=rendered_texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = move_inputs_to_device(inputs, get_model_input_device(model))

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "top_p": top_p,
        "use_cache": True,
    }
    if temperature > 0.0:
        generation_kwargs["temperature"] = temperature

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    trimmed_ids = [
        output_ids[len(input_ids) :]
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


def run_inference(
    examples: Sequence[Example],
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    processed_videos: Dict[str, List[Image.Image]],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[PredictionRecord]:
    records: List[PredictionRecord] = []
    total_examples = len(examples)
    total_batches = (total_examples + batch_size - 1) // batch_size

    print(f"[INFO] Examples: {total_examples}")
    print(f"[INFO] Batches: {total_batches}")

    for batch_idx, batch_examples in enumerate(chunks(list(examples), batch_size), start=1):
        print(f"[INFO] Batch {batch_idx}/{total_batches}")

        prompts: List[str] = []
        image_batches: List[List[Image.Image]] = []

        for example in batch_examples:
            prompts.append(build_prompt(example.choice_0, example.choice_1))
            image_batches.append(processed_videos[example.video_id])

        raw_outputs = generate_answers_batch(
            model=model,
            processor=processor,
            prompts=prompts,
            image_batches=image_batches,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        for example, prompt, raw_output in zip(batch_examples, prompts, raw_outputs):
            safe_raw_output = normalize_raw_output(raw_output)
            predicted_label = parse_binary_answer(safe_raw_output)
            is_correct = predicted_label == example.target

            print("-" * 100)
            print(f"VIDEO: {example.video_id}")
            print(f"CATEGORY: {example.question_category}")
            print(f"POOL: {example.pool_key}")
            print(f"MODEL OUTPUT: {safe_raw_output}")
            print(f"PREDICTED: {predicted_label}")
            print(f"TARGET: {example.target}")
            print(f"CORRECT: {is_correct}")
            print("-" * 100)

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

    return records


def build_results_json(
    records: Sequence[PredictionRecord],
    metrics_summary: JsonDict,
    num_frames: int,
    max_image_dimension: int,
) -> JsonDict:
    results: JsonDict = {}

    for record in records:
        if record.video_id not in results:
            results[record.video_id] = {
                "metrics": metrics_summary["by_video"].get(record.video_id, {}),
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "inference_config": {
                    "num_frames": num_frames,
                    "max_image_dimension": max_image_dimension,
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
            "prompt": record.prompt,
            "raw_model_output": record.raw_model_output,
            "predicted_label": record.predicted_label,
            "is_correct": record.is_correct,
        }

    return results


def build_jsonl_rows(records: Sequence[PredictionRecord]) -> List[JsonDict]:
    rows: List[JsonDict] = []
    for record in records:
        rows.append(
            {
                "id": f"{record.video_id}::{record.question_category}::{record.pool_key}",
                "video_id": record.video_id,
                "question_category": record.question_category,
                "normalized_question_category": record.normalized_question_category,
                "pool_key": record.pool_key,
                "answer1": record.choice_0,
                "answer2": record.choice_1,
                "target": record.target,
                "model_generation": record.raw_model_output,
                "predicted_label": record.predicted_label,
                "is_correct": record.is_correct,
            }
        )
    return rows


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else 0.0


def compute_confusion_counts(records: Sequence[PredictionRecord]) -> Dict[str, int]:
    c00 = sum(record.target == 0 and record.predicted_label == 0 for record in records)
    c01 = sum(record.target == 0 and record.predicted_label == 1 for record in records)
    c10 = sum(record.target == 1 and record.predicted_label == 0 for record in records)
    c11 = sum(record.target == 1 and record.predicted_label == 1 for record in records)

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
    tp = sum(record.predicted_label == label and record.target == label for record in records)
    fp = sum(record.predicted_label == label and record.target != label for record in records)
    fn = sum(record.target == label and record.predicted_label != label for record in records)

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
    n_samples = len(records)
    empty_raw_outputs = sum(record.raw_model_output == EMPTY_RAW_OUTPUT_TOKEN for record in records)
    invalid_predictions = sum(record.predicted_label not in (0, 1) for record in records)
    valid_predictions = n_samples - invalid_predictions
    correct_predictions = sum(record.is_correct for record in records)

    label_0 = compute_label_metrics(records, label=0)
    label_1 = compute_label_metrics(records, label=1)
    confusion_counts = compute_confusion_counts(records)

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
    by_class: Dict[str, List[PredictionRecord]] = {}
    by_video: Dict[str, List[PredictionRecord]] = {}

    for record in records:
        by_class.setdefault(record.normalized_question_category, []).append(record)
        by_video.setdefault(record.video_id, []).append(record)

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
            [metrics["actual_0_pred_0"], metrics["actual_0_pred_1"]],
            [metrics["actual_1_pred_0"], metrics["actual_1_pred_1"]],
        ],
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )


def build_category_confusion_dataframe(metrics_summary: JsonDict) -> pd.DataFrame:
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
        "MODEL EVALUATION REPORT",
        "=======================",
        "",
        f"PROMPT_TEMPLATE_VERSION: {PROMPT_TEMPLATE_VERSION}",
        f"EMPTY_RAW_OUTPUT_TOKEN: {EMPTY_RAW_OUTPUT_TOKEN}",
        "",
        "DEFINITIONS",
        "-----------",
        "- Global metrics summarize the overall model performance on the full dataset.",
        "- Per-video metrics are computed only on the examples of that specific video.",
        "- Per-category metrics are computed only on the examples of that specific normalized question category.",
        "- Accuracy is computed on the entire current subset.",
        "- Accuracy_valid_only is computed only on valid parsed predictions.",
        "- Precision, recall and F1 are reported separately for label 0 and label 1.",
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


def save_evaluation_artifacts(metrics_summary: JsonDict, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    global_df = build_global_metrics_dataframe(metrics_summary)
    category_df = build_category_metrics_dataframe(metrics_summary)
    video_df = build_video_metrics_dataframe(metrics_summary)
    global_conf_df = build_confusion_matrix_dataframe(metrics_summary["global"])
    category_conf_df = build_category_confusion_dataframe(metrics_summary)

    workbook_path = output_dir / "3_MAIA-like_evaluation_summary.xlsx"
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
        output_path=plots_dir / "accuracy_by_category.png",
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


def main() -> None:
    config = InferenceConfig()

    out_dir = Path(config.out_dir)
    predictions_output_path = out_dir / "qwen_mc_video_only_predictions_by_video.json"
    flat_predictions_output_path = out_dir / "qwen_mc_video_only_predictions_flat.json"
    metrics_output_path = out_dir / "qwen_mc_video_only_metrics.json"
    metrics_report_output_path = out_dir / "qwen_mc_video_only_metrics_report.txt"
    results_jsonl_path = out_dir / "results.jsonl"
    evaluation_output_dir = out_dir / "evaluation"

    dataset_json = load_json_file(config.dataset_path)
    examples = extract_examples(dataset_json)
    if not examples:
        raise ValueError("No examples were found in the dataset JSON.")

    video_paths = list_mp4_videos(config.videos_dir)
    video_path_index = build_video_path_index(video_paths)
    processed_videos = preprocess_videos(
        examples=examples,
        video_path_index=video_path_index,
        num_frames=config.num_frames,
        max_image_dimension=config.max_image_dimension,
    )

    model, processor, tokenizer = load_model(
        model_name=config.model_name,
        torch_dtype=resolve_torch_dtype(config.torch_dtype_name),
        attn_implementation=config.attn_implementation,
        min_pixels=config.processor_min_pixels,
        max_pixels=config.processor_max_pixels,
    )
    print(f"[INFO] Tokenizer: {type(tokenizer).__name__}")

    records = run_inference(
        examples=examples,
        model=model,
        processor=processor,
        processed_videos=processed_videos,
        batch_size=config.batch_size,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
    )

    metrics_summary = compute_all_metrics(records)

    save_json_file(
        build_results_json(
            records=records,
            metrics_summary=metrics_summary,
            num_frames=config.num_frames,
            max_image_dimension=config.max_image_dimension,
        ),
        predictions_output_path,
    )
    save_json_file(
        {
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "predictions": [asdict(record) for record in records],
        },
        flat_predictions_output_path,
    )
    save_json_file(metrics_summary, metrics_output_path)
    save_jsonl(build_jsonl_rows(records), results_jsonl_path)

    metrics_report = build_metrics_report(metrics_summary)
    save_text_file(metrics_report, metrics_report_output_path)
    save_evaluation_artifacts(metrics_summary, evaluation_output_dir)

    print(f"[INFO] Predictions by video saved to: {predictions_output_path}")
    print(f"[INFO] Flat predictions saved to: {flat_predictions_output_path}")
    print(f"[INFO] Metrics saved to: {metrics_output_path}")
    print(f"[INFO] JSONL results saved to: {results_jsonl_path}")
    print(f"[INFO] Metrics report saved to: {metrics_report_output_path}")
    print(f"[INFO] Evaluation artifacts saved to: {evaluation_output_dir}")


if __name__ == "__main__":
    main()
