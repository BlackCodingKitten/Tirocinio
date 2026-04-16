from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import (
    AutoProcessor,
    PreTrainedTokenizerBase,
    Qwen2_5_VLForConditionalGeneration,
)

# Select the GPU to use.
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

JsonDict = Dict[str, Any]
EMPTY_RAW_OUTPUT_TOKEN = "[[EMPTY_OUTPUT]]"
EPSILON = 1e-12
EXPERIMENT_NAME = "Experiment 1B"


@dataclass(frozen=True)
class Example:
    """
    Single multiple-choice example extracted from the dataset JSON.
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
    Flat prediction record used both for export and metric computation.
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
    choice_0_logprob: float
    choice_1_logprob: float
    choice_0_score: float
    choice_1_score: float
    predicted_label_by_score: int
    is_correct_by_score: bool
    score_gap: float
    confidence: float
    free_vs_score_agree: bool


def load_model(
    model_name: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "cuda:7",
    attn_implementation: str = "flash_attention_2",
) -> tuple[
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    PreTrainedTokenizerBase,
]:
    """
    Load the model, processor and tokenizer.
    """
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

    processor = AutoProcessor.from_pretrained(
        model_name,
        # use_fast=True,
        padding_side="left",
    )
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
    Normalize question_category by removing a trailing A/B suffix.
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

    print(f"[INFO] Extracted {len(examples)} examples.")
    return examples


def build_prompt(choice_0: str, choice_1: str) -> str:
    """
    Build the prompt used in Experiment 1B.
    """
    return (
        "Scegli la descrizione corretta rispetto al contenuto del video:\n"
        f"0:{choice_0}\n"
        f"1:{choice_1}\n"
        "Rispondi solo con 0 o 1."
    )


def build_messages(prompts: Sequence[str]) -> List[List[Dict[str, Any]]]:
    """
    Convert a list of prompts into chat-formatted messages.
    """
    messages: List[List[Dict[str, Any]]] = []

    for prompt in prompts:
        messages.append(
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a precise binary assistant. Reply only with 0 or 1.",
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


def render_chat_texts(
    processor: AutoProcessor,
    prompts: Sequence[str],
) -> List[str]:
    """
    Render prompts with the Qwen chat template.
    """
    messages = build_messages(prompts)
    return [
        processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        for message in messages
    ]


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
    rendered_texts = render_chat_texts(processor, prompts)

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
    Extract the first standalone binary answer from the model output.
    """
    match = re.search(r"\b([01])\b", raw_output)
    if match:
        return int(match.group(1))
    return None


def get_binary_token_ids(
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[Optional[int], Optional[int]]:
    """
    Try to obtain single-token ids for '0' and '1'.
    """
    ids_0 = tokenizer.encode("0", add_special_tokens=False)
    ids_1 = tokenizer.encode("1", add_special_tokens=False)

    if len(ids_0) == 1 and len(ids_1) == 1:
        return ids_0[0], ids_1[0]

    return None, None


def compute_sequence_logprob(
    model: Qwen2_5_VLForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    candidate_text: str,
) -> float:
    """
    Fallback scorer when the candidate is not encoded as a single token.
    """
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]

    full_ids = tokenizer(
        prompt_text + candidate_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]

    prefix_len = 0
    max_prefix = min(len(prompt_ids), len(full_ids))
    for index in range(max_prefix):
        if prompt_ids[index].item() == full_ids[index].item():
            prefix_len += 1
        else:
            break

    if prefix_len >= len(full_ids):
        raise ValueError("Candidate continuation produced no extra tokens.")

    input_ids = full_ids.unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    inputs = move_inputs_to_model_device(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        model,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    total_logprob = 0.0

    moved_full_ids = inputs["input_ids"][0]
    for position in range(prefix_len, moved_full_ids.shape[0]):
        total_logprob += float(log_probs[position - 1, moved_full_ids[position]].item())

    return total_logprob


def build_score_summary(logprob_0: float, logprob_1: float) -> Dict[str, float]:
    """
    Convert raw log probabilities into normalized scores over the two choices.
    """
    pair = torch.tensor([logprob_0, logprob_1], dtype=torch.float32)
    probs = torch.softmax(pair, dim=0)

    choice_0_score = float(probs[0].item())
    choice_1_score = float(probs[1].item())
    predicted_label_by_score = 0 if choice_0_score >= choice_1_score else 1

    return {
        "choice_0_logprob": float(logprob_0),
        "choice_1_logprob": float(logprob_1),
        "choice_0_score": choice_0_score,
        "choice_1_score": choice_1_score,
        "predicted_label_by_score": int(predicted_label_by_score),
        "score_gap": abs(choice_0_score - choice_1_score),
        "confidence": max(choice_0_score, choice_1_score),
    }


def compute_binary_scores_batch_single_token(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    tokenizer: PreTrainedTokenizerBase,
    rendered_texts: Sequence[str],
) -> List[Dict[str, float]]:
    """
    Compute binary scores for answers '0' and '1' from the next-token logits.
    """
    token_id_0, token_id_1 = get_binary_token_ids(tokenizer)
    if token_id_0 is None or token_id_1 is None:
        raise ValueError(
            "Tokenizer does not encode '0' and '1' as single tokens. "
            "Use the fallback sequence scorer instead."
        )

    inputs = processor(
        text=list(rendered_texts),
        padding=True,
        return_tensors="pt",
    )
    inputs = move_inputs_to_model_device(inputs, model)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    attention_mask = inputs["attention_mask"]
    last_token_positions = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(logits.shape[0], device=logits.device)

    next_token_logits = logits[batch_indices, last_token_positions, :]
    next_token_logprobs = torch.log_softmax(next_token_logits, dim=-1)

    results: List[Dict[str, float]] = []
    for row_index in range(next_token_logprobs.shape[0]):
        logprob_0 = float(next_token_logprobs[row_index, token_id_0].item())
        logprob_1 = float(next_token_logprobs[row_index, token_id_1].item())
        results.append(build_score_summary(logprob_0, logprob_1))

    return results


def compute_binary_scores_batch(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
) -> List[Dict[str, float]]:
    """
    Compute log probabilities and normalized scores for choice 0 and choice 1.
    """
    rendered_texts = render_chat_texts(processor, prompts)
    token_id_0, token_id_1 = get_binary_token_ids(tokenizer)

    if token_id_0 is not None and token_id_1 is not None:
        return compute_binary_scores_batch_single_token(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            rendered_texts=rendered_texts,
        )

    results: List[Dict[str, float]] = []
    for rendered_text in rendered_texts:
        logprob_0 = compute_sequence_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt_text=rendered_text,
            candidate_text="0",
        )
        logprob_1 = compute_sequence_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt_text=rendered_text,
            candidate_text="1",
        )
        results.append(build_score_summary(logprob_0, logprob_1))

    return results


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Safe floating-point division.
    """
    return numerator / denominator if denominator != 0 else 0.0


def mean_or_zero(values: Sequence[float]) -> float:
    """
    Compute the arithmetic mean or return 0.0 for an empty sequence.
    """
    return float(sum(values) / len(values)) if values else 0.0


def compute_confusion_counts(
    records: Sequence[PredictionRecord],
    use_score_predictions: bool,
) -> Dict[str, int]:
    """
    Compute a binary confusion matrix.
    """
    def predicted_label(record: PredictionRecord) -> Optional[int]:
        return record.predicted_label_by_score if use_score_predictions else record.predicted_label

    actual_0_pred_0 = sum(
        record.target == 0 and predicted_label(record) == 0
        for record in records
    )
    actual_0_pred_1 = sum(
        record.target == 0 and predicted_label(record) == 1
        for record in records
    )
    actual_1_pred_0 = sum(
        record.target == 1 and predicted_label(record) == 0
        for record in records
    )
    actual_1_pred_1 = sum(
        record.target == 1 and predicted_label(record) == 1
        for record in records
    )

    return {
        "actual_0_pred_0": actual_0_pred_0,
        "actual_0_pred_1": actual_0_pred_1,
        "actual_1_pred_0": actual_1_pred_0,
        "actual_1_pred_1": actual_1_pred_1,
    }


def compute_label_metrics(
    records: Sequence[PredictionRecord],
    label: int,
    use_score_predictions: bool,
) -> Dict[str, float]:
    """
    Compute precision, recall and F1 for one label.
    """
    def predicted_label(record: PredictionRecord) -> Optional[int]:
        return record.predicted_label_by_score if use_score_predictions else record.predicted_label

    tp = sum(
        predicted_label(record) == label and record.target == label
        for record in records
    )
    fp = sum(
        predicted_label(record) == label and record.target != label
        for record in records
    )
    fn = sum(
        record.target == label and predicted_label(record) != label
        for record in records
    )

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "support": float(sum(record.target == label for record in records)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_metrics(records: Sequence[PredictionRecord]) -> JsonDict:
    """
    Compute a compact set of metrics with clearer names.
    """
    print(f"[INFO] Computing metrics for {len(records)} records...")

    n_samples = len(records)
    invalid_free_predictions = sum(record.predicted_label not in (0, 1) for record in records)
    valid_free_predictions = n_samples - invalid_free_predictions
    empty_raw_outputs = sum(record.raw_model_output == EMPTY_RAW_OUTPUT_TOKEN for record in records)

    free_correct = sum(record.is_correct for record in records)
    score_correct = sum(record.is_correct_by_score for record in records)

    free_label_0 = compute_label_metrics(records, label=0, use_score_predictions=False)
    free_label_1 = compute_label_metrics(records, label=1, use_score_predictions=False)
    score_label_0 = compute_label_metrics(records, label=0, use_score_predictions=True)
    score_label_1 = compute_label_metrics(records, label=1, use_score_predictions=True)

    free_confusion = compute_confusion_counts(records, use_score_predictions=False)
    score_confusion = compute_confusion_counts(records, use_score_predictions=True)

    return {
        "n_samples": n_samples,
        "valid_free_predictions": valid_free_predictions,
        "invalid_free_predictions": invalid_free_predictions,
        "empty_raw_outputs": empty_raw_outputs,
        "free_output_accuracy": safe_divide(free_correct, n_samples),
        "free_output_valid_accuracy": safe_divide(
            free_confusion["actual_0_pred_0"] + free_confusion["actual_1_pred_1"],
            valid_free_predictions,
        ),
        "score_based_accuracy": safe_divide(score_correct, n_samples),
        "free_label_0_precision": free_label_0["precision"],
        "free_label_0_recall": free_label_0["recall"],
        "free_label_0_f1": free_label_0["f1"],
        "free_label_1_precision": free_label_1["precision"],
        "free_label_1_recall": free_label_1["recall"],
        "free_label_1_f1": free_label_1["f1"],
        "score_label_0_precision": score_label_0["precision"],
        "score_label_0_recall": score_label_0["recall"],
        "score_label_0_f1": score_label_0["f1"],
        "score_label_1_precision": score_label_1["precision"],
        "score_label_1_recall": score_label_1["recall"],
        "score_label_1_f1": score_label_1["f1"],
        "avg_choice_0_score": mean_or_zero([record.choice_0_score for record in records]),
        "avg_choice_1_score": mean_or_zero([record.choice_1_score for record in records]),
        "avg_confidence": mean_or_zero([record.confidence for record in records]),
        "avg_score_gap": mean_or_zero([record.score_gap for record in records]),
        "free_vs_score_agreement_rate": safe_divide(
            sum(record.free_vs_score_agree for record in records),
            n_samples,
        ),
        "free_actual_0_pred_0": free_confusion["actual_0_pred_0"],
        "free_actual_0_pred_1": free_confusion["actual_0_pred_1"],
        "free_actual_1_pred_0": free_confusion["actual_1_pred_0"],
        "free_actual_1_pred_1": free_confusion["actual_1_pred_1"],
        "score_actual_0_pred_0": score_confusion["actual_0_pred_0"],
        "score_actual_0_pred_1": score_confusion["actual_0_pred_1"],
        "score_actual_1_pred_0": score_confusion["actual_1_pred_0"],
        "score_actual_1_pred_1": score_confusion["actual_1_pred_1"],
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

    return {
        "global": compute_metrics(records),
        "by_normalized_question_category": {
            category_name: compute_metrics(category_records)
            for category_name, category_records in sorted(by_category.items())
        },
        "by_video": {
            video_id: compute_metrics(video_records)
            for video_id, video_records in sorted(
                by_video.items(),
                key=lambda item: natural_video_sort_key(item[0]),
            )
        },
    }


def metric_row_from_summary(
    entity_name: str,
    metrics: JsonDict,
    entity_column_name: str,
) -> Dict[str, Any]:
    """
    Convert a metric summary dictionary into a single flat row.
    """
    return {
        entity_column_name: entity_name,
        "n_samples": metrics["n_samples"],
        "valid_free_predictions": metrics["valid_free_predictions"],
        "invalid_free_predictions": metrics["invalid_free_predictions"],
        "empty_raw_outputs": metrics["empty_raw_outputs"],
        "free_output_accuracy": metrics["free_output_accuracy"],
        "free_output_valid_accuracy": metrics["free_output_valid_accuracy"],
        "score_based_accuracy": metrics["score_based_accuracy"],
        "free_label_0_precision": metrics["free_label_0_precision"],
        "free_label_0_recall": metrics["free_label_0_recall"],
        "free_label_0_f1": metrics["free_label_0_f1"],
        "free_label_1_precision": metrics["free_label_1_precision"],
        "free_label_1_recall": metrics["free_label_1_recall"],
        "free_label_1_f1": metrics["free_label_1_f1"],
        "score_label_0_precision": metrics["score_label_0_precision"],
        "score_label_0_recall": metrics["score_label_0_recall"],
        "score_label_0_f1": metrics["score_label_0_f1"],
        "score_label_1_precision": metrics["score_label_1_precision"],
        "score_label_1_recall": metrics["score_label_1_recall"],
        "score_label_1_f1": metrics["score_label_1_f1"],
        "avg_choice_0_score": metrics["avg_choice_0_score"],
        "avg_choice_1_score": metrics["avg_choice_1_score"],
        "avg_confidence": metrics["avg_confidence"],
        "avg_score_gap": metrics["avg_score_gap"],
        "free_vs_score_agreement_rate": metrics["free_vs_score_agreement_rate"],
        "free_actual_0_pred_0": metrics["free_actual_0_pred_0"],
        "free_actual_0_pred_1": metrics["free_actual_0_pred_1"],
        "free_actual_1_pred_0": metrics["free_actual_1_pred_0"],
        "free_actual_1_pred_1": metrics["free_actual_1_pred_1"],
        "score_actual_0_pred_0": metrics["score_actual_0_pred_0"],
        "score_actual_0_pred_1": metrics["score_actual_0_pred_1"],
        "score_actual_1_pred_0": metrics["score_actual_1_pred_0"],
        "score_actual_1_pred_1": metrics["score_actual_1_pred_1"],
    }


def build_global_metrics_dataframe(metrics_summary: JsonDict) -> pd.DataFrame:
    """
    Build the global metrics DataFrame.
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
    Build the per-category metrics DataFrame.
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
    Build the per-video metrics DataFrame.
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


def build_confusion_matrix_dataframe(
    metrics: JsonDict,
    prefix: str,
) -> pd.DataFrame:
    """
    Build a 2x2 confusion matrix DataFrame.
    """
    return pd.DataFrame(
        [
            [metrics[f"{prefix}_actual_0_pred_0"], metrics[f"{prefix}_actual_0_pred_1"]],
            [metrics[f"{prefix}_actual_1_pred_0"], metrics[f"{prefix}_actual_1_pred_1"]],
        ],
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )


def dataframe_to_string(df: pd.DataFrame, index: bool = False) -> str:
    """
    Convert a DataFrame into a readable string for text reports.
    """
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 240,
        "display.max_colwidth", 200,
    ):
        return df.to_string(index=index, float_format=lambda value: f"{value:.4f}")


def build_metrics_report(metrics_summary: JsonDict) -> str:
    """
    Build a compact text report for Experiment 1B.
    """
    global_df = build_global_metrics_dataframe(metrics_summary)
    category_df = build_category_metrics_dataframe(metrics_summary)
    video_df = build_video_metrics_dataframe(metrics_summary)
    global_free_conf_df = build_confusion_matrix_dataframe(metrics_summary["global"], prefix="free")
    global_score_conf_df = build_confusion_matrix_dataframe(metrics_summary["global"], prefix="score")

    report_parts: List[str] = [
        f"{EXPERIMENT_NAME} EVALUATION REPORT",
        "=" * (len(EXPERIMENT_NAME) + 18),
        "",
        "DEFINITIONS",
        "-----------",
        "- free_output_accuracy: accuracy obtained from the decoded model answer.",
        "- free_output_valid_accuracy: free-form accuracy computed only on parsed 0/1 outputs.",
        "- score_based_accuracy: accuracy obtained by comparing the normalized scores of choice 0 and choice 1.",
        "- choice_0_score and choice_1_score are the normalized probabilities over the binary pair {0,1}.",
        "- avg_score_gap is the average absolute difference between choice_0_score and choice_1_score.",
        "- free_vs_score_agreement_rate measures how often decoded output and score-based decision agree.",
        "",
        "GLOBAL METRICS",
        "--------------",
        dataframe_to_string(global_df, index=False),
        "",
        f"GLOBAL CONFUSION MATRIX - {EXPERIMENT_NAME} FREE OUTPUT",
        "-" * (35 + len(EXPERIMENT_NAME)),
        dataframe_to_string(global_free_conf_df, index=True),
        "",
        f"GLOBAL CONFUSION MATRIX - {EXPERIMENT_NAME} SCORE-BASED CHOICE",
        "-" * (42 + len(EXPERIMENT_NAME)),
        dataframe_to_string(global_score_conf_df, index=True),
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


def plot_confusion_matrix(
    metrics_summary: JsonDict,
    prefix: str,
    title: str,
    output_path: str | Path,
) -> None:
    """
    Plot one confusion matrix.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conf_df = build_confusion_matrix_dataframe(metrics_summary["global"], prefix=prefix)
    values = conf_df.values

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(values)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(conf_df.columns)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(conf_df.index)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")

    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            ax.text(col_index, row_index, str(values[row_index, col_index]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Plot saved to: {output_path}")


def plot_metric_by_category(
    category_df: pd.DataFrame,
    metric_column: str,
    title: str,
    ylabel: str,
    output_path: str | Path,
) -> None:
    """
    Plot a category-level metric.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if category_df.empty:
        print(f"[INFO] Skipping plot {title}: empty DataFrame.")
        return

    plot_df = category_df.sort_values(metric_column, ascending=False)

    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 0.8), 5))
    ax.bar(plot_df[plot_df.columns[0]], plot_df[metric_column])

    ax.set_title(title)
    ax.set_xlabel(plot_df.columns[0])
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)

    for index, value in enumerate(plot_df[metric_column].tolist()):
        ax.text(index, value, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Plot saved to: {output_path}")


def build_results_json(
    records: Sequence[PredictionRecord],
    metrics_summary: JsonDict,
) -> JsonDict:
    """
    Build the final nested JSON output grouped by video and category.
    """
    print("[INFO] Building final results JSON structure...")
    results: JsonDict = {}

    for record in records:
        if record.video_id not in results:
            results[record.video_id] = {
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
            "predicted_label_by_score": record.predicted_label_by_score,
            "is_correct": record.is_correct,
            "is_correct_by_score": record.is_correct_by_score,
            "choice_0_logprob": record.choice_0_logprob,
            "choice_1_logprob": record.choice_1_logprob,
            "choice_0_score": record.choice_0_score,
            "choice_1_score": record.choice_1_score,
            "score_gap": record.score_gap,
            "confidence": record.confidence,
            "free_vs_score_agree": record.free_vs_score_agree,
        }

    return results


def build_prediction_dataframe(records: Sequence[PredictionRecord]) -> pd.DataFrame:
    """
    Build the flat prediction DataFrame.
    """
    return pd.DataFrame([asdict(record) for record in records])


def build_prediction_json(records: Sequence[PredictionRecord]) -> JsonDict:
    """
    Build the flat prediction JSON.
    """
    return {"records": [asdict(record) for record in records]}


def save_evaluation_artifacts(
    metrics_summary: JsonDict,
    output_dir: str | Path,
) -> None:
    """
    Save the main evaluation tables and plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    global_df = build_global_metrics_dataframe(metrics_summary)
    category_df = build_category_metrics_dataframe(metrics_summary)
    video_df = build_video_metrics_dataframe(metrics_summary)
    global_free_conf_df = build_confusion_matrix_dataframe(metrics_summary["global"], prefix="free")
    global_score_conf_df = build_confusion_matrix_dataframe(metrics_summary["global"], prefix="score")

    workbook_path = output_dir / "1B_evaluation_summary.xlsx"

    with pd.ExcelWriter(workbook_path) as writer:
        global_df.to_excel(writer, sheet_name="global_metrics", index=False)
        category_df.to_excel(writer, sheet_name="category_metrics", index=False)
        video_df.to_excel(writer, sheet_name="video_metrics", index=False)
        global_free_conf_df.to_excel(writer, sheet_name="free_confusion")
        global_score_conf_df.to_excel(writer, sheet_name="score_confusion")

    save_dataframe_csv(global_df, output_dir / "global_metrics.csv")
    save_dataframe_csv(category_df, output_dir / "normalized_question_category_metrics.csv")
    save_dataframe_csv(video_df, output_dir / "video_metrics.csv")

    plot_confusion_matrix(
        metrics_summary=metrics_summary,
        prefix="free",
        title=f"{EXPERIMENT_NAME} - Global confusion matrix (free output)",
        output_path=plots_dir / "1B_global_confusion_free_output.png",
    )
    plot_confusion_matrix(
        metrics_summary=metrics_summary,
        prefix="score",
        title=f"{EXPERIMENT_NAME} - Global confusion matrix (score-based choice)",
        output_path=plots_dir / "1B_global_confusion_score_based.png",
    )
    plot_metric_by_category(
        category_df=category_df,
        metric_column="free_output_accuracy",
        title=f"{EXPERIMENT_NAME} - Free output accuracy by question category",
        ylabel="Free output accuracy",
        output_path=plots_dir / "1B_free_output_accuracy_by_category.png",
    )
    plot_metric_by_category(
        category_df=category_df,
        metric_column="score_based_accuracy",
        title=f"{EXPERIMENT_NAME} - Score-based accuracy by question category",
        ylabel="Score-based accuracy",
        output_path=plots_dir / "1B_score_based_accuracy_by_category.png",
    )
    plot_metric_by_category(
        category_df=category_df,
        metric_column="avg_confidence",
        title=f"{EXPERIMENT_NAME} - Average confidence by question category",
        ylabel="Average confidence",
        output_path=plots_dir / "1B_average_confidence_by_category.png",
    )
    plot_metric_by_category(
        category_df=category_df,
        metric_column="avg_score_gap",
        title=f"{EXPERIMENT_NAME} - Average score gap by question category",
        ylabel="Average score gap",
        output_path=plots_dir / "1B_average_score_gap_by_category.png",
    )

    print(f"[INFO] Evaluation workbook saved to: {workbook_path}")
    print(f"[INFO] Evaluation artifacts saved to: {output_dir}")


def save_score_artifacts(
    records: Sequence[PredictionRecord],
    metrics_summary: JsonDict,
    output_dir: str | Path,
) -> None:
    """
    Save flat prediction artifacts and summary files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_df = build_prediction_dataframe(records)
    prediction_json = build_prediction_json(records)
    metrics_report = build_metrics_report(metrics_summary)

    save_json_file(prediction_json, output_dir / "qwen_mc_1B_score_results.json")
    save_dataframe_csv(prediction_df, output_dir / "qwen_mc_1B_score_results.csv")
    save_text_file(metrics_report, output_dir / "qwen_mc_1B_metrics_report.txt")
    save_json_file({"global": metrics_summary["global"]}, output_dir / "qwen_mc_1B_summary.json")

    with pd.ExcelWriter(output_dir / "qwen_mc_1B_score_results.xlsx") as writer:
        prediction_df.to_excel(writer, sheet_name="predictions", index=False)
        build_global_metrics_dataframe(metrics_summary).to_excel(
            writer,
            sheet_name="global_metrics",
            index=False,
        )
        build_category_metrics_dataframe(metrics_summary).to_excel(
            writer,
            sheet_name="category_metrics",
            index=False,
        )
        build_video_metrics_dataframe(metrics_summary).to_excel(
            writer,
            sheet_name="video_metrics",
            index=False,
        )

    print(f"[INFO] Score artifacts saved to: {output_dir}")


def run_inference(
    examples: Sequence[Example],
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 16,
) -> List[PredictionRecord]:
    """
    Run batched inference on all examples.
    """
    print("[INFO] Starting batched inference...")
    records: List[PredictionRecord] = []
    total_examples = len(examples)
    total_batches = (total_examples + batch_size - 1) // batch_size

    print(f"[INFO] Total examples: {total_examples}")
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
        score_results = compute_binary_scores_batch(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            prompts=prompts,
        )

        for example, prompt, raw_output, score_result in zip(
            batch_examples,
            prompts,
            raw_outputs,
            score_results,
        ):
            safe_raw_output = normalize_raw_output(raw_output)
            predicted_label = parse_binary_answer(safe_raw_output)
            predicted_label_by_score = int(score_result["predicted_label_by_score"])

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
                    is_correct=(predicted_label == example.target),
                    choice_0_logprob=float(score_result["choice_0_logprob"]),
                    choice_1_logprob=float(score_result["choice_1_logprob"]),
                    choice_0_score=float(score_result["choice_0_score"]),
                    choice_1_score=float(score_result["choice_1_score"]),
                    predicted_label_by_score=predicted_label_by_score,
                    is_correct_by_score=(predicted_label_by_score == example.target),
                    score_gap=float(score_result["score_gap"]),
                    confidence=float(score_result["confidence"]),
                    free_vs_score_agree=(
                        predicted_label is not None and predicted_label == predicted_label_by_score
                    ),
                )
            )

        print(f"[INFO] Processed {min(start_index + len(batch_examples), total_examples)}/{total_examples} examples.")

    print("[INFO] Inference completed.")
    return records


def main() -> None:
    """
    Main entry point.
    """
    print("[INFO] Script started.")
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    dataset_path = Path("Data/Dataset/maia_ita_mc_by_video_category_pool.json")

    predictions_output_path = Path("Data/ModelResponse/1B/qwen_mc_1B_predictions_by_video.json")
    metrics_report_output_path = Path("Data/ModelResponse/1B/qwen_mc_1B_metrics_report.txt")
    evaluation_output_dir = Path("Data/ModelResponse/1B/evaluation")
    score_output_dir = Path("Data/ModelResponse/1B")

    batch_size = 4

    print(f"[INFO] dataset_path: {dataset_path}")
    print(f"[INFO] predictions_output_path: {predictions_output_path}")
    print(f"[INFO] metrics_report_output_path: {metrics_report_output_path}")
    print(f"[INFO] evaluation_output_dir: {evaluation_output_dir}")
    print(f"[INFO] score_output_dir: {score_output_dir}")
    print(f"[INFO] batch_size: {batch_size}")

    dataset_json = load_json_file(dataset_path)
    examples = extract_examples(dataset_json)
    if not examples:
        raise ValueError("No examples were found in the dataset JSON.")

    model, processor, tokenizer = load_model(model_name)

    records = run_inference(
        examples=examples,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    metrics_summary = compute_all_metrics(records)
    predictions_json = build_results_json(
        records=records,
        metrics_summary=metrics_summary,
    )

    save_json_file(predictions_json, predictions_output_path)

    metrics_report = build_metrics_report(metrics_summary)
    save_text_file(metrics_report, metrics_report_output_path)

    save_evaluation_artifacts(
        metrics_summary=metrics_summary,
        output_dir=evaluation_output_dir,
    )
    save_score_artifacts(
        records=records,
        metrics_summary=metrics_summary,
        output_dir=score_output_dir,
    )

    print(f"[INFO] Predictions saved to: {predictions_output_path}")
    print(f"[INFO] Metrics report saved to: {metrics_report_output_path}")
    print(f"[INFO] Evaluation artifacts saved to: {evaluation_output_dir}")
    print(f"[INFO] Score artifacts saved to: {score_output_dir}")
    print("[INFO] Script finished successfully.")


if __name__ == "__main__":
    main()
