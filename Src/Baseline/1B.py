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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

JsonDict = Dict[str, Any]
EMPTY_RAW_OUTPUT_TOKEN = "[[EMPTY_OUTPUT]]"
EPSILON = 1e-12


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
    logprob_0: float
    logprob_1: float
    prob_0: float
    prob_1: float
    predicted_label_by_logprob: int
    logprob_margin: float
    confidence: float
    entropy: float
    free_vs_logprob_agree: bool
    is_ambiguous_by_logprob: bool
    is_high_confidence_wrong_by_logprob: bool
    is_low_confidence_correct_by_logprob: bool
    is_correct_by_logprob: bool


@dataclass(frozen=True)
class AggregateThresholds:
    """
    Thresholds used to summarize log-probability behavior.
    """
    ambiguous_margin: float = 0.20
    high_confidence: float = 0.80
    low_confidence: float = 0.55


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
        use_fast=True,
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
        f"Scegli la descrizione corretta rispetto al contenuto del video, considera anche la trascrizione del suo audio:\n"
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
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a precise binary assistant. "
                                "Answer only with 0 or 1. "
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                },
            ]
        )

    print("[INFO] Chat messages built successfully.")
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
    rendered_texts = render_chat_texts(processor, prompts)

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
    """
    cleaned = raw_output.strip()

    if "0" in cleaned:
        return 0
    if "1" in cleaned:
        return 1
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
    Fallback scorer for a candidate continuation when the candidate is not a
    single token. It computes the total conditional log probability of the
    continuation after the prompt.
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
    for i in range(max_prefix):
        if prompt_ids[i].item() == full_ids[i].item():
            prefix_len += 1
        else:
            break

    if prefix_len >= len(full_ids):
        raise ValueError("Candidate continuation produced no extra tokens.")

    input_ids = full_ids.unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    inputs = move_inputs_to_model_device(inputs, model)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)

    total_logprob = 0.0
    moved_full_ids = inputs["input_ids"][0]

    for pos in range(prefix_len, moved_full_ids.shape[0]):
        total_logprob += float(log_probs[pos - 1, moved_full_ids[pos]].item())

    return total_logprob


def compute_binary_logprobs_batch_single_token(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    tokenizer: PreTrainedTokenizerBase,
    rendered_texts: Sequence[str],
) -> List[Dict[str, float]]:
    """
    Compute log probabilities for answers '0' and '1' using the logits for the
    next token after the prompt, assuming both are single tokenizer tokens.
    """
    token_id_0, token_id_1 = get_binary_token_ids(tokenizer)
    if token_id_0 is None or token_id_1 is None:
        raise ValueError(
            "Tokenizer does not encode '0' and '1' as single tokens. "
            "Use the fallback sequence-scoring function instead."
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

    for row in range(next_token_logprobs.shape[0]):
        logprob_0 = float(next_token_logprobs[row, token_id_0].item())
        logprob_1 = float(next_token_logprobs[row, token_id_1].item())

        pair = torch.tensor([logprob_0, logprob_1], dtype=torch.float32)
        probs = torch.softmax(pair, dim=0)
        prob_0 = float(probs[0].item())
        prob_1 = float(probs[1].item())

        confidence = max(prob_0, prob_1)
        entropy = float(
            -(
                prob_0 * math.log(max(prob_0, EPSILON))
                + prob_1 * math.log(max(prob_1, EPSILON))
            )
        )
        predicted_label_by_logprob = 0 if logprob_0 >= logprob_1 else 1

        results.append(
            {
                "logprob_0": logprob_0,
                "logprob_1": logprob_1,
                "prob_0": prob_0,
                "prob_1": prob_1,
                "predicted_label_by_logprob": predicted_label_by_logprob,
                "logprob_margin": abs(logprob_0 - logprob_1),
                "confidence": confidence,
                "entropy": entropy,
            }
        )

    return results


def compute_binary_logprobs_batch(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
) -> List[Dict[str, float]]:
    """
    Compute logprob('0'|prompt) and logprob('1'|prompt) for a batch of prompts.
    """
    rendered_texts = render_chat_texts(processor, prompts)

    token_id_0, token_id_1 = get_binary_token_ids(tokenizer)
    if token_id_0 is not None and token_id_1 is not None:
        return compute_binary_logprobs_batch_single_token(
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

        pair = torch.tensor([logprob_0, logprob_1], dtype=torch.float32)
        probs = torch.softmax(pair, dim=0)
        prob_0 = float(probs[0].item())
        prob_1 = float(probs[1].item())
        confidence = max(prob_0, prob_1)
        entropy = float(
            -(
                prob_0 * math.log(max(prob_0, EPSILON))
                + prob_1 * math.log(max(prob_1, EPSILON))
            )
        )
        predicted_label_by_logprob = 0 if logprob_0 >= logprob_1 else 1

        results.append(
            {
                "logprob_0": float(logprob_0),
                "logprob_1": float(logprob_1),
                "prob_0": prob_0,
                "prob_1": prob_1,
                "predicted_label_by_logprob": predicted_label_by_logprob,
                "logprob_margin": abs(float(logprob_0) - float(logprob_1)),
                "confidence": confidence,
                "entropy": entropy,
            }
        )

    return results


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Safe floating-point division.
    """
    return numerator / denominator if denominator != 0 else 0.0


def mean_or_zero(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def compute_confusion_counts(records: Sequence[PredictionRecord]) -> Dict[str, int]:
    """
    Compute a binary 2x2 confusion matrix over valid free-form predictions only.
    """
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


def compute_logprob_confusion_counts(records: Sequence[PredictionRecord]) -> Dict[str, int]:
    """
    Compute a binary 2x2 confusion matrix using the label preferred by logprob.
    """
    c00 = sum(record.target == 0 and record.predicted_label_by_logprob == 0 for record in records)
    c01 = sum(record.target == 0 and record.predicted_label_by_logprob == 1 for record in records)
    c10 = sum(record.target == 1 and record.predicted_label_by_logprob == 0 for record in records)
    c11 = sum(record.target == 1 and record.predicted_label_by_logprob == 1 for record in records)

    return {
        "logprob_actual_0_pred_0": c00,
        "logprob_actual_0_pred_1": c01,
        "logprob_actual_1_pred_0": c10,
        "logprob_actual_1_pred_1": c11,
    }


def compute_label_metrics_free(
    records: Sequence[PredictionRecord],
    label: int,
) -> Dict[str, float]:
    """
    Class-specific metrics for free-form decoded predictions.
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


def compute_label_metrics_logprob(
    records: Sequence[PredictionRecord],
    label: int,
) -> Dict[str, float]:
    """
    Class-specific metrics for logprob-preferred predictions.
    No invalid predictions exist here because the decision is always binary.
    """
    tp = sum(
        record.predicted_label_by_logprob == label and record.target == label
        for record in records
    )
    fp = sum(
        record.predicted_label_by_logprob == label and record.target != label
        for record in records
    )
    fn = sum(
        record.target == label and record.predicted_label_by_logprob != label
        for record in records
    )

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    support = sum(record.target == label for record in records)
    predicted_as_label = sum(record.predicted_label_by_logprob == label for record in records)

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

    If records are global, the metrics are global.
    If records belong to one video, the metrics are per-video.
    If records belong to one category, the metrics are per-category.

    No macro-average or micro-average is used.
    """
    print(f"[INFO] Computing metrics for {len(records)} records...")

    n_samples = len(records)
    empty_raw_outputs = sum(record.raw_model_output == EMPTY_RAW_OUTPUT_TOKEN for record in records)
    invalid_predictions = sum(record.predicted_label not in (0, 1) for record in records)
    valid_predictions = n_samples - invalid_predictions
    correct_predictions = sum(record.is_correct for record in records)
    correct_predictions_by_logprob = sum(record.is_correct_by_logprob for record in records)

    free_label_0 = compute_label_metrics_free(records, label=0)
    free_label_1 = compute_label_metrics_free(records, label=1)
    log_label_0 = compute_label_metrics_logprob(records, label=0)
    log_label_1 = compute_label_metrics_logprob(records, label=1)

    confusion_counts = compute_confusion_counts(records)
    logprob_confusion_counts = compute_logprob_confusion_counts(records)

    avg_logprob_margin = mean_or_zero([record.logprob_margin for record in records])
    avg_confidence = mean_or_zero([record.confidence for record in records])
    avg_entropy = mean_or_zero([record.entropy for record in records])

    correct_confidences = [record.confidence for record in records if record.is_correct_by_logprob]
    wrong_confidences = [record.confidence for record in records if not record.is_correct_by_logprob]
    correct_margins = [record.logprob_margin for record in records if record.is_correct_by_logprob]
    wrong_margins = [record.logprob_margin for record in records if not record.is_correct_by_logprob]

    agreement_count = sum(record.free_vs_logprob_agree for record in records)
    ambiguous_count = sum(record.is_ambiguous_by_logprob for record in records)
    high_conf_wrong_count = sum(record.is_high_confidence_wrong_by_logprob for record in records)
    low_conf_correct_count = sum(record.is_low_confidence_correct_by_logprob for record in records)

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

        "Support_0": int(free_label_0["support"]),
        "Predicted_as_0": int(free_label_0["predicted_as_label"]),
        "Precision_0": free_label_0["precision"],
        "Recall_0": free_label_0["recall"],
        "F1_0": free_label_0["f1"],

        "Support_1": int(free_label_1["support"]),
        "Predicted_as_1": int(free_label_1["predicted_as_label"]),
        "Precision_1": free_label_1["precision"],
        "Recall_1": free_label_1["recall"],
        "F1_1": free_label_1["f1"],

        "LogProb_Accuracy": safe_divide(correct_predictions_by_logprob, n_samples),

        "LogProb_Support_0": int(log_label_0["support"]),
        "LogProb_Predicted_as_0": int(log_label_0["predicted_as_label"]),
        "LogProb_Precision_0": log_label_0["precision"],
        "LogProb_Recall_0": log_label_0["recall"],
        "LogProb_F1_0": log_label_0["f1"],

        "LogProb_Support_1": int(log_label_1["support"]),
        "LogProb_Predicted_as_1": int(log_label_1["predicted_as_label"]),
        "LogProb_Precision_1": log_label_1["precision"],
        "LogProb_Recall_1": log_label_1["recall"],
        "LogProb_F1_1": log_label_1["f1"],

        "avg_logprob_margin": avg_logprob_margin,
        "avg_confidence": avg_confidence,
        "avg_entropy": avg_entropy,
        "avg_confidence_correct_logprob": mean_or_zero(correct_confidences),
        "avg_confidence_wrong_logprob": mean_or_zero(wrong_confidences),
        "avg_margin_correct_logprob": mean_or_zero(correct_margins),
        "avg_margin_wrong_logprob": mean_or_zero(wrong_margins),
        "free_vs_logprob_agreement_rate": safe_divide(agreement_count, n_samples),
        "ambiguous_rate": safe_divide(ambiguous_count, n_samples),
        "high_confidence_wrong_rate": safe_divide(high_conf_wrong_count, n_samples),
        "low_confidence_correct_rate": safe_divide(low_conf_correct_count, n_samples),
        **confusion_counts,
        **logprob_confusion_counts,
    }


def assign_confidence_bucket(confidence: float) -> str:
    if confidence < 0.55:
        return "[0.50,0.55)"
    if confidence < 0.65:
        return "[0.55,0.65)"
    if confidence < 0.75:
        return "[0.65,0.75)"
    if confidence < 0.85:
        return "[0.75,0.85)"
    if confidence < 0.95:
        return "[0.85,0.95)"
    return "[0.95,1.00]"


def build_confidence_bucket_dataframe(records: Sequence[PredictionRecord]) -> pd.DataFrame:
    rows: Dict[str, Dict[str, Any]] = {}

    for record in records:
        bucket = assign_confidence_bucket(record.confidence)
        if bucket not in rows:
            rows[bucket] = {
                "confidence_bucket": bucket,
                "n_samples": 0,
                "avg_confidence": 0.0,
                "avg_logprob_margin": 0.0,
                "logprob_accuracy": 0.0,
            }

    grouped: Dict[str, List[PredictionRecord]] = {}
    for record in records:
        grouped.setdefault(assign_confidence_bucket(record.confidence), []).append(record)

    ordered_buckets = [
        "[0.50,0.55)",
        "[0.55,0.65)",
        "[0.65,0.75)",
        "[0.75,0.85)",
        "[0.85,0.95)",
        "[0.95,1.00]",
    ]

    result_rows: List[Dict[str, Any]] = []
    for bucket in ordered_buckets:
        bucket_records = grouped.get(bucket, [])
        if not bucket_records:
            continue
        result_rows.append(
            {
                "confidence_bucket": bucket,
                "n_samples": len(bucket_records),
                "avg_confidence": mean_or_zero([r.confidence for r in bucket_records]),
                "avg_logprob_margin": mean_or_zero([r.logprob_margin for r in bucket_records]),
                "logprob_accuracy": safe_divide(
                    sum(r.is_correct_by_logprob for r in bucket_records),
                    len(bucket_records),
                ),
            }
        )

    return pd.DataFrame(result_rows)


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
    Convert an aggregate metric summary dictionary into a single flat row.
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

        "LogProb_Accuracy": metrics["LogProb_Accuracy"],

        "LogProb_Support_0": metrics["LogProb_Support_0"],
        "LogProb_Predicted_as_0": metrics["LogProb_Predicted_as_0"],
        "LogProb_Precision_0": metrics["LogProb_Precision_0"],
        "LogProb_Recall_0": metrics["LogProb_Recall_0"],
        "LogProb_F1_0": metrics["LogProb_F1_0"],

        "LogProb_Support_1": metrics["LogProb_Support_1"],
        "LogProb_Predicted_as_1": metrics["LogProb_Predicted_as_1"],
        "LogProb_Precision_1": metrics["LogProb_Precision_1"],
        "LogProb_Recall_1": metrics["LogProb_Recall_1"],
        "LogProb_F1_1": metrics["LogProb_F1_1"],

        "avg_logprob_margin": metrics["avg_logprob_margin"],
        "avg_confidence": metrics["avg_confidence"],
        "avg_entropy": metrics["avg_entropy"],
        "avg_confidence_correct_logprob": metrics["avg_confidence_correct_logprob"],
        "avg_confidence_wrong_logprob": metrics["avg_confidence_wrong_logprob"],
        "avg_margin_correct_logprob": metrics["avg_margin_correct_logprob"],
        "avg_margin_wrong_logprob": metrics["avg_margin_wrong_logprob"],
        "free_vs_logprob_agreement_rate": metrics["free_vs_logprob_agreement_rate"],
        "ambiguous_rate": metrics["ambiguous_rate"],
        "high_confidence_wrong_rate": metrics["high_confidence_wrong_rate"],
        "low_confidence_correct_rate": metrics["low_confidence_correct_rate"],
        "actual_0_pred_0": metrics["actual_0_pred_0"],
        "actual_0_pred_1": metrics["actual_0_pred_1"],
        "actual_1_pred_0": metrics["actual_1_pred_0"],
        "actual_1_pred_1": metrics["actual_1_pred_1"],
        "logprob_actual_0_pred_0": metrics["logprob_actual_0_pred_0"],
        "logprob_actual_0_pred_1": metrics["logprob_actual_0_pred_1"],
        "logprob_actual_1_pred_0": metrics["logprob_actual_1_pred_0"],
        "logprob_actual_1_pred_1": metrics["logprob_actual_1_pred_1"],
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


def build_logprob_confusion_matrix_dataframe(metrics: JsonDict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            [metrics["logprob_actual_0_pred_0"], metrics["logprob_actual_0_pred_1"]],
            [metrics["logprob_actual_1_pred_0"], metrics["logprob_actual_1_pred_1"]],
        ],
        index=["actual_0", "actual_1"],
        columns=["logprob_predicted_0", "logprob_predicted_1"],
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
                "logprob_actual_0_pred_0": metrics["logprob_actual_0_pred_0"],
                "logprob_actual_0_pred_1": metrics["logprob_actual_0_pred_1"],
                "logprob_actual_1_pred_0": metrics["logprob_actual_1_pred_0"],
                "logprob_actual_1_pred_1": metrics["logprob_actual_1_pred_1"],
            }
        )

    return pd.DataFrame(rows)


def dataframe_to_string(df: pd.DataFrame, index: bool = False) -> str:
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 240,
        "display.max_colwidth", 200,
    ):
        return df.to_string(index=index, float_format=lambda x: f"{x:.4f}")


def build_metrics_report(
    metrics_summary: JsonDict,
    confidence_bucket_df: pd.DataFrame,
    thresholds: AggregateThresholds,
) -> str:
    print("[INFO] Building metrics text report...")

    global_df = build_global_metrics_dataframe(metrics_summary)
    category_df = build_category_metrics_dataframe(metrics_summary)
    video_df = build_video_metrics_dataframe(metrics_summary)
    global_conf_df = build_confusion_matrix_dataframe(metrics_summary["global"])
    global_logprob_conf_df = build_logprob_confusion_matrix_dataframe(metrics_summary["global"])
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
        "- Accuracy is exact-match accuracy over all examples using the free-form decoded output.",
        "- Accuracy_valid_only is computed only on valid parsed free-form predictions.",
        "- Precision, recall and F1 for free-form outputs are reported separately for label 0 and label 1.",
        "- LogProb_Accuracy is computed from the preferred label between 0 and 1 according to the model log probabilities.",
        "- LogProb precision, recall and F1 are also reported separately for label 0 and label 1.",
        "- No macro-average or micro-average is used.",
        "- Invalid predictions are outputs that could not be parsed as 0 or 1.",
        f"- Empty raw outputs are normalized to the explicit token: {EMPTY_RAW_OUTPUT_TOKEN}",
        "- avg_logprob_margin = average absolute gap |logprob_0 - logprob_1|.",
        "- avg_confidence = average max(prob_0, prob_1) after normalizing over {0,1}.",
        "- avg_entropy measures uncertainty over the binary decision; lower is sharper.",
        f"- ambiguous_rate uses margin < {thresholds.ambiguous_margin:.2f}.",
        f"- high_confidence_wrong_rate uses confidence >= {thresholds.high_confidence:.2f} and wrong logprob decision.",
        f"- low_confidence_correct_rate uses confidence < {thresholds.low_confidence:.2f} and correct logprob decision.",
        "- free_vs_logprob_agreement_rate measures agreement between the decoded free answer and the label preferred by logprob.",
        "",
        "GLOBAL PERFORMANCE",
        "------------------",
        dataframe_to_string(global_df, index=False),
        "",
        "GLOBAL CONFUSION MATRIX (FREE OUTPUT)",
        "-------------------------------------",
        dataframe_to_string(global_conf_df, index=True),
        "",
        "GLOBAL CONFUSION MATRIX (LOGPROB CHOICE)",
        "----------------------------------------",
        dataframe_to_string(global_logprob_conf_df, index=True),
        "",
        "CONFIDENCE BUCKET ANALYSIS",
        "--------------------------",
        dataframe_to_string(confidence_bucket_df, index=False),
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


def plot_logprob_confusion_matrix(metrics_summary: JsonDict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conf_df = build_logprob_confusion_matrix_dataframe(metrics_summary["global"])
    values = conf_df.values

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(values)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(conf_df.columns)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(conf_df.index)
    ax.set_title("Global confusion matrix (logprob choice)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, str(values[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Global logprob confusion matrix plot saved to: {output_path}")


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
        print(f"[INFO] Skipping plot {title}: empty DataFrame.")
        return

    plot_df = df.sort_values(metric_column, ascending=False)

    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 0.8), 5))
    ax.bar(plot_df[plot_df.columns[0]], plot_df[metric_column])

    ax.set_title(title)
    ax.set_xlabel(plot_df.columns[0])
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)

    values = plot_df[metric_column].tolist()
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Plot saved to: {output_path}")


def plot_confidence_bucket_accuracy(confidence_bucket_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if confidence_bucket_df.empty:
        print("[INFO] Skipping confidence bucket plot: empty DataFrame.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(confidence_bucket_df["confidence_bucket"], confidence_bucket_df["logprob_accuracy"])
    ax.set_title("Logprob accuracy by confidence bucket")
    ax.set_xlabel("Confidence bucket")
    ax.set_ylabel("Logprob accuracy")
    ax.tick_params(axis="x", rotation=30)

    for idx, value in enumerate(confidence_bucket_df["logprob_accuracy"].tolist()):
        ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Confidence bucket plot saved to: {output_path}")


def build_results_json(
    records: Sequence[PredictionRecord],
    final_results: JsonDict,
    metrics_summary: JsonDict,
) -> JsonDict:
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
            "predicted_label_by_logprob": record.predicted_label_by_logprob,
            "is_correct": record.is_correct,
            "is_correct_by_logprob": record.is_correct_by_logprob,
            "logprob_0": record.logprob_0,
            "logprob_1": record.logprob_1,
            "prob_0": record.prob_0,
            "prob_1": record.prob_1,
            "logprob_margin": record.logprob_margin,
            "confidence": record.confidence,
            "entropy": record.entropy,
            "free_vs_logprob_agree": record.free_vs_logprob_agree,
            "is_ambiguous_by_logprob": record.is_ambiguous_by_logprob,
            "is_high_confidence_wrong_by_logprob": record.is_high_confidence_wrong_by_logprob,
            "is_low_confidence_correct_by_logprob": record.is_low_confidence_correct_by_logprob,
        }

    print("[INFO] Final results JSON structure built successfully.")
    return results


def build_logprob_dataframe(records: Sequence[PredictionRecord]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for record in records:
        row = asdict(record)
        rows.append(row)

    return pd.DataFrame(rows)


def build_logprob_json(records: Sequence[PredictionRecord]) -> JsonDict:
    return {"records": [asdict(record) for record in records]}


def build_summary_snapshot_json(
    metrics_summary: JsonDict,
    confidence_bucket_df: pd.DataFrame,
    thresholds: AggregateThresholds,
) -> JsonDict:
    return {
        "thresholds": asdict(thresholds),
        "global": metrics_summary["global"],
        "confidence_buckets": confidence_bucket_df.to_dict(orient="records"),
    }


def save_evaluation_artifacts(
    metrics_summary: JsonDict,
    output_dir: str | Path,
    confidence_bucket_df: pd.DataFrame,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    global_df = build_global_metrics_dataframe(metrics_summary)
    category_df = build_category_metrics_dataframe(metrics_summary)
    video_df = build_video_metrics_dataframe(metrics_summary)
    global_conf_df = build_confusion_matrix_dataframe(metrics_summary["global"])
    global_logprob_conf_df = build_logprob_confusion_matrix_dataframe(metrics_summary["global"])
    category_conf_df = build_category_confusion_dataframe(metrics_summary)

    workbook_path = output_dir / "evaluation_summary.xlsx"
    global_csv_path = output_dir / "global_metrics.csv"
    category_csv_path = output_dir / "normalized_question_category_metrics.csv"
    video_csv_path = output_dir / "video_metrics.csv"
    category_conf_csv_path = output_dir / "normalized_question_category_confusion.csv"
    confidence_bucket_csv_path = output_dir / "confidence_bucket_analysis.csv"

    with pd.ExcelWriter(workbook_path) as writer:
        global_df.to_excel(writer, sheet_name="global_metrics", index=False)
        category_df.to_excel(writer, sheet_name="category_metrics", index=False)
        video_df.to_excel(writer, sheet_name="video_metrics", index=False)
        global_conf_df.to_excel(writer, sheet_name="global_confusion")
        global_logprob_conf_df.to_excel(writer, sheet_name="logprob_global_confusion")
        category_conf_df.to_excel(writer, sheet_name="category_confusion", index=False)
        confidence_bucket_df.to_excel(writer, sheet_name="confidence_buckets", index=False)

    save_dataframe_csv(global_df, global_csv_path)
    save_dataframe_csv(category_df, category_csv_path)
    save_dataframe_csv(video_df, video_csv_path)
    save_dataframe_csv(category_conf_df, category_conf_csv_path)
    save_dataframe_csv(confidence_bucket_df, confidence_bucket_csv_path)

    plot_global_confusion_matrix(metrics_summary, plots_dir / "global_confusion_matrix.png")
    plot_logprob_confusion_matrix(metrics_summary, plots_dir / "global_logprob_confusion_matrix.png")
    plot_metric_by_category(
        category_df,
        metric_column="Accuracy",
        title="Accuracy by normalized question category",
        ylabel="Accuracy",
        output_path=plots_dir / "accuracy_by_category.png",
    )
    plot_metric_by_category(
        category_df,
        metric_column="LogProb_Accuracy",
        title="LogProb accuracy by normalized question category",
        ylabel="LogProb accuracy",
        output_path=plots_dir / "logprob_accuracy_by_category.png",
    )
    plot_metric_by_category(
        category_df,
        metric_column="F1_0",
        title="F1 for free-form label 0 by normalized question category",
        ylabel="F1 (label 0)",
        output_path=plots_dir / "f1_0_by_category.png",
    )
    plot_metric_by_category(
        category_df,
        metric_column="F1_1",
        title="F1 for free-form label 1 by normalized question category",
        ylabel="F1 (label 1)",
        output_path=plots_dir / "f1_1_by_category.png",
    )
    plot_metric_by_category(
        category_df,
        metric_column="LogProb_F1_0",
        title="F1 for logprob label 0 by normalized question category",
        ylabel="LogProb F1 (label 0)",
        output_path=plots_dir / "logprob_f1_0_by_category.png",
    )
    plot_metric_by_category(
        category_df,
        metric_column="LogProb_F1_1",
        title="F1 for logprob label 1 by normalized question category",
        ylabel="LogProb F1 (label 1)",
        output_path=plots_dir / "logprob_f1_1_by_category.png",
    )
    plot_metric_by_category(
        category_df,
        metric_column="avg_confidence",
        title="Average binary confidence by normalized question category",
        ylabel="Average confidence",
        output_path=plots_dir / "avg_confidence_by_category.png",
    )
    plot_metric_by_category(
        category_df,
        metric_column="ambiguous_rate",
        title="Ambiguous rate by normalized question category",
        ylabel="Ambiguous rate",
        output_path=plots_dir / "ambiguous_rate_by_category.png",
    )
    plot_metric_by_category(
        category_df,
        metric_column="high_confidence_wrong_rate",
        title="High-confidence wrong rate by normalized question category",
        ylabel="High-confidence wrong rate",
        output_path=plots_dir / "high_confidence_wrong_rate_by_category.png",
    )
    plot_confidence_bucket_accuracy(
        confidence_bucket_df,
        plots_dir / "logprob_accuracy_by_confidence_bucket.png",
    )

    print(f"[INFO] Evaluation workbook saved to: {workbook_path}")
    print(f"[INFO] Evaluation CSVs saved in: {output_dir}")
    print(f"[INFO] Evaluation plots saved in: {plots_dir}")


def save_logprob_artifacts(
    records: Sequence[PredictionRecord],
    metrics_summary: JsonDict,
    output_dir: str | Path,
    confidence_bucket_df: pd.DataFrame,
    thresholds: AggregateThresholds,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "qwen_mc_1B_logprob_results.json"
    csv_path = output_dir / "qwen_mc_1B_logprob_results.csv"
    xlsx_path = output_dir / "qwen_mc_1B_logprob_results.xlsx"
    summary_json_path = output_dir / "qwen_mc_1B_logprob_summary.json"
    summary_txt_path = output_dir / "qwen_mc_1B_logprob_summary.txt"

    logprob_df = build_logprob_dataframe(records)
    logprob_json = build_logprob_json(records)
    summary_snapshot_json = build_summary_snapshot_json(
        metrics_summary=metrics_summary,
        confidence_bucket_df=confidence_bucket_df,
        thresholds=thresholds,
    )

    save_json_file(logprob_json, json_path)
    save_dataframe_csv(logprob_df, csv_path)
    save_json_file(summary_snapshot_json, summary_json_path)

    with pd.ExcelWriter(xlsx_path) as writer:
        logprob_df.to_excel(writer, sheet_name="logprob_results", index=False)
        confidence_bucket_df.to_excel(writer, sheet_name="confidence_buckets", index=False)
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

    summary_text = build_metrics_report(
        metrics_summary=metrics_summary,
        confidence_bucket_df=confidence_bucket_df,
        thresholds=thresholds,
    )
    save_text_file(summary_text, summary_txt_path)

    print(f"[INFO] Logprob JSON saved to: {json_path}")
    print(f"[INFO] Logprob CSV saved to: {csv_path}")
    print(f"[INFO] Logprob XLSX saved to: {xlsx_path}")
    print(f"[INFO] Logprob summary JSON saved to: {summary_json_path}")
    print(f"[INFO] Logprob summary TXT saved to: {summary_txt_path}")


def run_inference(
    examples: Sequence[Example],
    final_results: JsonDict,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    tokenizer: PreTrainedTokenizerBase,
    thresholds: AggregateThresholds,
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

        logprob_results = compute_binary_logprobs_batch(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            prompts=prompts,
        )

        print(f"[INFO] Received {len(raw_outputs)} raw outputs for current batch.")
        print(f"[INFO] Computed {len(logprob_results)} logprob results for current batch.")

        for example, transcript, prompt, raw_output, logprob_result in zip(
            batch_examples,
            transcripts,
            prompts,
            raw_outputs,
            logprob_results,
        ):
            safe_raw_output = normalize_raw_output(raw_output)
            predicted_label = parse_binary_answer(safe_raw_output)
            is_correct = predicted_label == example.target

            predicted_label_by_logprob = int(logprob_result["predicted_label_by_logprob"])
            is_correct_by_logprob = predicted_label_by_logprob == example.target
            confidence = float(logprob_result["confidence"])
            logprob_margin = float(logprob_result["logprob_margin"])

            free_vs_logprob_agree = (
                predicted_label is not None and predicted_label == predicted_label_by_logprob
            )
            is_ambiguous_by_logprob = logprob_margin < thresholds.ambiguous_margin
            is_high_confidence_wrong_by_logprob = (
                confidence >= thresholds.high_confidence and not is_correct_by_logprob
            )
            is_low_confidence_correct_by_logprob = (
                confidence < thresholds.low_confidence and is_correct_by_logprob
            )

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
                    logprob_0=float(logprob_result["logprob_0"]),
                    logprob_1=float(logprob_result["logprob_1"]),
                    prob_0=float(logprob_result["prob_0"]),
                    prob_1=float(logprob_result["prob_1"]),
                    predicted_label_by_logprob=predicted_label_by_logprob,
                    logprob_margin=logprob_margin,
                    confidence=confidence,
                    entropy=float(logprob_result["entropy"]),
                    free_vs_logprob_agree=free_vs_logprob_agree,
                    is_ambiguous_by_logprob=is_ambiguous_by_logprob,
                    is_high_confidence_wrong_by_logprob=is_high_confidence_wrong_by_logprob,
                    is_low_confidence_correct_by_logprob=is_low_confidence_correct_by_logprob,
                    is_correct_by_logprob=is_correct_by_logprob,
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
    predictions_output_path = Path("Data/ModelResponse/1B/qwen_mc_1B_predictions_by_video.json")
    metrics_report_output_path = Path("Data/ModelResponse/1B/qwen_mc_1B_metrics_report.txt")
    evaluation_output_dir = Path("Data/ModelResponse/1B/evaluation")

    # New output dir for 1B logprob artifacts
    logprob_output_dir = Path("Data/ModelResponse/1B")

    # Inference configuration
    batch_size = 4
    thresholds = AggregateThresholds(
        ambiguous_margin=0.20,
        high_confidence=0.80,
        low_confidence=0.55,
    )

    print("[INFO] Configuration loaded.")
    print(f"[INFO] final_results_path: {final_results_path}")
    print(f"[INFO] dataset_path: {dataset_path}")
    print(f"[INFO] predictions_output_path: {predictions_output_path}")
    print(f"[INFO] metrics_report_output_path: {metrics_report_output_path}")
    print(f"[INFO] evaluation_output_dir: {evaluation_output_dir}")
    print(f"[INFO] logprob_output_dir: {logprob_output_dir}")
    print(f"[INFO] batch_size: {batch_size}")
    print(f"[INFO] thresholds: {thresholds}")

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
        tokenizer=tokenizer,
        thresholds=thresholds,
        batch_size=batch_size,
    )
    print(f"[INFO] Inference phase completed. Total prediction records: {len(records)}")

    print("[INFO] Starting metrics computation...")
    metrics_summary = compute_all_metrics(records)
    confidence_bucket_df = build_confidence_bucket_dataframe(records)
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

    metrics_report = build_metrics_report(
        metrics_summary=metrics_summary,
        confidence_bucket_df=confidence_bucket_df,
        thresholds=thresholds,
    )
    save_text_file(metrics_report, metrics_report_output_path)

    save_evaluation_artifacts(
        metrics_summary=metrics_summary,
        output_dir=evaluation_output_dir,
        confidence_bucket_df=confidence_bucket_df,
    )
    save_logprob_artifacts(
        records=records,
        metrics_summary=metrics_summary,
        output_dir=logprob_output_dir,
        confidence_bucket_df=confidence_bucket_df,
        thresholds=thresholds,
    )

    print(f"[INFO] Predictions saved to: {predictions_output_path}")
    print(f"[INFO] Metrics report saved to: {metrics_report_output_path}")
    print(f"[INFO] Evaluation artifacts saved to: {evaluation_output_dir}")
    print(f"[INFO] Logprob artifacts saved to: {logprob_output_dir}")
    print("[INFO] Script finished successfully.")


if __name__ == "__main__":
    main()
