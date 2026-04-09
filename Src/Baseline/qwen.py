from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Select the GPU to use.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


JsonDict = Dict[str, Any]


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
    device_map: str = "auto",
    attn_implementation: str = "flash_attention_2",
) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    """
    Load the Qwen2.5-VL model and processor.
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

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

    print("[INFO] Processor loaded successfully.")
    print("[INFO] Model setup completed.")
    return model, processor


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
    print(f"[INFO] Saving JSON output to: {path}")
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] JSON output saved successfully: {path}")


def save_text_file(text: str, text_path: str | Path) -> None:
    """
    Save a plain text file to disk.
    """
    path = Path(text_path)
    print(f"[INFO] Saving text report to: {path}")
    with path.open("w", encoding="utf-8") as f:
        f.write(text)
    print(f"[INFO] Text report saved successfully: {path}")


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
    if the classification contains the substring 'dialogue' the transcription is returned. Otherwise, a standard fallback string
    is returned.
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

    The model must answer strictly with 0 or 1.
    """ 
    return (        
        "Scegli randomicamente quale descrizione ha la maggiore probabilità di essere corretta.\n"
        f"0. {choice_0}\n"
        f"1. {choice_1}\n\n"
        "Rispondi esclusivamente con 0 o 1."
        )
    # return (
    #     "Ti fornirò, se disponibile, la trascrizione dell'audio del video, potrebbe contenere degli errori, avere parti mancanti, o essere parziale "
    #     "e due descrizioni candidate del contenuto del video.\n\n"
    #     f"Trascrizione audio: {transcript}\n\n"
    #     "Scegli quale descrizione ha la maggiore probabilità di essere corretta.\n"
    #     f"0. {choice_0}\n"
    #     f"1. {choice_1}\n\n"
    #     "Rispondi esclusivamente con 0 o 1."
    # )


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
                                "You are a precise assistant. "
                                "Answer only with 0 or 1."
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

    # Remove the prompt tokens from the generated sequence.
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


def parse_binary_answer(raw_output: str) -> Optional[int]:
    """
    Extract the first valid binary answer from the model output.

    Returns:
    - 0 or 1 if a valid answer is found
    - None otherwise
    """
    cleaned = raw_output.strip()

    if cleaned in {"0", "1"}:
        return int(cleaned)

    match = re.search(r"\b([01])\b", cleaned)
    if match:
        return int(match.group(1))

    return None


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Safe floating-point division.
    """
    return numerator / denominator if denominator != 0 else 0.0


def compute_metrics(records: Sequence[PredictionRecord]) -> JsonDict:
    """
    Compute binary classification metrics over labels {0, 1}.

    Definitions used:
    - accuracy: exact-match accuracy
    - precision_macro / recall_macro / f1_macro:
      macro average over labels 0 and 1
    - invalid_predictions:
      number of model outputs that could not be parsed into 0 or 1

    Invalid predictions count as wrong for accuracy and contribute to FN where relevant.
    """
    print(f"[INFO] Computing metrics for {len(records)} records...")
    labels = (0, 1)
    n_samples = len(records)
    invalid_predictions = sum(
        record.predicted_label not in labels for record in records
    )
    correct_predictions = sum(record.is_correct for record in records)

    per_label: JsonDict = {}
    precision_values: List[float] = []
    recall_values: List[float] = []
    f1_values: List[float] = []

    for label in labels:
        tp = sum(
            record.predicted_label == label and record.target == label
            for record in records
        )
        fp = sum(
            record.predicted_label == label and record.target != label
            for record in records
        )
        fn = sum(
            record.predicted_label != label and record.target == label
            for record in records
        )
        support = sum(record.target == label for record in records)

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

        per_label[str(label)] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    print("[INFO] Metrics computed successfully.")
    return {
        "n_samples": n_samples,
        "invalid_predictions": invalid_predictions,
        "accuracy": safe_divide(correct_predictions, n_samples),
        "precision_macro": safe_divide(sum(precision_values), len(labels)),
        "recall_macro": safe_divide(sum(recall_values), len(labels)),
        "f1_macro": safe_divide(sum(f1_values), len(labels)),
        "per_label": per_label,
    }


def compute_all_metrics(records: Sequence[PredictionRecord]) -> JsonDict:
    """
    Compute:
    - global metrics
    - metrics by normalized question class
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


def build_results_json(
    records: Sequence[PredictionRecord],
    final_results: JsonDict,
    metrics_summary: JsonDict,
) -> JsonDict:
    """
    Build the final per-video JSON file.

    Structure:
    {
      "Video1.mp4": {
        "classification": "...",
        "score": ...,
        "score_meaning": "...",
        "selected_model": "...",
        "generated_transcription": "...",
        "metrics": {...},
        "questions": {
          "Controfattuale_A": {
            "pool_pos_1": {
              "normalized_question_category": "Controfattuale",
              "0": "...",
              "1": "...",
              "target": 0,
              "prompt": "...",
              "raw_model_output": "...",
              "predicted_label": 0,
              "is_correct": true
            }
          }
        }
      }
    }
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


def format_metric_block(title: str, metrics: JsonDict) -> str:
    """
    Format a single metric block as human-readable text.
    """
    lines: List[str] = [
        title,
        f"  n_samples: {metrics['n_samples']}",
        f"  invalid_predictions: {metrics['invalid_predictions']}",
        f"  accuracy: {metrics['accuracy']:.4f}",
        f"  precision_macro: {metrics['precision_macro']:.4f}",
        f"  recall_macro: {metrics['recall_macro']:.4f}",
        f"  f1_macro: {metrics['f1_macro']:.4f}",
        "  per_label:",
    ]

    per_label = metrics.get("per_label", {})
    for label in ("0", "1"):
        label_metrics = per_label.get(label, {})
        lines.extend(
            [
                f"    label {label}:",
                f"      support: {label_metrics.get('support', 0)}",
                f"      tp: {label_metrics.get('tp', 0)}",
                f"      fp: {label_metrics.get('fp', 0)}",
                f"      fn: {label_metrics.get('fn', 0)}",
                f"      precision: {label_metrics.get('precision', 0.0):.4f}",
                f"      recall: {label_metrics.get('recall', 0.0):.4f}",
                f"      f1: {label_metrics.get('f1', 0.0):.4f}",
            ]
        )

    return "\n".join(lines)


def build_metrics_report(metrics_summary: JsonDict) -> str:
    """
    Build a readable TXT report for global metrics, per normalized class, and per video.
    """
    print("[INFO] Building metrics text report...")
    lines: List[str] = [
        "METRICS REPORT",
        "==============",
        "",
        "Note:",
        "- accuracy is exact-match accuracy",
        "- precision/recall/f1 are macro averages over labels 0 and 1",
        "- invalid predictions are model outputs that could not be parsed as 0 or 1",
        "",
        format_metric_block("GLOBAL", metrics_summary["global"]),
        "",
        "BY NORMALIZED QUESTION CATEGORY",
        "-------------------------------",
    ]

    for class_name, class_metrics in metrics_summary["by_normalized_question_category"].items():
        lines.append("")
        lines.append(format_metric_block(class_name, class_metrics))

    lines.extend(
        [
            "",
            "BY VIDEO",
            "--------",
        ]
    )

    by_video = metrics_summary["by_video"]
    for video_id in sorted(by_video.keys(), key=natural_video_sort_key):
        lines.append("")
        lines.append(format_metric_block(video_id, by_video[video_id]))

    print("[INFO] Metrics text report built successfully.")
    return "\n".join(lines)


def run_inference(
    examples: Sequence[Example],
    final_results: JsonDict,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    batch_size: int = 8,
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
            predicted_label = parse_binary_answer(raw_output)
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
                    raw_model_output=raw_output,
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
    predictions_output_path = Path("Data/ModelResponse/Random/qwen_mc_random_predictions_by_video.json")
    metrics_report_output_path = Path("Data/ModelResponse/Random/qwen_mc_random_metrics_report.txt")

    # Inference configuration
    batch_size = 8

    print("[INFO] Configuration loaded.")
    print(f"[INFO] final_results_path: {final_results_path}")
    print(f"[INFO] dataset_path: {dataset_path}")
    print(f"[INFO] predictions_output_path: {predictions_output_path}")
    print(f"[INFO] metrics_report_output_path: {metrics_report_output_path}")
    print(f"[INFO] batch_size: {batch_size}")

    # Load inputs
    print("[INFO] Loading input files...")
    final_results = load_json_file(final_results_path)
    dataset_json = load_json_file(dataset_path)
    print("[INFO] Input files loaded successfully.")

    # Extract all examples from the nested dataset JSON.
    examples = extract_examples(dataset_json)
    if not examples:
        raise ValueError("No examples were found in the dataset JSON.")

    print(f"[INFO] Total extracted examples: {len(examples)}")

    # Load model only once.
    print("[INFO] Loading model and processor...")
    model, processor = load_model(model_name)
    print("[INFO] Model and processor are ready.")

    # Run inference on all examples.
    print("[INFO] Starting inference phase...")
    records = run_inference(
        examples=examples,
        final_results=final_results,
        model=model,
        processor=processor,
        batch_size=batch_size,
    )
    print(f"[INFO] Inference phase completed. Total prediction records: {len(records)}")

    # Compute all requested metrics.
    print("[INFO] Starting metrics computation...")
    metrics_summary = compute_all_metrics(records)
    print("[INFO] Metrics computation completed.")

    # Build the final per-video JSON output.
    print("[INFO] Building final predictions JSON...")
    predictions_json = build_results_json(
        records=records,
        final_results=final_results,
        metrics_summary=metrics_summary,
    )
    print("[INFO] Final predictions JSON built successfully.")

    # Save outputs.
    print("[INFO] Saving output files...")
    save_json_file(predictions_json, predictions_output_path)

    metrics_report = build_metrics_report(metrics_summary)
    save_text_file(metrics_report, metrics_report_output_path)

    print(f"[INFO] Predictions saved to: {predictions_output_path}")
    print(f"[INFO] Metrics report saved to: {metrics_report_output_path}")
    print("[INFO] Script finished successfully.")


if __name__ == "__main__":
    # torch.cuda.init()
    main()