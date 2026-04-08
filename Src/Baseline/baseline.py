import json
import os
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union, cast

import torch
from transformers import AutoProcessor, PreTrainedTokenizerBase, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Restrict visible CUDA devices, mirroring the original script style.
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

JsonValue = Union[dict[str, Any], list[Any], str, int, float, bool, None]


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def load_model(
    model_name: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    attn_implementation: str = "sdpa",
) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor, PreTrainedTokenizerBase]:
    """
    Load the Qwen2.5-VL model, processor, and tokenizer.

    Notes:
        - The original script used `flash_attention_2`.
        - Here the default is `sdpa`, which is usually easier to run because it
          does not require compiling `flash-attn`.

    Args:
        model_name: Hugging Face model identifier.
        torch_dtype: Data type used to load the model.
        device_map: Device placement strategy.
        attn_implementation: Attention backend used by Transformers.

    Returns:
        A tuple containing model, processor, and tokenizer.
    """
    print(f"Loading Qwen model: {model_name}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    tokenizer = cast(PreTrainedTokenizerBase, processor.tokenizer)

    print("Qwen loaded.")
    return model, processor, tokenizer


# -----------------------------------------------------------------------------
# JSON / transcript utilities
# -----------------------------------------------------------------------------

def load_json(json_path: str) -> JsonValue:
    """
    Load a JSON file from disk.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Parsed JSON content.
    """
    with open(json_path, "r", encoding="utf-8") as file:
        return cast(JsonValue, json.load(file))



def normalize_path(path: str) -> str:
    """
    Normalize a path into POSIX format.

    This helps compare file paths across operating systems and different
    path-writing conventions.

    Args:
        path: Input path.

    Returns:
        Normalized POSIX-like path.
    """
    return Path(path).as_posix()



def extract_transcript_from_item(item: Any) -> Optional[str]:
    """
    Extract a transcript string from a JSON item.

    This helper supports several common schemas, for example:
        - {"text": "..."}
        - {"transcript": "..."}
        - {"transcription": "..."}
        - {"prediction": {"text": "..."}}
        - "plain transcript string"

    Args:
        item: JSON item that may contain a transcript.

    Returns:
        The extracted transcript if found, otherwise None.
    """
    if isinstance(item, str):
        cleaned = item.strip()
        return cleaned if cleaned else None

    if not isinstance(item, Mapping):
        return None

    direct_keys = ("text", "transcript", "transcription")
    for key in direct_keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    nested_keys = ("prediction", "result", "output", "data")
    for nested_key in nested_keys:
        nested_value = item.get(nested_key)
        if isinstance(nested_value, Mapping):
            nested_transcript = extract_transcript_from_item(nested_value)
            if nested_transcript:
                return nested_transcript

    return None



def get_transcript_for_video(results: JsonValue, video_path: str) -> str:
    """
    Retrieve the transcript associated with a given video path.

    Supported JSON patterns:
        1. Dictionary-like:
           {"/path/to/video.mp4": {"text": "..."}}

        2. List-like:
           [{"video_path": "/path/to/video.mp4", "text": "..."}]

    Matching is performed in two steps:
        - exact normalized path match
        - fallback match by file name only

    Args:
        results: Parsed content of `final_results.json`.
        video_path: Path of the target video.

    Returns:
        The transcript associated with the video.

    Raises:
        KeyError: If no transcript can be found.
    """
    normalized_video_path = normalize_path(video_path)
    target_name = Path(video_path).name

    if isinstance(results, Mapping):
        if normalized_video_path in results:
            transcript = extract_transcript_from_item(results[normalized_video_path])
            if transcript:
                return transcript

        for key, value in results.items():
            if isinstance(key, str) and Path(key).name == target_name:
                transcript = extract_transcript_from_item(value)
                if transcript:
                    return transcript

    elif isinstance(results, Sequence) and not isinstance(results, (str, bytes, bytearray)):
        for item in results:
            if not isinstance(item, Mapping):
                continue

            candidate_path_keys = ("video_path", "path", "video", "file", "filepath")
            candidate_path: Optional[str] = None
            for key in candidate_path_keys:
                value = item.get(key)
                if isinstance(value, str):
                    candidate_path = value
                    break

            if candidate_path is None:
                continue

            if normalize_path(candidate_path) == normalized_video_path or Path(candidate_path).name == target_name:
                transcript = extract_transcript_from_item(item)
                if transcript:
                    return transcript

    raise KeyError(f"No transcript found for video path: {video_path}")


# -----------------------------------------------------------------------------
# Prompt builders
# -----------------------------------------------------------------------------

def build_prompt_random(caption: str, foil: str) -> str:
    """
    Build a prompt for the random baseline.

    Important:
        A true random baseline should *not* use the model at all. We keep this
        helper only for logging consistency.

    Args:
        caption: Correct description candidate.
        foil: Incorrect description candidate.

    Returns:
        A simple prompt string.
    """
    return (
        "Choose the correct description:\n"
        f"A. {caption}\n"
        f"B. {foil}\n"
        "Answer with only A or B.\n"
    )



def build_prompt_transcript_only(caption: str, foil: str, transcript: str) -> str:
    """
    Build the prompt for the transcript-only baseline.

    The model is explicitly told that it does not have access to the video and
    must rely only on the automatic transcript.

    Args:
        caption: Candidate A.
        foil: Candidate B.
        transcript: Transcript associated with the video.

    Returns:
        The final transcript-only prompt.
    """
    return (
        "You are given the automatic transcript of a video's audio.\n"
        "The transcript may contain errors, omissions, or incorrect words.\n"
        "You do not have access to the video.\n"
        "Use only the transcript to choose the correct description.\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Choose the correct description:\n"
        f"A. {caption}\n"
        f"B. {foil}\n"
        "Answer with only A or B.\n"
    )



def build_prompt_video_only(caption: str, foil: str) -> str:
    """
    Build the prompt for the video-only baseline.

    This mirrors the spirit of the original script: the model receives the video
    and the A/B question, but no transcript.

    Args:
        caption: Candidate A.
        foil: Candidate B.

    Returns:
        The video-only prompt.
    """
    return (
        "Choose the correct description based only on the video.\n"
        f"A. {caption}\n"
        f"B. {foil}\n"
        "Answer with only A or B.\n"
    )



def build_prompt_video_plus_transcript(caption: str, foil: str, transcript: str) -> str:
    """
    Build the prompt for the video+transcript baseline.

    The model receives both modalities:
        - the video itself
        - the transcript as extra textual evidence

    Args:
        caption: Candidate A.
        foil: Candidate B.
        transcript: Transcript associated with the video.

    Returns:
        The video+transcript prompt.
    """
    return (
        "You are given a video and the automatic transcript of its audio.\n"
        "The transcript may contain errors, omissions, or incorrect words.\n"
        "Use both the video and the transcript to choose the correct description.\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Choose the correct description:\n"
        f"A. {caption}\n"
        f"B. {foil}\n"
        "Answer with only A or B.\n"
    )


# -----------------------------------------------------------------------------
# Message builders
# -----------------------------------------------------------------------------

def build_message_text_only(prompt_text: str) -> list[dict[str, Any]]:
    """
    Build a text-only chat message.

    Args:
        prompt_text: Final prompt sent to the model.

    Returns:
        Chat message list in Qwen-compatible format.
    """
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
            ],
        },
    ]



def build_message_video(prompt_text: str, video_path: str) -> list[dict[str, Any]]:
    """
    Build a video+text chat message.

    This follows the same structure as the original script, where the user
    message contains one video block and one text block.

    Args:
        prompt_text: Final textual prompt.
        video_path: Path to the video file.

    Returns:
        Chat message list in Qwen-compatible multimodal format.
    """
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 0.55,
                },
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


# -----------------------------------------------------------------------------
# Inference helpers
# -----------------------------------------------------------------------------

def inference_text_only(
    message: list[dict[str, Any]],
    processor: AutoProcessor,
    model: Qwen2_5_VLForConditionalGeneration,
) -> tuple[str, dict[str, Any]]:
    """
    Convert a text-only message into model-ready inputs.

    Args:
        message: Text-only chat message.
        processor: Hugging Face processor.
        model: Loaded Qwen model.

    Returns:
        A tuple containing:
            - the formatted prompt text
            - the processed input tensors on the model device
    """
    text = processor.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )

    prepared_inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    return text, prepared_inputs



def inference_video(
    message: list[dict[str, Any]],
    processor: AutoProcessor,
    model: Qwen2_5_VLForConditionalGeneration,
) -> tuple[str, dict[str, Any], Any, Any]:
    """
    Convert a video+text message into model-ready inputs.

    This closely mirrors the original script.

    Args:
        message: Multimodal chat message.
        processor: Hugging Face processor.
        model: Loaded Qwen model.

    Returns:
        A tuple containing:
            - the formatted prompt text
            - processed input tensors on the model device
            - extracted video inputs
            - auxiliary video kwargs
    """
    text = processor.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        message,
        return_video_kwargs=True,
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )

    prepared_inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    return text, prepared_inputs, video_inputs, video_kwargs



def generate_answer(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    inputs: dict[str, Any],
    max_new_tokens: int = 16,
) -> str:
    """
    Generate the final model answer from prepared inputs.

    Args:
        model: Loaded Qwen model.
        processor: Hugging Face processor.
        inputs: Processed model inputs.
        max_new_tokens: Maximum number of generated tokens.

    Returns:
        The decoded model output.
    """
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    return output_text.strip()


# -----------------------------------------------------------------------------
# Baseline runners
# -----------------------------------------------------------------------------

def run_random_baseline(caption: str, foil: str, seed: int = 42) -> str:
    """
    Run a true random baseline.

    Important:
        This function deliberately does NOT call the model. A controlled random
        baseline must ignore the input completely and sample uniformly from
        {"A", "B"}.

    Args:
        caption: Candidate A (unused, kept for API symmetry).
        foil: Candidate B (unused, kept for API symmetry).
        seed: Random seed for reproducibility.

    Returns:
        Either "A" or "B".
    """
    _ = caption
    _ = foil
    rng = random.Random(seed)
    return rng.choice(["A", "B"])



def run_transcript_only_baseline(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    caption: str,
    foil: str,
    transcript: str,
) -> tuple[str, str]:
    """
    Run the transcript-only baseline.

    Args:
        model: Loaded Qwen model.
        processor: Hugging Face processor.
        caption: Candidate A.
        foil: Candidate B.
        transcript: Transcript associated with the video.

    Returns:
        A tuple containing:
            - the formatted prompt
            - the model answer
    """
    prompt_text = build_prompt_transcript_only(caption, foil, transcript)
    message = build_message_text_only(prompt_text)
    formatted_prompt, inputs = inference_text_only(message, processor, model)
    answer = generate_answer(model, processor, inputs)
    return formatted_prompt, answer



def run_video_only_baseline(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    caption: str,
    foil: str,
    video_path: str,
) -> tuple[str, str]:
    """
    Run the video-only baseline.

    Args:
        model: Loaded Qwen model.
        processor: Hugging Face processor.
        caption: Candidate A.
        foil: Candidate B.
        video_path: Path to the target video.

    Returns:
        A tuple containing:
            - the formatted prompt
            - the model answer
    """
    prompt_text = build_prompt_video_only(caption, foil)
    message = build_message_video(prompt_text, video_path)
    formatted_prompt, inputs, _video_inputs, _video_kwargs = inference_video(message, processor, model)
    answer = generate_answer(model, processor, inputs)
    return formatted_prompt, answer



def run_video_plus_transcript_baseline(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    caption: str,
    foil: str,
    transcript: str,
    video_path: str,
) -> tuple[str, str]:
    """
    Run the video+transcript baseline.

    Args:
        model: Loaded Qwen model.
        processor: Hugging Face processor.
        caption: Candidate A.
        foil: Candidate B.
        transcript: Transcript associated with the video.
        video_path: Path to the target video.

    Returns:
        A tuple containing:
            - the formatted prompt
            - the model answer
    """
    prompt_text = build_prompt_video_plus_transcript(caption, foil, transcript)
    message = build_message_video(prompt_text, video_path)
    formatted_prompt, inputs, _video_inputs, _video_kwargs = inference_video(message, processor, model)
    answer = generate_answer(model, processor, inputs)
    return formatted_prompt, answer


# -----------------------------------------------------------------------------
# Main example
# -----------------------------------------------------------------------------

def main() -> None:
    """
    Example entry point.

    Available modes:
        - "random"
        - "transcript_only"
        - "video_only"
        - "video_plus_transcript"

    Change `mode` below depending on the baseline you want to run.
    """
    mode = "random"
    # mode = "transcript_only"
    # mode = "video_only"
    # mode = "video_plus_transcript"

    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    json_path = (
        "/home/mikela/Documents/Tirocinio/src/Tirocinio/Data/TranscriptionData/"
        "final_classification/final_results.json"
    )
    video_path = "/home/dtesta/MAIA_def/dataset_def/videos/1.mp4"

    caption = "L'uomo è caduto in acqua perché ha perso l'equilibrio"
    foil = "L'uomo è caduto in acqua perché è stato spinto"

    print(f"\n===== MODE: {mode} =====")

    if mode == "random":
        # The random baseline does not use model, video, or transcript.
        prompt_text = build_prompt_random(caption, foil)
        answer = run_random_baseline(caption, foil, seed=42)

        print("\n===== FORMATTED PROMPT =====")
        print(prompt_text)
        print("\n===== BASELINE ANSWER =====")
        print(answer)
        return

    # The remaining modes use the model.
    model, processor, _tokenizer = load_model(model_name)

    if mode == "transcript_only":
        results = load_json(json_path)
        transcript = get_transcript_for_video(results, video_path)

        print("\n===== TRANSCRIPT =====")
        print(transcript)

        formatted_prompt, answer = run_transcript_only_baseline(
            model=model,
            processor=processor,
            caption=caption,
            foil=foil,
            transcript=transcript,
        )

    elif mode == "video_only":
        formatted_prompt, answer = run_video_only_baseline(
            model=model,
            processor=processor,
            caption=caption,
            foil=foil,
            video_path=video_path,
        )

    elif mode == "video_plus_transcript":
        results = load_json(json_path)
        transcript = get_transcript_for_video(results, video_path)

        print("\n===== TRANSCRIPT =====")
        print(transcript)

        formatted_prompt, answer = run_video_plus_transcript_baseline(
            model=model,
            processor=processor,
            caption=caption,
            foil=foil,
            transcript=transcript,
            video_path=video_path,
        )

    else:
        raise ValueError(
            "Invalid mode. Choose one of: 'random', 'transcript_only', 'video_only', 'video_plus_transcript'."
        )

    print("\n===== FORMATTED PROMPT =====")
    print(formatted_prompt)

    print("\n===== MODEL ANSWER =====")
    print(answer)


if __name__ == "__main__":
    main()
