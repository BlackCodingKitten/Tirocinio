from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, cast

import torch
from transformers import AutoProcessor, PreTrainedTokenizerBase, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


# Select the GPU to use.
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def load_model(
    model_name: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "cuda:6",
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
        use_fast=True,
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


def resolve_video_path(video_path: str | Path) -> Path:
    path = Path(video_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    return path


def path_to_file_uri(path: str | Path) -> str:
    return Path(path).resolve().as_uri()


def build_message_for_video(
    prompt_text: str,
    video_path: str | Path,
    max_pixels: int = 360 * 420,
    fps: float = 2.0,
) -> List[Dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": "You are a precise assistant.",
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
    prompt_text: str,
    video_paths: Sequence[Path],
    max_pixels: int = 360 * 420,
    fps: float = 2.0,
) -> List[List[Dict[str, Any]]]:
    return [
        build_message_for_video(
            prompt_text=prompt_text,
            video_path=video_path,
            max_pixels=max_pixels,
            fps=fps,
        )
        for video_path in video_paths
    ]


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
            f"Missing keys: {missing}. Available keys: {available}."
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


def generate_answers_batch(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    prompt_text: str,
    video_paths: Sequence[Path],
    max_new_tokens: int = 128,
    max_pixels: int = 360 * 420,
    fps: float = 2.0,
) -> List[str]:
    messages = build_messages(
        prompt_text=prompt_text,
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

    assert_video_features_present(inputs, video_paths)

    target_device = get_model_input_device(model)
    inputs = move_inputs_to_device(inputs, target_device)

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


def list_mp4_videos(video_root_dir: str | Path) -> List[Path]:
    root = Path(video_root_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    videos = list(root.glob("*.mp4"))
    if not videos:
        raise ValueError(f"No .mp4 files found in: {root}")

    return videos


def main() -> None:
    print("[INFO] Script started.")

    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Metti qui o un singolo video...
    # video_paths = [resolve_video_path("Data/Videos/Video1.mp4")]

    # ...oppure tutti i video della cartella.
    # video_paths = list_mp4_videos("Data/Videos")
    video_paths = [Path(f"Data/Videos/Video{i}.mp4") for i in range(1,101)]
    # Cambia liberamente il prompt.
    prompt_text = "Describe briefly in what happens in this video."

    batch_size = 4
    max_pixels = 360 * 420
    fps = 2.0

    model, processor, tokenizer = load_model(model_name)
    print(f"[INFO] Tokenizer loaded: {type(tokenizer).__name__}")

    total_videos = len(video_paths)
    total_batches = (total_videos + batch_size - 1) // batch_size

    print(f"[INFO] Total videos: {total_videos}")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Total batches: {total_batches}")

    for start_idx in range(0, total_videos, batch_size):
        batch_video_paths = video_paths[start_idx:start_idx + batch_size]
        batch_number = (start_idx // batch_size) + 1

        print(
            f"[INFO] Processing batch {batch_number}/{total_batches} "
            f"(videos {start_idx + 1}-{start_idx + len(batch_video_paths)} of {total_videos})"
        )

        answers = generate_answers_batch(
            model=model,
            processor=processor,
            prompt_text=prompt_text,
            video_paths=batch_video_paths,
            max_new_tokens=128,
            max_pixels=max_pixels,
            fps=fps,
        )

        for video_path, answer in zip(batch_video_paths, answers):
            print("=" * 100)
            print(f"VIDEO: {video_path.name}")
            print(f"ANSWER: {answer}")
            print("=" * 100)

    print("[INFO] Script finished successfully.")


if __name__ == "__main__":
    main()