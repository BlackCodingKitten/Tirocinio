import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from typing import Any
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

global video_path = "Data/Videos/"

def load_model(
    model_name: str,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
):
    """
    Load only Qwen2.5-VL model, processor and tokenizer.
    """
    print(f"Loading Qwen model: {model_name}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    tokenizer = processor.tokenizer

    print("Qwen loaded.")
    return model, processor, tokenizer


def load_final_results(json_path: str):
    """
    Load final_results.json from disk.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_final_results(data: dict ,json_path: str):
    """
    Save the results on disk.
    """
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data,f)
    print("File Salvato OK")
    return

def get_transcript_for_video(final_results, fr_key: str) -> str:
    """
    Retrieve the transcript, if exists, for the given video file or return transcription not available.

    Expected JSON structure example:
    {
        "Video12.mp4": {
            "classification": "dialogue",
            "score": -0.3305642837974472,
            "score_meaning": "very reliable",
            "selected_model": "both",
            "generated_transcription": "..."
        }
    }

    The lookup is performed using the dictionary key.
    """
    if "dialogue" in final_results[fr_key] ["classification"]:
        return f"La trascrizione dell'audio contenuto nel video è \"{final_results[fr_key]["generated_transcription"]}\""
    else: 
        return "La trascrizione dell'audio contenuto nel video non è disponibile."


def build_message_TV(prompt_text):
    """
    Keep the original function name to minimize code changes,
    but now build a text-only message instead of a video+text message.
    """
    message = [
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
        }
    ]
    return message


def inference(message, processor, model):
    """
    Apply chat template and process text-only inputs.
    Returns:
        text
        inputs
    """
    text = processor.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    return text, inputs


def generate_answer(model, processor, inputs):
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    return output_text.strip()


def main():
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    caption = "L'uomo è caduto in acqua perché ha perso l'equilibrio"
    foil = "L'uomo è caduto in acqua perché è stato spinto"

    json_path = "Data/TranscriptionData/final_classification/final_results.json"
    

    final_results = load_final_results(json_path)
    

    prompt_text_transcription = (
        "Ti viene fornita solo la trascrizione automatica dell'audio del video.\n"
        "La trascrizione è in italiano, può contenere errori, omissioni o parole sbagliate.\n"
        "Usa solo la trascrizione per scegliere la descrizione corretta.\n\n"
        f"Trascrizione:\n{transcript}\n\n"
        f"Scegli la descrizione corretta:\n"
        f"A. {caption}\n"
        f"B. {foil}.\n"
        f"Rispondi solo A o B.\n"
    )
    
    prompt_text_no_transcription= (
        f"Scegli la descrizione che ha la maggior probabilità di essere corretta:\n"
        f"A. {caption}\n"
        f"B. {foil}.\n"
        f"Rispondi solo A o B.\n"
    )

    model, processor, tokenizer = load_model(model_name)


with open("Data/ModelResponse/random.json", "w", encoding="utf-8") as result_file: 
    result_dict = dict[str, Any] = {}
    for in in range (1,101):
        key = f"Video{i}.mp4"  

        transcript =get_transcript_for_video(final_results, key)
        result_dict[key]["transcript"] = transcript
        result_dict[key]["formatted_prompt"] = text

        message = build_message_TV(prompt_text_no_transcription)
        text, inputs = inference(message, processor, model)
        answer = generate_answer(model, processor, inputs)

        print("\n===== MODEL ANSWER =====")
        print(answer)


if __name__ == "__main__":
    main()


