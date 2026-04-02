import json
import pandas as pd
import re
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple

# Initialize the dictionary that will store all Whisper results.
whisper_dict: Dict[str, Any] = {}


# Compute the average metrics across all segments of a single video entry.
def _segment_avg(video_entry: Dict[str, Any]) -> Dict[str, float]:
    lp = 0
    cr = 0
    nsp = 0
    id = 0

    # Sum all segment-level metrics.
    for segment in video_entry["segments"]:
        id = segment["id"]
        lp += segment["avg_logprob"]
        cr += segment["compression_ratio"]
        nsp += segment["no_speech_prob"]

    # DEBUG: print(id + 1)

    # Return the average values over all segments.
    return {
        "text_logprob": lp / (id + 1),
        "text_compression_ratio": cr / (id + 1),
        "text_no_speech_prob": nsp / (id + 1)
    }


# Add the averaged metrics to each video entry in the Whisper dictionary.
def whisper_performance_calculator(whisper_dict: Dict[str, Any]) -> Dict[str, Any]:
    for path in whisper_dict.keys():
        whisper_dict[path]["metrics"] = _segment_avg(whisper_dict[path])
    return whisper_dict


# Normalize text by trimming spaces, converting to lowercase,
# and collapsing multiple spaces into one.
def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


# Split normalized text into word tokens.
def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", _normalize_text(text), flags=re.UNICODE)


# Compute the repetition score of a token list.
# It is the relative frequency of the most common token.
def _repetition_score(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    c = Counter(tokens)
    return c.most_common(1)[0][1] / len(tokens)


# Heuristically detect whether a transcription looks like music,
# applause, noise, or repetitive vocal sounds.
def _looks_like_music(text: str) -> bool:
    t = _normalize_text(text)
    tokens = _tokenize(t)

    if not t:
        return False

    patterns = [
        r"\bla\b", r"\boh\b", r"\bah\b", r"\buh\b",
        r"\bmmm+\b", r"\bna\b", r"\bdum\b",
        r"\bmusica\b", r"\bapplausi\b", r"\brumore\b"
    ]

    # Check if any known music/noise-like pattern appears.
    if any(re.search(p, t) for p in patterns):
        return True

    # Check strong token repetition.
    if len(tokens) >= 2 and _repetition_score(tokens) >= 0.6:
        return True

    # Check very short repeated text like "la la la".
    if len(tokens) in (1, 2, 3) and len(set(tokens)) == 1:
        return True

    return False


# Classify the transcription into one of the target context labels.
# Returns:
# - the context label
# - the transcription text if it is classified as dialogue, otherwise None
def _classify_text(whisper: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    text = _normalize_text(whisper["text"])
    avg_logprob = float(whisper["metrics"]["text_logprob"])
    compression_ratio = float(whisper["metrics"]["text_compression_ratio"])
    no_speech_prob = float(whisper["metrics"]["text_no_speech_prob"])

    tokens = _tokenize(text)
    repetition = _repetition_score(tokens) if tokens else 0
    music_text = _looks_like_music(text)

    # 1) Silence / noise
    if not text or (avg_logprob < -1 and no_speech_prob >= 0.5):
        return "silence/noise", None

    # 2) Music (heuristic)
    if (music_text and avg_logprob < -0.75) or (repetition >= 0.6 and avg_logprob < -0.7):
        return "music", None

    # 3) Dialogue
    if (
        avg_logprob > -0.6
        and compression_ratio < 2.4
        and no_speech_prob < 0.5
        and len(tokens) >= 2
    ) or (avg_logprob > -1 and compression_ratio < 2.4 and not music_text):
        return "dialogue", whisper["text"]

    # Fallback classification
    return "unknown_dialogue", whisper["text"]


# Add the context label and the final transcription field to each video entry.
def context_flagger(whisper_dict: Dict[str, Any]) -> Dict[str, Any]:
    for path in whisper_dict.keys():
        whisper_dict[path]["context"], text = _classify_text(whisper_dict[path])

        if text is None:
            whisper_dict[path]["transcription"] = ""
        else:
            whisper_dict[path]["transcription"] = text

    return whisper_dict


# Main execution function.
def main() -> None:
    global whisper_dict

    # Load the raw Whisper verbose transcription file,
    # compute aggregate metrics, and assign context labels.
    with open("Data/TranscriptionData/whisper/whisper_verbose_transcription.json", "r", encoding="utf-8") as whisper_file:
        whisper_dict = context_flagger(whisper_performance_calculator(json.load(whisper_file)))

    # whisper_dict = dict(sorted(whisper_dict.items(), key=lambda x: x[1]["context_type"]))

    # Build a tabular structure containing the metrics and the assigned context.
    data = [
        {
            "text_logprob": video_entry["metrics"]["text_logprob"],
            "compression_ratio": video_entry["metrics"]["text_compression_ratio"],
            "no_speech_prob": video_entry["metrics"]["text_no_speech_prob"],
            "context": video_entry["context"],
            "transcription": video_entry["transcription"]
        }
        for video_entry in whisper_dict.values()
    ]

    df = pd.DataFrame(data, index=whisper_dict.keys())

    # text_data = [{"type": video_entry["context"]} for video_entry in whisper_dict.values()]
    # df2 = pd.DataFrame(text_data, index=whisper_dict.keys())

    # Save the readable metrics table to a text file.
    with open("Data/TranscriptionData/whisper/metrics/pandas/pandas_whisper_metrics.txt", "w", encoding="utf-8") as file:
        file.write(df.to_string())
        # file.write("\n" * 3)
        # file.write(df2.to_string())

    # Save the metrics DataFrame as a parquet file.
    df.to_parquet("Data/TranscriptionData/whisper/metrics/pandas/whisper.parquet", engine="pyarrow", compression="snappy")

    # Remove raw segment data before saving the final JSON output.
    for d in whisper_dict.values():
        del d["segments"]
        del d["text"]

    # print(whisper_dict)

    # Save the processed Whisper dictionary as JSON.
    with open("Data/TranscriptionData/whisper/metrics/whisper_metrics.json", "w", encoding="utf-8") as save:
        json.dump(whisper_dict, save, ensure_ascii=False, indent=2)
    return

# Execute the script only when run directly.
if __name__ == "__main__":
    main()