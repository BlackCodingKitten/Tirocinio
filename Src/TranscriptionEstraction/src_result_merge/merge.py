import json
import pandas as pd
from typing import Dict, Any, Optional

# Initialize the dictionary that will store the final merged results.
# Each key will be a formatted file name (for example: "Video1.mp4"),
# and each value will contain the final classification metadata.
result: Dict[str, Dict[str, Any]] = {}


# Convert a full file path into a shorter key.
# Example:
# "data/videos/Video1.mp4" -> "Video1.mp4"
def formatted_key(key: str) -> str:
    # Split the path by "/" and extract the third component,
    # which is expected to be the file name.
    return str(key).split("/")[2].strip()


# Compute a Whisper-specific reliability score.
# This score starts from the average log probability and then applies
# two penalties:
#
# 1) no_speech_prob penalty
#    - If Whisper believes the segment may contain no speech,
#      reliability should decrease.
#    - This is weighted quite strongly because silence/noise detection
#      is an important signal for transcription quality.
#
# 2) compression_ratio penalty
#    - Compression ratio is used as a signal of abnormal or degenerate output.
#    - The penalty is applied only when the ratio becomes suspicious,
#      using max(0, avg_cr - 1.5), so normal values are not penalized.
#
# The final formula is:
# avg_logprob - 0.7 * no_speech_prob - 0.25 * max(0, compression_ratio - 1.5)
def _score_whisper(avg_lp: float, avg_cr: float, avg_nsp: float) -> float:
    # Weight assigned to the no-speech probability penalty.
    # This value is intentionally strong enough to matter,
    # but not so strong that it completely overrides avg_logprob.
    lambda1: float = 0.7

    # Weight assigned to the compression-ratio penalty.
    # This is a secondary penalty used mainly to downgrade suspicious,
    # repetitive, or degenerate outputs.
    lambda2: float = 0.25

    # Apply both penalties and return the final Whisper score.
    return avg_lp - (lambda1 * avg_nsp) - (lambda2 * (max(0, (avg_cr - 1.5))))


# Combine Whisper and GPT-4o reliability into a single score
# when both models agree on the same context classification.
#
# The combination uses a weighted average:
# - Whisper contributes 45%
# - GPT-4o contributes 55%
#
# The slight preference for GPT-4o reflects the assumption that GPT-4o
# is usually more robust linguistically, while Whisper provides useful
# extra acoustic signals such as no_speech_prob and compression_ratio.
def _final_score(w: Dict[str, Any], g: Dict[str, Any]) -> float:
    # Compute the Whisper score using all available Whisper metrics.
    w_score: float = _score_whisper(
        w["text_logprob"],
        w["compression_ratio"],
        w["no_speech_prob"]
    )

    # GPT-4o contributes only its text log probability.
    g_score: float = g["text_logprob"]

    # Model weights used for the combined score.
    lambda_whisper: float = 0.45
    lambda_gpt: float = 0.55

    # Return the weighted combination of both model scores.
    return (lambda_whisper * w_score) + (lambda_gpt * g_score)


# Convert a numeric score into a human-readable reliability label.
# The thresholds define four reliability bands:
# - very reliable
# - reliable
# - uncertain
# - unreliable
def _score_evaluation(score: float) -> str:
    if score >= -0.5:
        return "very reliable"
    elif score >= -0.7:
        return "reliable"
    elif score >= -1.2:
        return "uncertain"
    else:
        return "unreliable"


# Evaluate the reliability of a single model output using its metrics.
#
# This function is used when Whisper and GPT-4o disagree and you want
# to judge one model on its own.
#
# The score starts from avg_logprob and optionally applies:
# - a no_speech_prob penalty
# - a compression_ratio penalty
#
# In addition to the score-based thresholds, the function applies
# hard overrides:
# - if no_speech_prob > 0.8 -> immediately "unreliable"
# - if compression_ratio > 2.4 -> immediately "unreliable"
#
# These overrides reflect cases where the transcription is considered
# highly suspicious regardless of the final numeric score.
def evaluate_single_reliability(
    avg_logprob: float,
    no_speech_prob: Optional[float] = None,
    compression_ratio: Optional[float] = None,
) -> str:
    """
    Return only one of the following labels:
    "very reliable", "reliable", "uncertain", "unreliable"
    """

    # Start from the average log probability as the base score.
    score: float = avg_logprob

    # Apply a penalty if the probability of "no speech" is available.
    if no_speech_prob is not None:
        score -= 0.7 * no_speech_prob

    # Apply a penalty if the compression ratio is available and suspicious.
    if compression_ratio is not None:
        score -= 0.2 * max(0.0, compression_ratio - 1.5)

    # Strong override: if the no-speech probability is extremely high,
    # the output is considered unreliable regardless of the score.
    if no_speech_prob is not None and no_speech_prob > 0.8:
        return "unreliable"

    # Strong override: if the compression ratio is too high,
    # the transcription may be degenerate or unstable.
    if compression_ratio is not None and compression_ratio > 2.4:
        return "unreliable"

    # Score-based classification.
    if score >= -0.4:
        return "very reliable"
    elif score >= -0.7:
        return "reliable"
    elif score >= -1.2:
        return "uncertain"
    else:
        return "unreliable"


# Merge the outputs of Whisper and GPT-4o for a single file.
#
# Logic:
# 1) If both models predict the same context:
#    - combine the two scores into a single final score
#    - use that shared context as the final classification
#
# 2) If the two models disagree:
#    - compare their text_logprob values directly
#    - choose the model with the higher score
#    - report which model was selected
#
# The returned dictionary includes:
# - classification: the chosen final label
# - score: the numeric score used for the decision
# - score_meaning: a human-readable reliability label
# - selected_model: empty if both agreed, otherwise the chosen model name
def metrics(w: Dict[str, Any], g: Dict[str, Any]) -> Dict[str, Any]:
    # Case 1: both models agree on the same context label.
    if w["context"] == g["context"]:
        score: float = _final_score(w, g)
        return {
            "classification": g["context"],
            "score": score,
            "score_meaning": _score_evaluation(score),
            "selected_model": ""
        }

    # Case 2: the models disagree.
    else:
        # Extract the base scores for direct comparison.
        w_score: float = w["text_logprob"]
        g_score: float = g["text_logprob"]

        # If GPT-4o has the better score, use its classification.
        if g_score >= w_score:
            return {
                "classification": g["context"],
                "score": g_score,
                "score_meaning": evaluate_single_reliability(g_score),
                "selected_model": "gpt-4o-transcribe"
            }

        # If Whisper has the better score, use its classification
        # and evaluate reliability using Whisper-specific extra signals.
        if g_score < w_score:
            return {
                "classification": w["context"],
                "score": w_score,
                "score_meaning": evaluate_single_reliability(
                    w["text_logprob"],
                    w["no_speech_prob"],
                    w["compression_ratio"]
                ),
                "selected_model": "whisper-1"
            }


# Execute the full merge pipeline:
# - load the Whisper parquet file
# - load the GPT-4o parquet file
# - convert both DataFrames to dictionaries indexed by file path
# - merge their predictions file by file
# - save the final result as JSON
def main() -> None:
    global result

    # Load the parquet file generated from Whisper metrics.
    whisper_df: pd.DataFrame = pd.read_parquet(
        "Data/TranscriptionData/whisper/metrics/pandas/whisper.parquet"
    )

    # Load the parquet file generated from GPT-4o metrics.
    gpt4o_df: pd.DataFrame = pd.read_parquet(
        "Data/TranscriptionData/gpt-4o-transcription/metrics/pandas/gpt.parquet"
    )

    # Convert each DataFrame into a dictionary indexed by file path.
    # Each value becomes a row dictionary containing the stored metrics.
    #
    # Whisper rows are expected to contain:
    # - text_logprob
    # - compression_ratio
    # - no_speech_prob
    # - context
    whisper: Dict[str, Dict[str, Any]] = whisper_df.to_dict("index")

    # GPT-4o rows are expected to contain:
    # - text_logprob
    # - context
    gpt4o: Dict[str, Dict[str, Any]] = gpt4o_df.to_dict("index")

    # Iterate over all file keys coming from the Whisper dictionary.
    # The code assumes that the same keys also exist in the GPT-4o dictionary.
    dict_keys = whisper.keys()

    # Merge the metrics for each file and store them under a shortened key.
    for k in dict_keys:
        result[formatted_key(str(k))] = metrics(whisper[k], gpt4o[k])

        # print(formatted_key(str(k)))
        # print(result[formatted_key(str(k))])

    # Save the final merged results to a JSON file.
    with open("Data/TranscriptionData/final_classification/final_results.json", "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)


# Run the script only when executed directly.
if __name__ == "__main__":
    main()