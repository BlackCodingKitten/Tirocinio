from __future__ import annotations
import json
import pandas as pd
from typing import Dict, Any, Optional
import re
from functools import lru_cache
from typing import List, Tuple
import spacy
from sentence_transformers import SentenceTransformer, util
from spacy.language import Language

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

def _normalize_whitespace(text: str) -> str:
    """
    Collapse repeated whitespace and trim the text.
    """
    return re.sub(r"\s+", " ", text).strip()


def _fix_punctuation_spacing(text: str) -> str:
    """
    Fix spacing around punctuation and capitalize the first character.
    """
    text = _normalize_whitespace(text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", text)
    text = _normalize_whitespace(text)

    if text:
        text = text[0].upper() + text[1:]

    return text


def _sentencize(nlp: Language, text: str) -> List[str]:
    """
    Split text into sentences with spaCy.

    If spaCy returns no sentence boundaries, fallback to the whole text.
    """
    doc = nlp(text)
    sentences = [_normalize_whitespace(sent.text) for sent in doc.sents if sent.text.strip()]
    return sentences if sentences else [_normalize_whitespace(text)]


def _text_quality(text: str, nlp: Language) -> Tuple[int, int, int]:
    """
    Compute a lightweight quality tuple for tie-breaking.

    The tuple is ordered so that a lexicographic comparison prefers:
    1. more detected sentences,
    2. more punctuation tokens,
    3. more alphabetic tokens.

    This is intentionally minimal: the semantic work is delegated to NLP models.
    """
    doc = nlp(text)

    sentence_count = sum(1 for _ in doc.sents)
    punctuation_count = sum(1 for token in doc if token.is_punct)
    alpha_count = sum(1 for token in doc if token.is_alpha)

    return (sentence_count, punctuation_count, alpha_count)


def _choose_better_variant(first: str, second: str, nlp: Language) -> str:
    """
    Choose the better textual variant when both inputs say nearly the same thing.
    """
    first_score = _text_quality(first, nlp)
    second_score = _text_quality(second, nlp)

    if first_score > second_score:
        return first
    if second_score > first_score:
        return second

    return first if len(first) >= len(second) else second


@lru_cache(maxsize=1)
def _load_models() -> Tuple[Language, SentenceTransformer]:
    """
    Load and cache the NLP models.

    Requirements:
    - spaCy Italian model installed, e.g. it_core_news_md
    - sentence-transformers model available locally or downloadable
    """
    nlp = spacy.load("it_core_news_md")
    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return nlp, embedder

def combine_italian_transcriptions(
    first: str,
    second: str,
    duplicate_threshold: float = 0.84,
    sentence_duplicate_threshold: float = 0.82,
) -> str:
    """
    Combine two Italian transcription variants into a single coherent text.

    This version relies mostly on NLP libraries:
    - spaCy for Italian sentence segmentation and token analysis,
    - Sentence Transformers for semantic similarity.

    Behavior:
    - If the two full transcriptions are semantically very close, keep the better one.
    - Otherwise, merge sentence-level content while removing semantically redundant parts.

    Args:
        first: First transcription.
        second: Second transcription.
        duplicate_threshold: Similarity threshold above which the two full texts
            are treated as near-duplicates.
        sentence_duplicate_threshold: Similarity threshold above which two sentences
            are treated as semantically redundant.

    Returns:
        A combined Italian text.
    """
    first = _normalize_whitespace(first)
    second = _normalize_whitespace(second)

    # Early exit if both transcriptions are empty.
    if first == "" and second == "":
        return ""

    if not first:
        return _fix_punctuation_spacing(second)
    if not second:
        return _fix_punctuation_spacing(first)

    nlp, embedder = _load_models()

    # Step 1: Compare the two complete transcriptions semantically.
    text_embeddings = embedder.encode(
        [first, second],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    full_similarity = float(util.cos_sim(text_embeddings[0], text_embeddings[1]))

    # If both texts are basically two versions of the same utterance,
    # return only the better textual variant.
    if full_similarity >= duplicate_threshold:
        best = _choose_better_variant(first, second, nlp)
        return _fix_punctuation_spacing(best)

    # Step 2: Split both texts into sentences.
    first_sentences = _sentencize(nlp, first)
    second_sentences = _sentencize(nlp, second)

    merged_sentences: List[str] = list(first_sentences)

    if merged_sentences:
        merged_embeddings = embedder.encode(
            merged_sentences,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
    else:
        merged_embeddings = None

    for candidate in second_sentences:
        candidate_embedding = embedder.encode(
            [candidate],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )[0]

        if merged_embeddings is None or len(merged_sentences) == 0:
            merged_sentences.append(candidate)
            merged_embeddings = candidate_embedding.unsqueeze(0)
            continue

        similarities = util.cos_sim(candidate_embedding, merged_embeddings)[0]
        best_idx = int(similarities.argmax().item())
        best_similarity = float(similarities[best_idx].item())

        if best_similarity < sentence_duplicate_threshold:
            merged_sentences.append(candidate)
            merged_embeddings = embedder.encode(
                merged_sentences,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            continue

        current = merged_sentences[best_idx]
        replacement = _choose_better_variant(current, candidate, nlp)
        merged_sentences[best_idx] = replacement
        merged_embeddings = embedder.encode(
            merged_sentences,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

    combined = " ".join(merged_sentences)
    return _fix_punctuation_spacing(combined)

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
        return "transcription text unreliable"

    # Strong override: if the compression ratio is too high,
    # the transcription may be degenerate or unstable.
    if compression_ratio is not None and compression_ratio > 2.4:
        return "transcription text unreliable"

    # Score-based classification.
    if score >= -0.3:
        return "transcription text very reliable"
    elif score >= -0.6:
        return "transcription text reliable"
    elif score >= -0.9:
        return "transcription text uncertain"
    else:
        return "transcription text unreliable"


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
        if g["context"] == 'unknown_dialogue' or g["context"] == 'dialogue':
            return {
                "classification": g["context"],
                "score": score,
                "score_meaning": _score_evaluation(score),
                "selected_model": "both",
                "generated_transcription": combine_italian_transcriptions(w["transcription"],g["transcription"])
            }
    
        return {
            "classification": g["context"],
            "selected_model": "both"
        }

    # Case 2: the models disagree.
    else:
        # Extract the base scores for direct comparison.
        w_score: float = w["text_logprob"]
        g_score: float = g["text_logprob"]

        # If GPT-4o has the better score, use its classification.
        if g_score >= w_score:
            if g["context"] == 'unknown_dialogue' or g["context"] == 'dialogue':
                return {
                    "classification": g["context"],
                    "score": g_score,
                    "score_meaning": evaluate_single_reliability(g_score),
                    "selected_model": "gpt-4o-transcribe",
                    "generated_transcription": g["transcription"]
                }
              
                return {
                    "classification": g["context"],
                    "selected_model": "gpt-4o-transcribe"
                }


        # If Whisper has the better score, use its classification
        # and evaluate reliability using Whisper-specific extra signals.
        if g_score < w_score:
            if w["context"] == 'unknown_dialogue' or w["context"] == 'dialogue':
                return {
                    "classification": w["context"],
                    "score": w_score,
                    "score_meaning": evaluate_single_reliability(
                        w["text_logprob"],
                        w["no_speech_prob"],
                        w["compression_ratio"]
                    ),
                    "selected_model": "whisper-1",
                    "generated_transcription": w["transcription"]
                }
            return {
                    "classification": w["context"],
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