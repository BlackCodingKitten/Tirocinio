import json
import pandas as pd
from typing import Dict, Any

# Initialize the dictionary that will store the processed metrics.
metrics_dict: Dict[str, Any] = {}
# Example structure:
# {
#   "data/videos/Video1.mp4": {
#     "text": "unknown_dialogue",
#     "segments": [
#       {
#         "logprob": -0.6838496923446655
#       },
#       {
#         "logprob": -0.0020215471740812063
#       }
#     ]
#   }
# }


# Compute the mean log probability across all segment logprob values.
def _metrics_extraction(video_entry: Dict[str, Any]) -> float:
    lp = 0
    i = 0

    # Sum all logprob values and count the segments.
    for logprob in video_entry["segments"]:
        lp += logprob["logprob"]
        i += 1

    # Return the average logprob.
    return (lp / i)


# Add the average log probability to each entry
# and remove the original segments list.
def mean_logprob(metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
    for k in metrics_dict.keys():
        metrics_dict[k]["avg_logprob"] = _metrics_extraction(metrics_dict[k])
        del metrics_dict[k]["segments"]
    return metrics_dict


# Execute the full processing pipeline:
# load the JSON file, compute average logprob values,
# create a DataFrame, save a text export, and save a parquet file.
def main() -> None:
    global metrics_dict

    # Load the GPT-4o prompted verbose transcription file
    # and compute the average log probability for each entry.
    with open(
        "Data/TranscriptionData/gpt-4o-transcription/gpt-4o-verbose_promped_transcription.json",
        "r",
        encoding="utf-8"
    ) as verbose_gpt:
        metrics_dict = mean_logprob(json.load(verbose_gpt))

    # Build a list of dictionaries for DataFrame creation,
    # including the average logprob and the text-based context label.
    
    data = [
        {
            "text_logprob": entry["avg_logprob"],
            "context": entry["text"].strip()
        }
        for entry in metrics_dict.values()
    ]
    
    for e in data:
        if e["context"] == "music" or e["context"] == "silence/noise" or e["context"] == "unknown_dialogue":
            e["transcription"] = ""
        else:
            e["transcription"] = e ["context"]
            e["context"] = "dialogue"

    # Create the DataFrame using media file paths as index.
    df = pd.DataFrame(data, index=metrics_dict.keys())

    # df.style.set_properties(**{'text-align': 'left'})

    # Save the DataFrame as a readable text file.
    with open("Data/TranscriptionData/gpt-4o-transcription/metrics/pandas/pandas_gpt-4o_metrics.txt", "w", encoding="utf-8") as save:
        save.write(df.to_string())

    # Save the DataFrame as a parquet file.
    df.to_parquet(
        "Data/TranscriptionData/gpt-4o-transcription/metrics/pandas/gpt.parquet",
        engine="pyarrow",
        compression="snappy"
    )


# Run the script only when executed directly.
if __name__ == "__main__":
    main()