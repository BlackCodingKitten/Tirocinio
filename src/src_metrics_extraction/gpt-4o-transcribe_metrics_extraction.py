import json
import pandas as pd

metrics_dict ={}
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
#   },
def _metrics_extraction(video_entry):
    lp = 0
    i = 0
    for logprob in video_entry["segments"]:
        lp += logprob["logprob"]
        i += 1
    return (lp/i)

def mean_logprob(metrics_dict):
    for k in metrics_dict.keys():
        metrics_dict[k]["avg_logprob"] = _metrics_extraction(metrics_dict[k])
        del metrics_dict[k]["segments"]
    return metrics_dict
    
with open("data/prompted_transcription/gpt-4o-verbose_promped_transcription.json", "r", encoding="utf-8") as verbose_gpt:
    metrics_dict = mean_logprob(json.load(verbose_gpt))

data = [{
    "text_logprob": entry["avg_logprob"],
    "context": entry["text"].strip()  
} for entry in metrics_dict.values()]
df = pd.DataFrame(data, index=metrics_dict.keys())
# df.style.set_properties(**{'text-align': 'left'})
with open("data/metrics/to_show/flagged_gpt-4o_metrics.txt", "w", encoding="utf-8") as save:
    save.write(df.to_string())    
df.to_parquet('data/metrics/pandas_parquet/gpt.parquet', engine='pyarrow', compression='snappy')
