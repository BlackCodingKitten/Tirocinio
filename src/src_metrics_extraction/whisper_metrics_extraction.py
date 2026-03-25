import json
import pandas as pd
import re
from collections import Counter
from typing import Dict, Any, List

whisper_dict={}
   
def _segment_avg (video_entry: Dict)-> Dict:
    lp = 0
    cr = 0
    nsp = 0
    id = 0
    for segment in video_entry["segments"]:
        id = segment["id"]
        lp += segment["avg_logprob"]
        cr += segment["compression_ratio"]
        nsp += segment["no_speech_prob"]
    # DEBUG:print(id+1)
    return {"text_logprob": lp/(id+1), "text_compression_ratio": cr/(id+1), "text_no_speech_prob": nsp/(id+1)}

def whisper_performance_calculator(whisper_dict: Dict) -> Dict: 
    for path in whisper_dict.keys(): 
        whisper_dict[path]["metrics"]=_segment_avg(whisper_dict[path])
    return whisper_dict

def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", _normalize_text(text), flags=re.UNICODE)

def _repetition_score(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    c = Counter(tokens)
    return c.most_common(1)[0][1] / len(tokens)

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

    if any(re.search(p, t) for p in patterns):
        return True

    if len(tokens) >= 2 and _repetition_score(tokens) >= 0.6:
        return True

    if len(tokens) in (1, 2, 3) and len(set(tokens)) == 1:
        return True

    return False

def _classify_text(whisper: Dict[str, Any]) -> str:
    text = _normalize_text(whisper["text"])
    avg_logprob = float(whisper["metrics"]["text_logprob"])
    compression_ratio = float(whisper["metrics"]["text_compression_ratio"])
    no_speech_prob = float(whisper["metrics"]["text_no_speech_prob"])

    tokens = _tokenize(text)
    repetition = _repetition_score(tokens) if tokens else 0
    music_text = _looks_like_music(text)

    # 1) Silenzio / rumore
    if not text or (avg_logprob < -1 and no_speech_prob >= 0.5):
        return "silence/noise"

    # 2) Musica (euristico)
    if( music_text and avg_logprob < -0.75) or (repetition >= 0.6 and avg_logprob < -0.7):
        return "music"

    # 3) Dialogo
    if (
        avg_logprob > -0.6
        and compression_ratio < 2.4
        and no_speech_prob < 0.5
        and len(tokens) >= 2
    ) or (avg_logprob > -1 and compression_ratio < 2.4 and not music_text):
        return whisper["text"]

    # fallback
    return "unknown dialogue"

def context_flagger(whisper_dict: Dict) -> Dict: 
    for path in whisper_dict.keys(): 
        whisper_dict[path]["context"]=_classify_text(whisper_dict[path])
    return whisper_dict



   
with open("./data/raw_transcription/whisper_verbose_transcription.json", "r", encoding="utf-8") as whisper_file:
    whisper_dict = context_flagger(whisper_performance_calculator(json.load(whisper_file)))

# whisper_dict = dict(sorted(whisper_dict.items(), key=lambda x: x[1]["context_type"]))
data = [{"text_logprob":video_entry["metrics"]["text_logprob"], "compression_ratio":video_entry["metrics"]["text_compression_ratio"],"no_speech_prob":video_entry["metrics"]["text_no_speech_prob"], "context":video_entry["context"]} for video_entry in whisper_dict.values()]
df= pd.DataFrame(data, index=whisper_dict.keys())
# text_data =[{ "type":video_entry["context"]}for video_entry in whisper_dict.values()]
# df2 = pd.DataFrame( text_data, index=whisper_dict.keys())
with open("./data/metrics/to_show/flagged_whisper_metrics.txt", "w", encoding ="utf-8") as file:
     file.write(df.to_string())
    #  file.write("\n"*3)
    #  file.write(df2.to_string())
df.to_parquet('data/metrics/pandas_parquet/whisper.parquet', engine='pyarrow', compression='snappy')
for d in whisper_dict.values():
    del d["segments"]
# print(whisper_dict)
with open("./data/prompted_transcription/whisper_metrics.json", "w", encoding="utf-8") as save:
    json.dump(whisper_dict, save, ensure_ascii=False, indent=2)

