import json
import pandas as pd

result = {}

whisper_df = pd.read_parquet('data/metrics/pandas_parquet/whisper.parquet')
gpt4o_df = pd.read_parquet('data/metrics/pandas_parquet/gpt.parquet')
    
whisper = whisper_df.to_dict("index") #text_logprob  compression_ratio  no_speech_prob  context
gpt4o = gpt4o_df.to_dict("index") #"text_logprob"  context

dict_keys = whisper.keys()

def formatted_key (key):
    return str(key).split("/")[2].strip()
    
def _score_whisper(avg_lp, avg_cr, avg_nsp):
    # avg_logprob−0.7⋅no_speech_prob−0.2⋅max(0,compression_ratio−1.5)
    
    lambda1 = 0.7
    # Perché coefficiente 0.7
    # no_speech_prob ∈ [0,1]
    # avg_logprob ∈ [-2, 0] tipicamente
    # Quindi:
    # una penalità di 0.7 può abbassare il punteggio fino a ~0.7
    # che è comparabile alla variazione reale delle logprob
    # Se lo mettessi:
    # troppo basso → ignorato
    # troppo alto → annulla completamente la logprob
   
    lambda2 = 0.25
    # Perché coefficiente 0.25
    # è una penalità secondaria
    # non deve dominare la logprob
    # serve solo a “rompere i pareggi” nei casi degenerati
    
    # il max:
    # Vuoi penalizzare solo quando è anomalo
    # < 1.5 → normale → nessuna penalità
    # 1.5 → inizia sospetto
    # 2 → molto sospetto
    return avg_lp -(lambda1*avg_nsp) -(lambda2 * (max(0, (avg_cr-1.5))))

def _final_score(w,g): 
    w_score = _score_whisper(w["text_logprob"],w["compression_ratio"],w["no_speech_prob"])
    g_score = g["text_logprob"]
    # Perché pesi lambda_whisper = 0.45 / lambda_gpt = 0.55

    # Leggero bias verso GPT-4o perché:

    # in generale:
    # migliore modellazione linguistica
    # meno degenerazioni rispetto a Whisper
    # Whisper invece:
    # ha feature extra (no_speech_prob)
    # ma è più soggetto a errori fonetici o ripetizioni
    lambda_whisper = 0.45
    lambda_gpt = 0.55
    return (lambda_whisper * w_score) + (lambda_gpt * g_score)
    
def _score_evaluation(score):
    if score >= -0.5:
        return "very reliable"
    elif score >= -0.7:
        return "reliable"
    elif score >= -1.2 :
        return "uncertain"
    else: 
        return "unreliable"
     
def evaluate_single_reliability(
    avg_logprob,
    no_speech_prob=None,
    compression_ratio=None,
):
    """
    Restituisce solo:
    "very reliable", "reliable", "uncertain", "unreliable"
    """

    score = avg_logprob

    if no_speech_prob is not None:
        score -= 0.7 * no_speech_prob

    if compression_ratio is not None:
        score -= 0.2 * max(0.0, compression_ratio - 1.5)

    # penalità forti (override)
    if no_speech_prob is not None and no_speech_prob > 0.8:
        return "unreliable"

    if compression_ratio is not None and compression_ratio > 2.4:
        return "unreliable"

    # classificazione per score
    if score >= -0.4:
        return "very reliable"
    elif score >= -0.7:
        return "reliable"
    elif score >= -1.2:
        return "uncertain"
    else:
        return "unreliable"
    
def metrics(w,g):
    if w["context"] == g["context"]:
        score = _final_score(w,g)
        return {"classification":g["context"], "score":score, "score_meaning": _score_evaluation(score), "selected_model":""}
    else: 
        w_score = w["text_logprob"]
        g_score = g["text_logprob"]
        if g_score >= w_score:
            return {"classification":g["context"], "score":g_score, "score_meaning":evaluate_single_reliability(g_score), "selected_model":"gpt-4o-transcribe"}
        if g_score < w_score:  
            return {"classification":w["context"], "score":w_score, "score_meaning":evaluate_single_reliability(w["text_logprob"],w["no_speech_prob"],w["compression_ratio"]), "selected_model":"whisper-1"}
        
        
        
for k in dict_keys:
    result[formatted_key(str(k))]=metrics(whisper[k],gpt4o[k])
    
    # print(formatted_key(str(k)))
    # print(result[formatted_key(str(k))])
with open("data/result_merge/final_results.json", "w", encoding="utf-8")as file:
     json.dump(result, file, ensure_ascii=False, indent=2)