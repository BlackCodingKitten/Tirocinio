import pandas as pd
from sklearn.metrics import cohen_kappa_score
import json
import krippendorff

to_save = ""

with open ("data/result_merge/final_results.json", "r", encoding="utf-8") as file :
    merge = json.load(file)

whisper_df = pd.read_parquet('data/metrics/pandas_parquet/whisper.parquet')
gpt4o_df = pd.read_parquet('data/metrics/pandas_parquet/gpt.parquet')
    
whisper = whisper_df.to_dict("index") 
gpt4o = gpt4o_df.to_dict("index")

def classifier_vector (d):
    a = []
    for k in d.keys():
        if d[k]["context"] == "music":
            a.append(2)
        elif d[k]["context"] == "silence/noise":
            a.append(1)            
        elif d[k]["context"]== "unknown_dialogue":
            a.append(3)            
        else:
        # dialogo
            a.append(0)
    return a
w = classifier_vector(whisper)
g = classifier_vector(gpt4o)

to_save += f"Indicazioni:\n Il modello classifica l'audio come Dialogo\nIl modello classifica l'audio come Musica\nIl modello classifica l'audio come rumore o silenzio\nIl modello percepisce del dialogo, ma non riesce a trascriverlo\n"
to_save += "\n Di seguito verranno riportate alcune metriche misurabili relative alla classificazione eseguita sui file, "
data = [{
    "whisper": w[i],
    "gpt" : g[i]
}for i in range(0,100)]

df=pd.DataFrame(data, index=[f"Video{i}" for i in range(1,101)])

s=0
for i in range(0,100):
    if w[i]==g[i]:
        s+=1


alpha = krippendorff.alpha(reliability_data=[w,g], level_of_measurement='nominal')
kappa = round(cohen_kappa_score(w,g),5)

# for i in range (1,101) :
    
# print(f"i due modelli vanno d'accordo al {s}%, il loro Agreement, calcolato con k do Cohen è {round(cohen_kappa_score(w,g),5)}, il k di Kippendhorf è {alpha}")
            