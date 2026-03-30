import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import json
from sklearn.model_selection import GridSearchCV
import librosa
import pandas as pd


labels = {
    0: "dialogue",
    1: "silence/noise",
    2: "music",
    3: "unknown_dialogue"
}

# esempio
X_train = np.array([
    [-0.3642003536224365, 0.8461538553237915, 0.9767947793006897],
    [-0.33000534772872925, 1.4509804248809814, 0.012308348901569843],
    [-0.8980435132980347, 0.4642857164144516,  0.527010440826416],
    [ -0.4469425578912099, 0.9100064635276794, 0.035722192066411175],
    [-0.22929976667676652, 8.729074570110866, 0.6763851795877729],
])

y = np.array([1, 0, 2, 0, 3])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

#multi-classe automaticamente (one-vs-one)
model = SVC(
    kernel='rbf',      # non lineare
    C=1.0,
    gamma='scale',
    probability=True
)

model.fit(X_scaled, y)


def predict(x, model, scaler):
    x_scaled = scaler.transform(x)   
    return model.predict_proba(x_scaled)[0],model.predict(x_scaled)[0]

with open("data/prompted_transcription/whisper_metrics.json", "r", encoding="utf-8") as file:
    d = json.load(file)

w_prediction = {}

for k in d.keys(): 
    x = np.array([list(d[k]["metrics"].values())])
    array,index = predict(x, model, scaler)
    w_prediction[k] = {"classification": labels[int(index)], "classification_probability": round(float(max(array)),4), "text": ""}
    if index == 0:
        w_prediction[k]["text"] = d[k]["text"]
        
        
with open("data/metrics/svm_results/whisper_SVM.txt") as file:
    df = pd.DataFrame(w_prediction.values(), index=[video.split("/")[2] for video in w_prediction.keys()])
print("Fino a Whisper OK\n\n\n\n\n")   
#Wprediction da dumpare dentro un file 



# ----------------------------------------------------------------------------------------------------


def extract_audio_features(audio_path, sr=16000, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    if len(y) == 0:
        raise ValueError("Audio vuoto")

    features = []

    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    features.extend([
        np.mean(rms),
        np.std(rms),
        np.max(rms),
        np.min(rms)
    ])

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features.extend([
        np.mean(zcr),
        np.std(zcr)
    ])

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features.extend([
        np.mean(centroid),
        np.std(centroid)
    ])

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features.extend([
        np.mean(bandwidth),
        np.std(bandwidth)
    ])

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features.extend([
        np.mean(rolloff),
        np.std(rolloff)
    ])

    # Spectral flatness
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    features.extend([
        np.mean(flatness),
        np.std(flatness)
    ])

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    for coeff in mfcc:
        features.append(np.mean(coeff))
        features.append(np.std(coeff))

    # Delta MFCC
    delta_mfcc = librosa.feature.delta(mfcc)
    for coeff in delta_mfcc:
        features.append(np.mean(coeff))
        features.append(np.std(coeff))

    # Pitch / voicedness approssimata
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )

    voiced_ratio = np.mean(~np.isnan(f0))
    valid_f0 = f0[~np.isnan(f0)]

    if len(valid_f0) > 0:
        features.extend([
            voiced_ratio,
            np.mean(valid_f0),
            np.std(valid_f0)
        ])
    else:
        features.extend([
            voiced_ratio,
            0.0,
            0.0
        ])

    return np.array(features, dtype=np.float32)

audio_paths = [ f"data/audio/Audio{i}.mp3" for i in range (1,101)]

audio_data = []
for p in audio_paths:
    audio_data.append((extract_audio_features(p)).tolist())
    print(p," OK")

g_df = pd.read_parquet("data/metrics/pandas_parquet/gpt.parquet")
gpt4o = g_df.to_dict("index")

g_logprob = [gpt4o[k]["text_logprob"] for k in gpt4o.key()]
for i in range(1,101): 
    audio_data[i].append(g_logprob[i])
    
audio_data = np.array(audio_data)
print(audio_data)

