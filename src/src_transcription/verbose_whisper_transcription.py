from openai import OpenAI
import json


#OpenAi Client setup
#API_KEY from file
with open("../ApiKey/key.txt") as f:
    API_KEY = f.read().strip()

client = OpenAI(api_key=API_KEY)

# #Create media_file_path list: 
def create_media_file_path(media_file_path = []):
    for i in range(1,101):
        media_file_path.append(f"data/videos/Video{str(i)}.mp4")
    return media_file_path
 
formatted_transcription = {}
 
#file transcription to verbose-json anf filtered output
media_file_path = create_media_file_path()
for media_file in media_file_path:
    try:
        with open(media_file, "rb") as video_to_transcribe: 
            verbose_transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=video_to_transcribe,
                language= "it",
                temperature=0,
                response_format="verbose_json",
               
            )
        
        filtered_transcription = {
            "text": verbose_transcription.text.replace("Sottotitoli creati dalla comunità Amara.org", ""),
            "segments": [
                {  
                "id":s.id,
                "text": s.text.replace("Sottotitoli creati dalla comunità Amara.org", ""),
                "avg_logprob": s.avg_logprob,   #È la media dei log-probability dei token del segmento. In pratica misura quanto il modello fosse “convinto” delle parole che ha generato.
                "compression_ratio": s.compression_ratio, #È il rapporto di compressione del testo del segmento. Serve come segnale di anomalie: quando il testo è molto ripetitivo o “degenerato”, questo valore tende a salire. Nelle definizioni OpenAI è indicato che se supera 2.4, la generazione del segmento è da considerare problematica.
                "no_speech_prob": s.no_speech_prob    #È la probabilità che in quel segmento non ci sia parlato. Nelle definizioni riportate dai client OpenAI, il segmento può essere trattato come silenzioso quando no_speech_prob è alto e contemporaneamente avg_logprob è basso; lì compare la regola combinata usata per filtrare segmenti dubbi.
                }
                for s in verbose_transcription.segments
            ]
        }
        formatted_transcription[media_file] = filtered_transcription
        print(f"OK: {media_file}")
    except Exception as e:
        print(f"ERROR: file -> {media_file}: {e}") 



with open("data/raw_transcription/whisper_verbose_transcription.json", "w", encoding="utf-8") as transcription_file:
    json.dump(formatted_transcription, transcription_file, ensure_ascii=False, indent=2)