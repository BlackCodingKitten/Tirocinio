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
        media_file_path.append(f"./Videos/Video{str(i)}.mp4")
    return media_file_path
 
ordered_transcription = {}
 
#file transcription to verbose-json anf filtered output
media_file_path = create_media_file_path()
for media_file in media_file_path:
    try:
        with open(media_file, "rb") as video_to_transcribe: 
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe", 
                file=video_to_transcribe,
                language= "it",
                temperature=0,
                response_format="json",
                include=["logprobs"]    #è il logaritmo della probabilità che il modello assegna a quel token in quella precisa posizione, dato tutto il contesto precedente
               
            )
        formatted_transcription = {
            "text" :transcription.text,
            "segments": [
                {
                "token": token.token,
                "logprob": token.logprob
                }
                for token in transcription.logprobs
            ]
        }
        ordered_transcription[media_file] = formatted_transcription
        print(formatted_transcription)
        print(f"OK: {media_file}")
    except Exception as e:
        print(f"ERROR: file -> {media_file}: {e}") 



with open("gpt4otranscribeVerboseTranscription.json", "w", encoding="utf-8") as transcription_file:
    json.dump(ordered_transcription, transcription_file, ensure_ascii=False, indent=2)