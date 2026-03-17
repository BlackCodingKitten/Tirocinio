from openai import OpenAI

with open("../ApiKey/key.txt") as f:
    API_KEY = f.read().strip()

client = OpenAI(api_key=API_KEY)

media_file_path = "./Videos/Video13.mp4"

with open(media_file_path, "rb") as media_file:
    response = client.audio.transcriptions.create(
        model = "whisper-1",
        file = media_file,  
        prompt = """NON INVENTARE ASSOLUTAMENTE IL TESTO, usa la parola più proobabile e in italiano corretto metti la punteggiatura coerente.""",
        language = "it"
    )

print(response.text)


