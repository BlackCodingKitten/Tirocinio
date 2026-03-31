from openai import OpenAI
from typing import List, Any, Dict
from json import dump

# Create and return an OpenAI client using the API key read from a file.
def _create_openAI_Client(key: str = "../ApiKey/key.txt") -> OpenAI:
    # Open the file containing the API key.
    with open(key) as f:
        # Read the key, removing leading and trailing whitespace.
        API_KEY = f.read().strip()

    # Create the OpenAI client with the loaded API key.
    client = OpenAI(api_key = API_KEY)

    # Return the configured client.
    return client

# Create a list of media file paths.
def _create_media_file_path(media_file_path: List[str] = []) -> List[str]:
    # Iterate from 1 to 100 inclusive.
    for i in range(1, 101):
        # Append the current video file path
        # in the format: data/videos/Video1.mp4, ..., Video100.mp4
        media_file_path.append(f"Data/Videos/Video{i}.mp4")
    
    # Return the complete list of file paths.
    return media_file_path
 
# Iterate over each media file path in the input list.
def generate_and_save_transcription(
    client: Any = _create_openAI_Client(),
    media_file_path: List[str] = _create_media_file_path()
) -> None:
    # Initialize an empty dictionary to store transcriptions,
    # using the media file path as the key.
    ordered_transcription: Dict[str, Any] = {}

    # Iterate over each media file path in the input list.
    for media_file in media_file_path:
        try:
            # Open the current media file in binary read mode.
            with open(media_file, "rb") as video_to_transcribe:
                # Send the audio/video file to the transcription model.
                transcription: Any = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=video_to_transcribe,
                    language="it",
                    temperature=0,
                    prompt='se il video contiene della musica, scrivi solo "music", se il video contiene del rumore , non inventare le parole ma scrivimi "silence/noise", se il video contiene un dialogo che non riesci a capire scrivi "unknown_dialogue"',
                    response_format="json",
                    include=["logprobs"]  # Logarithm of the probability assigned by the model to each token.
                )

            # Build a simplified dictionary containing:
            # - the transcribed text
            # - the list of token log probabilities
            formatted_transcription: Dict[str, Any] = {
                "text": transcription.text,
                "segments": [
                    {
                        "logprob": token.logprob
                    }
                    for token in transcription.logprobs
                ]
            }

            # Store the formatted transcription using the media file path as the key.
            ordered_transcription[media_file] = formatted_transcription

            # Print the formatted transcription and a success message.
            print(formatted_transcription)
            print(f"OK: {media_file}")

        except Exception as e:
            # Print an error message if transcription fails for the current file.
            print(f"ERROR: file -> {media_file}: {e}")



     # Save the complete transcription dictionary to a JSON file.
    with open(
        "Data/TranscriptionData/gpt-4o-transcription/gpt-4o-verbose_promped_transcription.json",
        "w",
        encoding="utf-8"
    ) as transcription_file:
        dump(ordered_transcription, transcription_file, ensure_ascii=False, indent=2)
    return
    

def main() -> None:
    generate_and_save_transcription()
    return 
   
if __name__ == "__main__":
    main()