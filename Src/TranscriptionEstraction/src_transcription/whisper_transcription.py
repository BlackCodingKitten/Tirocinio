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
    client = OpenAI(api_key=API_KEY)

    # Return the configured client.
    return client


# Create a list of media file paths.
def _create_media_file_path(media_file_path: List[str] = []) -> List[str]:
    # Iterate from 1 to 100 inclusive.
    for i in range(1, 101):
        # Append the current video file path
        # in the format: data/videos/Video1.mp4, ..., Video100.mp4
        media_file_path.append(f"Data/Videos/Video{str(i)}.mp4")

    # Return the complete list of file paths.
    return media_file_path


# Generate Whisper verbose transcriptions and save them to a JSON file.
def generate_and_save_transcription(
    client: Any = _create_openAI_Client(),
    media_file_path: List[str] = _create_media_file_path()
) -> None:
    # Initialize an empty dictionary to store transcriptions,
    # using the media file path as the key.
    formatted_transcription: Dict[str, Any] = {}

    # Iterate over each media file path in the input list.
    for media_file in media_file_path:
        try:
            # Open the current media file in binary read mode.
            with open(media_file, "rb") as video_to_transcribe:
                # Send the audio/video file to the transcription model.
                verbose_transcription: Any = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=video_to_transcribe,
                    language="it",
                    temperature=0,
                    response_format="verbose_json",
                )

            # Build a filtered transcription dictionary containing:
            # - the cleaned full transcription text
            # - the cleaned segment list with selected metrics
            filtered_transcription: Dict[str, Any] = {
                "text": verbose_transcription.text.replace(
                    "Sottotitoli creati dalla comunità Amara.org", ""
                ),
                "segments": [
                    {
                        "id": s.id,
                        "text": s.text.replace(
                            "Sottotitoli creati dalla comunità Amara.org", ""
                        ),
                        "avg_logprob": s.avg_logprob,  # Average token log probability for the segment.
                        "compression_ratio": s.compression_ratio,  # Signal of repetitive or degenerate text generation.
                        "no_speech_prob": s.no_speech_prob  # Probability that the segment does not contain speech.
                    }
                    for s in verbose_transcription.segments
                ]
            }

            # Store the filtered transcription using the media file path as the key.
            formatted_transcription[media_file] = filtered_transcription

            # Print a success message for the current file.
            print(f"OK: {media_file}")

        except Exception as e:
            # Print an error message if transcription fails for the current file.
            print(f"ERROR: file -> {media_file}: {e}")

    # Save the complete transcription dictionary to a JSON file.
    with open(
        "Data/TranscriptionData/whisper/whisper_verbose_transcription.json",
        "w",
        encoding="utf-8"
    ) as transcription_file:
        dump(formatted_transcription, transcription_file, ensure_ascii=False, indent=2)

    


# Execute the transcription pipeline.
def main() -> None:
    generate_and_save_transcription()
    


if __name__ == "__main__":
    main()