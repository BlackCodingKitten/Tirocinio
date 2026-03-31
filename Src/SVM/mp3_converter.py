import subprocess
from pathlib import Path
from typing import Optional, Union


# Convert an MP4 file to MP3 using ffmpeg.
def mp4_to_mp3(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    bitrate: str = "192k"
) -> Path:
    # Convert the input path to a Path object.
    input_path = Path(input_path)

    # If no output path is provided, use the same file name
    # as the input file but with the .mp3 extension.
    if output_path is None:
        output_path = input_path.with_suffix(".mp3")
    else:
        # Convert the output path to a Path object.
        output_path = Path(output_path)

    # Build the ffmpeg command:
    # - overwrite the output file if it already exists
    # - read the input file
    # - disable video stream processing
    # - encode audio with the MP3 codec
    # - set the target bitrate
    cmd = [
        "ffmpeg",
        "-y",                 # Overwrite the output file if it already exists.
        "-i", str(input_path),
        "-vn",                # Disable video in the output.
        "-c:a", "libmp3lame", # Use the MP3 audio codec.
        "-b:a", bitrate,      # Set the audio bitrate.
        str(output_path)
    ]

    # Execute the ffmpeg command and raise an exception if it fails.
    subprocess.run(cmd, check=True)

    # Return the output MP3 file path.
    return output_path


# Execute the conversion pipeline for files Video1.mp4 to Video100.mp4.
def main() -> None:
    # Iterate over all expected input video files.
    for i in range(1, 101):
        # Build the input and output file paths.
        ipath: str = f"Data/Videos/Video{i}.mp4"
        opath: str = f"Data/Audios/Audio{i}.mp3"

        # Convert the current MP4 file to MP3.
        mp4_to_mp3(ipath, opath)


# Run the script only when executed directly.
if __name__ == "__main__":
    main()