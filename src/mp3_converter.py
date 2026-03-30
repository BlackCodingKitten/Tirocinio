import subprocess
from pathlib import Path

def mp4_to_mp3(input_path, output_path=None, bitrate="192k"):
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.with_suffix(".mp3")
    else:
        output_path = Path(output_path)
    
    cmd = [
        "ffmpeg",
        "-y",                    # sovrascrive se esiste
        "-i", str(input_path),
        "-vn",                   # niente video
        "-c:a", "libmp3lame",    # codec mp3
        "-b:a", bitrate,         # bitrate
        str(output_path)
    ]
    
    subprocess.run(cmd, check=True)
    return output_path


# uso
for i in range(1,101):
    ipath = f"data/videos/Video{i}.mp4"
    opath = f"data/audio/Audio{i}.mp3"
    mp4_to_mp3(ipath, opath)