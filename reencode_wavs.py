import os
from pydub import AudioSegment
from pydub.utils import which
from pydub import AudioSegment

AudioSegment.converter = which("ffmpeg")  # ensures Pydub uses your FFmpeg


data_dir = "data/genres"  # path to your dataset

for genre in os.listdir(data_dir):
    genre_path = os.path.join(data_dir, genre)
    print(f"Processing genre: {genre}")
    for file in os.listdir(genre_path):
        if file.endswith(".wav"):
            file_path = os.path.join(genre_path, file)
            try:
                print(f"  Re-encoding file: {file}")
                audio = AudioSegment.from_file(file_path)
                audio.export(file_path, format="wav", codec="pcm_s16le")
            except Exception as e:
                print(f"  Broken file skipped: {file_path} ({e})")

print("Re-encoding completed (broken files skipped).")
