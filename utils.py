
import os
import numpy as np
import librosa


def load_audio(path, sr=22050, mono=True, duration=30):
    y, _ = librosa.load(path, sr=sr, mono=mono, duration=duration)
    return y

def extract_mel_spectrogram(y, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def pad_or_truncate(y, sr=22050, duration=30):
    target_len = sr * duration
    if len(y) > target_len:
        return y[:target_len]
    elif len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    else:
        return y

def load_genre_folder(folder_path, sr=22050, duration=30, n_mels=128):
    X = []
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
    for f in files:
        p = os.path.join(folder_path, f)
        y = load_audio(p, sr=sr, duration=duration)
        y = pad_or_truncate(y, sr, duration)
        S = extract_mel_spectrogram(y, sr=sr, n_mels=n_mels)
        X.append(S)
    return np.array(X)
