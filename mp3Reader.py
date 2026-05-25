import json
import os

import librosa
import numpy as np
import time
import shutil

PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                 'F#', 'G', 'G#', 'A', 'A#', 'B']

def build_major_templates():
    major = []

    for i in range(12):
        m = np.zeros(12)
        m[i] = 1
        m[(i + 4) % 12] = 1
        m[(i + 7) % 12] = 1
        major.append(m)

    return major


def build_minor_templates():
    minor = []

    for i in range(12):
        m = np.zeros(12)
        m[i] = 1
        m[(i + 3) % 12] = 1
        m[(i + 7) % 12] = 1
        minor.append(m)

    return minor


MAJOR_TEMPLATES = build_major_templates()
MINOR_TEMPLATES = build_minor_templates()


def _predict_chord(chroma_vec):
    best_score = -1
    best_label = "N/A"

    for i, root in enumerate(PITCH_CLASSES):
        maj_score = np.dot(chroma_vec, MAJOR_TEMPLATES[i])
        min_score = np.dot(chroma_vec, MINOR_TEMPLATES[i])

        if maj_score > best_score:
            best_score = maj_score
            best_label = root

        if min_score > best_score:
            best_score = min_score
            best_label = root + "m"

    return best_label


def _predict_note(chroma_vec):
    idx = np.argmax(chroma_vec)
    return PITCH_CLASSES[idx]


def detect_key(y, sr):
    try:
        y_harmonic, _ = librosa.effects.hpss(y)

        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        key_index = np.argmax(chroma_mean)
        return PITCH_CLASSES[key_index]

    except Exception as e:
        return f"Error processing file: {e}"


def detect_tempo(y, sr):
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    return tempo.item(), beat_times.tolist()

def preprocess_audio(y):
    y, _ = librosa.effects.trim(y)         # remove silence
    y = librosa.util.normalize(y)          # normalize volume
    y = librosa.effects.preemphasis(y)     # reduce low-frequency noise
    return y

def analyze_beats(y, sr, beat_times, threshold=2):
    y = preprocess_audio(y)
    y_harmonic, _ = librosa.effects.hpss(y)

    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    times = librosa.times_like(chroma, sr=sr)

    results = []

    for bt in beat_times:
        idx = np.argmin(np.abs(times - bt))
        vec = chroma[:, idx]

        active = np.sum(vec > 0.3)

        label_type = "note" if active <= 1 else "chord"

        prediction = None

        if label_type == "note":
            prediction = _predict_note(vec)
        else:
            prediction = _predict_chord(vec)

        results.append({
            "beat_time": float(bt),
            "type": label_type,
            "prediction": prediction,
            "active_pitches": int(active),
            "chroma": vec.tolist()
        })

    return results


def detect_notes(y, sr, beat_times):
    """
    Only used when label == 'note'
    """

    notes = []

    for bt in beat_times:
        f0 = librosa.yin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )

        times = librosa.times_like(f0, sr=sr)
        idx = np.argmin(np.abs(times - bt))

        pitch_hz = f0[idx]
        note = librosa.hz_to_note(pitch_hz)

        notes.append(note)

    return notes


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def main(audio_path):
    y, sr = librosa.load(audio_path)

    key = detect_key(y, sr)
    tempo, beat_times = detect_tempo(y, sr)

    beat_analysis = analyze_beats(y, sr, beat_times)
    

    song_data = {
        "key": key,
        "tempo": tempo,
        "beats": beat_analysis
    }

    with open("readMP3Results/song_data.json", "w", encoding="utf-8") as f:
        json.dump(song_data, f, indent=4, ensure_ascii=False)


# ----------------------------
# Listener and executor
# ----------------------------
INPUT_DIR = "mp3ReaderInput"
PROCESSED_DIR = "processedMP3s"

# Create processed directory if missing
os.makedirs(PROCESSED_DIR, exist_ok=True)

while True:
    inputs = os.listdir(INPUT_DIR)

    for file_name in inputs:

        if file_name.endswith(".mp3"):

            input_path = os.path.join(INPUT_DIR, file_name)

            try:
                # Process file
                main(input_path)

                print("translated:", input_path)

                # Move file after successful processing
                processed_path = os.path.join(PROCESSED_DIR, file_name)

                shutil.move(input_path, processed_path)

                print("moved to:", processed_path)

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    time.sleep(1)
    