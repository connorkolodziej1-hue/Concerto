
import librosa
import numpy as np

def detect_key(y, sr):
    try:
        # Compute chroma features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

        # Average chroma over time
        chroma_mean = np.mean(chroma, axis=1)

        # Define pitch classes
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F',
                         'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Find the most prominent pitch
        key_index = np.argmax(chroma_mean)
        return pitch_classes[key_index]

    except Exception as e:
        return f"Error processing file: {e}"

def detect_tempo(y, sr):

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    return tempo, beat_times

def detect_notes(beat_times, y, sr):
    f0 = librosa.yin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )

    times = librosa.times_like(f0, sr=sr)

    notes = []
    for beat_time in beat_times:
        closest_index = np.argmin(np.abs(times - beat_time))
        pitch_hz = f0[closest_index]
        notes.append(librosa.hz_to_note(pitch_hz))

    return notes
        
def main(audio_path):
    # Load audio
    y, sr = librosa.load(audio_path)
    
    key = detect_key(y, sr)
    tempo, beat_times = detect_tempo(y, sr)
    
    notes = detect_notes(beat_times, y, sr)
    
    
    