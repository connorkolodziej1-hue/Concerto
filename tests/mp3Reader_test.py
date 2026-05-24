import numpy as np
import pytest
import librosa
import mp3Reader


def test_detect_key_returns_valid_pitch_class():
    sr = 22050
    y = np.random.randn(sr)  # 1 second of fake audio

    key = mp3Reader.detect_key(y, sr)

    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F',
                     'F#', 'G', 'G#', 'A', 'A#', 'B']

    assert key in pitch_classes

def test_detect_tempo_returns_tuple():
    sr = 22050
    y = np.random.randn(sr)

    tempo, beat_times = mp3Reader.detect_tempo(y, sr)

    assert isinstance(tempo, (float, np.ndarray))
    assert isinstance(beat_times, np.ndarray)
    assert len(beat_times) >= 0

def test_detect_notes_returns_list_of_strings():
    sr = 22050
    y = np.random.randn(sr)

    beat_times = np.array([0.1, 0.5, 1.0])

    notes = mp3Reader.detect_notes(beat_times, y, sr)

    assert isinstance(notes, list)
    assert len(notes) == len(beat_times)

    for note in notes:
        assert isinstance(note, str)

def test_detect_key_error_handling():
    result = mp3Reader.detect_key(None, None)
    assert isinstance(result, str)
    assert "Error" in result

def generate_sine_wave(freq, duration=1.0, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * freq * t)

def test_detect_note_a4():
    sr = 22050

    # Generate A4 tone (440 Hz)
    y = generate_sine_wave(440, sr=sr)

    beat_times = np.array([0.5])

    notes = mp3Reader.detect_notes(beat_times, y, sr)

    assert notes[0] == "A4"

def test_detect_key_c():
    sr = 22050

    c = generate_sine_wave(261.63, sr=sr)   # C
    e = generate_sine_wave(329.63, sr=sr)   # E
    g = generate_sine_wave(392.00, sr=sr)   # G

    y = c + e + g

    key = mp3Reader.detect_key(y, sr)

    assert key == "C"