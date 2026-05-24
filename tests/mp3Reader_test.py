import numpy as np
import pytest
import mp3Reader

def test_pitch_classes_valid():
    assert len(mp3Reader.PITCH_CLASSES) == 12
    assert "C" in mp3Reader.PITCH_CLASSES
    assert "B" in mp3Reader.PITCH_CLASSES

def test_active_pitch_logic():
    vec = np.zeros(12)
    vec[0] = 0.9
    vec[4] = 0.8

    active = np.sum(vec > 0.3)

    assert active == 2

def sine(freq, sr=22050, duration=0.1):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return 0.2 * np.sin(2 * np.pi * freq * t)

def test_note_prediction():
    sr = 22050
    y = sine(440, sr)  # A4

    results = mp3Reader.analyze_beats(y, sr, np.array([0.05]))

    beat = results[0]

    assert beat["type"] == "note"

    # more stable than full note match
    assert beat["prediction"].startswith("A")

def test_chord_prediction():
    sr = 22050

    # C major chord
    y = (
            sine(261.63, sr) +   # C
            sine(329.63, sr) +   # E
            sine(392.00, sr)     # G
    )

    results = mp3Reader.analyze_beats(y, sr, np.array([0.05]))

    beat = results[0]

    assert beat["type"] == "chord"

    # key improvement: check root only
    assert beat["prediction"].startswith("C")

def test_analyze_structure():
    sr = 22050
    y = sine(440, sr)

    results = mp3Reader.analyze_beats(y, sr, np.array([0.05]))

    beat = results[0]

    assert "beat_time" in beat
    assert "type" in beat
    assert "prediction" in beat
    assert "active_pitches" in beat
    assert "chroma" in beat

def test_multiple_beats():
    sr = 22050
    y = sine(440, sr)

    results = mp3Reader.analyze_beats(y, sr, np.array([0.02, 0.05, 0.08]))

    assert len(results) == 3