"""Utility functions for audio processing."""
from pathlib import Path
import os

def ensure_dir(path: str) -> Path:
    \"\"\"Ensure directory exists, return Path.\"\"\"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def validate_urdu_text(text: str) -> str:
    \"\"\"Basic Urdu text validation (non-empty, has Urdu chars).\"\"\"
    text = text.strip()
    if len(text) < 1:
        raise ValueError("Text cannot be empty")
    # Basic Urdu range check
    if not any(0x0600 <= ord(c) <= 0x06FF for c in text):
        print("Warning: No Urdu characters detected")
    return text

def get_audio_duration(path: str) -> float:
    \"\"\"Get audio duration in seconds (requires librosa).\"\"\"
    try:
        import librosa
        duration = librosa.get_duration(filename=path)
        return duration
    except ImportError:
        print("librosa not installed, skipping duration check")
        return 0.0

