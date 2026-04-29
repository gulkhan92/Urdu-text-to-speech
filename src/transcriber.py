"""Urdu Speech-to-Text using Whisper."""
import os
from pathlib import Path
from typing import Optional
from .config import TranscriberConfig
import whisper

SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}

class UrduTranscriber:
    \"\"\"Urdu ASR with Whisper and format validation.\"\"\"
    
    def __init__(self, config: Optional[TranscriberConfig] = None):
        self.config = config or TranscriberConfig()
        print(f"Loading Whisper {self.config.model_size} (device: {self.config.device})...")
        self.model = whisper.load_model(self.config.model_size, device=self.config.device)
    
    def validate_audio(self, audio_path: str) -> Path:
        \"\"\"Validate audio file path and format.\"\"\"
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format {path.suffix}. Supported: {SUPPORTED_FORMATS}")
        if path.stat().st_size == 0:
            raise ValueError("Audio file is empty")
        return path
    
    def transcribe(self, audio_path: str) -> str:
        \"\"\"Transcribe Urdu audio to text.\"\"\"
        audio_path = self.validate_audio(audio_path)
        print(f"Transcribing {audio_path.suffix}: {audio_path.name}")
        
        result = self.model.transcribe(
            str(audio_path),
            language=self.config.language,
            task="transcribe",
            fp16=(self.config.device == "cuda")
        )
        text = result["text"].strip()
        if not text:
            raise ValueError("No speech detected in audio")
        return text

