"""Urdu Text-to-Speech using DeepSpeak-v1."""
import os
from pathlib import Path
from typing import Optional
try:
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    IRODORI_AVAILABLE = True
except ImportError:
    IRODORI_AVAILABLE = False

from .config import TTSConfig
import soundfile as sf

class UrduTTS:
    \"\"\"Voice cloning TTS for Urdu.\"\"\"
    
    def __init__(self, config: Optional[TTSConfig] = None):
        if not IRODORI_AVAILABLE:
            raise RuntimeError("Install irodori-tts: pip install irodori-tts")
        self.config = config or TTSConfig()
        self.runtime = self._load_model()
    
    def _load_model(self):
        \"\"\"Load DeepSpeak model.\"\"\"
        print(f"Loading DeepSpeak from {self.config.checkpoint}...")
        key = RuntimeKey(
            checkpoint=self.config.checkpoint,
            model_device=self.config.model_device,
            model_precision=self.config.model_precision,
            codec_device=self.config.codec_device,
        )
        return InferenceRuntime.from_key(key)
    
    def validate_reference(self, ref_path: str) -> Path:
        \"\"\"Validate reference audio for voice cloning.\"\"\"
        path = Path(ref_path)
        if not path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_path}")
        if path.stat().st_size < 100*1024:  # Min 100KB
            raise ValueError("Reference audio too short (<100KB)")
        return path
    
    def synthesize(self, text: str, reference_audio: str, output_path: str, 
                   num_steps: Optional[int] = None, seconds: Optional[float] = None) -> float:
        \"\"\"Generate speech from text using reference speaker.\"\"\"
        if len(text.strip()) < 1:
            raise ValueError("Text cannot be empty")
        
        ref_path = self.validate_reference(reference_audio)
        output_path = Path(output_path)
        
        request = SamplingRequest(
            text=text.strip(),
            ref_wav=str(ref_path),
            seconds=seconds or self.config.max_seconds,
            num_steps=num_steps or self.config.num_steps,
            cfg_scale_text=self.config.cfg_scale_text,
            cfg_scale_speaker=self.config.cfg_scale_speaker,
        )
        
        print(f"Synthesizing: {text[:50]}...")
        result = self.runtime.synthesize(request)
        sf.write(output_path, result.audio.squeeze(0).numpy(), result.sample_rate)
        print(f"Generated: {output_path}")
        return result.sample_rate

