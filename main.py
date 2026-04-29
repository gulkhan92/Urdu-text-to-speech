"""
Urdu Speech Processing Pipeline
- Transcription: Whisper (small model) for Urdu
- Text-to-Speech: DeepSpeak-v1 (mahwizzzz/deepspeak-v1)

Requirements:
pip install torch soundfile openai-whisper transformers safetensors huggingface_hub
# Also install irodori_tts (DeepSpeak's inference runtime)
pip install irodori-tts  # or from source if not available
"""

import os
import soundfile as sf
import whisper
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

# DeepSpeak imports
try:
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    IRODORI_AVAILABLE = True
except ImportError:
    IRODORI_AVAILABLE = False
    print("Warning: irodori_tts not found. Install it for TTS functionality.")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass
class TTSConfig:
    \"\"\"Configuration for DeepSpeak TTS\"\"\"
    checkpoint: str = "mahwizzzz/deepspeak-v1"
    model_device: str = "cuda"  # or "cpu"
    model_precision: str = "bf16"  # "bf16" or "fp32"
    codec_device: str = "cuda"
    num_steps: int = 40
    cfg_scale_text: float = 3.0
    cfg_scale_speaker: float = 5.0
    max_seconds: float = 10.0

@dataclass
class TranscriberConfig:
    model_size: str = "small"  # "tiny", "base", "small", "medium", "large"
    language: str = "ur"
    device: Optional[str] = None  # None = auto


# ----------------------------------------------------------------------
# Urdu Transcriber Module (Whisper)
# ----------------------------------------------------------------------
class UrduTranscriber:
    \"\"\"Handles Urdu speech transcription using Whisper.\"\"\"
    
    def __init__(self, config: TranscriberConfig = None):
        self.config = config or TranscriberConfig()
        # Automatically select device
        if self.config.device is None:
            import torch
            self.config.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model (small for good quality/speed tradeoff)
        print(f"Loading Whisper {self.config.model_size} model for Urdu...")
        self.model = whisper.load_model(
            self.config.model_size, 
            device=self.config.device
        )
    
    def transcribe(self, audio_path: str) -> str:
        \"\"\"
        Transcribe Urdu audio file to text. Supports WAV, MP3, FLAC, OGG, M4A.
        
        Args:
            audio_path: Path to audio file (.wav, .mp3, .flac, .ogg, .m4a)
        
        Returns:
            Transcribed Urdu text
        \"\"\"
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        supported_formats = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
        if Path(audio_path).suffix.lower() not in supported_formats:
            raise ValueError(f"Unsupported format: {Path(audio_path).suffix}. Use: {', '.join(supported_formats)}")
        
        print(f"Transcribing ({Path(audio_path).suffix}): {audio_path}")
        result = self.model.transcribe(
            audio_path,
            language=self.config.language,
            task="transcribe",
            fp16=(self.config.device == "cuda")
        )
        return result["text"].strip()
    
    def transcribe_from_bytes(self, audio_bytes: bytes, sample_rate: int = 16000):
        \"\"\"If you have raw audio bytes, you can implement this.\"\"\"
        raise NotImplementedError("Use transcribe() with a file path for now.")


# ----------------------------------------------------------------------
# Urdu TTS Module (DeepSpeak-v1)
# ----------------------------------------------------------------------
class UrduTTS:
    \"\"\"Handles Urdu Text-to-Speech using DeepSpeak-v1.\"\"\"
    
    def __init__(self, config: TTSConfig = None):
        if not IRODORI_AVAILABLE:
            raise RuntimeError(
                "irodori_tts is required. Install via: pip install irodori-tts"
            )
        
        self.config = config or TTSConfig()
        self.runtime = self._load_model()
    
    def _load_model(self):
        \"\"\"Load DeepSpeak model from HuggingFace.\"\"\"
        print(f"Loading DeepSpeak from {self.config.checkpoint}...")
        key = RuntimeKey(
            checkpoint=self.config.checkpoint,
            model_device=self.config.model_device,
            model_precision=self.config.model_precision,
            codec_device=self.config.codec_device,
        )
        runtime = InferenceRuntime.from_key(key)
        print("TTS model loaded.")
        return runtime
    
    def synthesize(
        self, 
        text: str, 
        reference_audio: str, 
        output_wav: str,
        num_steps: Optional[int] = None,
        seconds: Optional[float] = None
    ) -> float:
        \"\"\"
        Convert Urdu text to speech using a reference speaker.
        
        Args:
            text: Urdu text string.
            reference_audio: Path to .wav file (3-10 seconds, same speaker).
            output_wav: Path to save synthesized audio.
            num_steps: Diffusion steps (default from config).
            seconds: Max audio length in seconds.
        
        Returns:
            Sample rate of generated audio.
        \"\"\"
        if not os.path.exists(reference_audio):
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")
        
        request = SamplingRequest(
            text=text,
            ref_wav=reference_audio,
            seconds=seconds or self.config.max_seconds,
            num_steps=num_steps or self.config.num_steps,
            cfg_scale_text=self.config.cfg_scale_text,
            cfg_scale_speaker=self.config.cfg_scale_speaker,
        )
        
        print(f"Synthesizing: {text[:50]}...")
        result = self.runtime.synthesize(request)
        
        # Save audio
        sf.write(output_wav, result.audio.squeeze(0).numpy(), result.sample_rate)
        print(f"Audio saved to: {output_wav}")
        return result.sample_rate


# ----------------------------------------------------------------------
# Pipeline: Transcribe -> TTS (voice cloning from the same audio)
# ----------------------------------------------------------------------
class UrduSpeechPipeline:
    \"\"\"
    Complete pipeline:
    - Input: Urdu audio file (speech)
    - Output: Synthesized speech of the transcribed text, using original speaker.
    \"\"\"
    
    def __init__(self, tts_config: TTSConfig = None, whisper_config: TranscriberConfig = None):
        self.transcriber = UrduTranscriber(whisper_config)
        self.tts = UrduTTS(tts_config)
    
    def process_audio(
        self, 
        input_audio: str, 
        output_audio: str,
        modify_text: Optional[callable] = None
    ) -> str:
        \"\"\"
        Transcribe input audio (MP3/WAV/FLAC/etc.), optionally edit text, synthesize with voice cloning.
        
        Args:
            input_audio: Path to Urdu speech audio file (.wav, .mp3, etc.).
            output_audio: Path to save synthesized speech (.wav).
            modify_text: Optional function to edit transcribed text.
        
        Returns:
            Final text used for synthesis.
        \"\"\"
        # Step 1: Transcribe (supports multiple formats)
        original_text = self.transcriber.transcribe(input_audio)
        print(f"Transcribed text: {original_text}")
        
        # Step 2: Optional text modification
        final_text = modify_text(original_text) if modify_text else original_text
        if final_text != original_text:
            print(f"Modified text: {final_text}")
        
        # Step 3: Synthesize using input as speaker reference (TTS handles formats)
        self.tts.synthesize(final_text, input_audio, output_audio)
        
        return final_text


# ----------------------------------------------------------------------
# Example Usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example paths (change these)
    INPUT_AUDIO = "sample_urdu_speech.wav"   # input speech to transcribe
    OUTPUT_TTS = "synthesized_output.wav"
    
    # Option 1: Only transcribe
    transcriber = UrduTranscriber()
    text = transcriber.transcribe(INPUT_AUDIO)
    print(f"\nTranscription result:\n{text}")
    
    # Option 2: Only TTS (requires reference audio and arbitrary text)
    # tts = UrduTTS()
    # tts.synthesize("آج موسم بہت اچھا ہے۔", reference_audio="speaker_sample.wav", output_wav="test.wav")
    
    # Option 3: Full pipeline (transcribe + voice-cloned TTS)
    pipeline = UrduSpeechPipeline()
    pipeline.process_audio(
        input_audio=INPUT_AUDIO,
        output_audio=OUTPUT_TTS,
        modify_text=lambda t: t + " یہ ایک تجربہ ہے۔"  # optional addition
    )
    print("Pipeline completed.")

