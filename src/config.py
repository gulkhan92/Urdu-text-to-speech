"""Configuration models with Pydantic validation."""
from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, validator, Field
import torch

@dataclass
class TTSConfig:
    \"\"\"DeepSpeak TTS Configuration.\"\"\"
    checkpoint: str = "mahwizzzz/deepspeak-v1"
    model_device: str = Field(default="cuda", regex="^(cuda|cpu)$")
    model_precision: str = Field(default="bf16", regex="^(bf16|fp32)$")
    codec_device: str = Field(default="cuda", regex="^(cuda|cpu)$")
    num_steps: int = Field(default=40, ge=10, le=100)
    cfg_scale_text: float = Field(default=3.0, ge=1.0, le=10.0)
    cfg_scale_speaker: float = Field(default=5.0, ge=1.0, le=10.0)
    max_seconds: float = Field(default=10.0, ge=1.0, le=60.0)

class TranscriberConfig(BaseModel):
    \"\"\"Whisper Transcriber Configuration.\"\"\"
    model_size: str = Field(default="small", regex="^(tiny|base|small|medium|large)$")
    language: str = "ur"
    device: Optional[str] = None
    
    @validator('device', pre=True, always=True)
    def set_device(cls, v):
        if v is None:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return v

if __name__ == "__main__":
    # Test validation
    tts_cfg = TTSConfig(num_steps=20)
    trans_cfg = TranscriberConfig(model_size="base")
    print("Configs validated:", tts_cfg, trans_cfg)

