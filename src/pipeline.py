"""Main pipeline orchestrating ASR and TTS."""
from typing import Optional, Callable
from pathlib import Path
from .config import TTSConfig, TranscriberConfig
from .transcriber import UrduTranscriber
from .tts import UrduTTS

class UrduSpeechPipeline:
    \"\"\"Complete Urdu speech processing pipeline.\"\"\"
    
    def __init__(self, tts_config: Optional[TTSConfig] = None, 
                 transcriber_config: Optional[TranscriberConfig] = None):
        self.transcriber = UrduTranscriber(transcriber_config)
        self.tts = UrduTTS(tts_config)
    
    def process_audio(self, input_audio: str, output_audio: str, 
                     modify_text: Optional[Callable[[str], str]] = None) -> str:
        \"\"\"
        Full pipeline: Transcribe → Modify → Voice-clone TTS.
        
        Args:
            input_audio: Input audio file path
            output_audio: Output synthesized audio path  
            modify_text: Optional text transformer function
            
        Returns:
            Final processed text
        \"\"\"
        # 1. Transcribe
        original_text = self.transcriber.transcribe(input_audio)
        print(f"Original transcription: {original_text}")
        
        # 2. Modify text (optional)
        final_text = modify_text(original_text) if modify_text else original_text
        print(f"Final text: {final_text}")
        
        # 3. Voice-clone TTS using input as reference
        self.tts.synthesize(final_text, input_audio, output_audio)
        
        return final_text
    
    def transcribe_only(self, audio_path: str) -> str:
        \"\"\"ASR only.\"\"\"
        return self.transcriber.transcribe(audio_path)
    
    def synthesize_only(self, text: str, ref_audio: str, output_path: str) -> float:
        \"\"\"TTS only.\"\"\"
        return self.tts.synthesize(text, ref_audio, output_path)

if __name__ == "__main__":
    pipeline = UrduSpeechPipeline()
    text = pipeline.process_audio("input.wav", "output.wav")
    print("Pipeline complete. Text:", text)

