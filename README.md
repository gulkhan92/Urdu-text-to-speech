# Urdu Text-to-Speech Pipeline 🗣️🇺🇷

Modular pipeline for **Urdu ASR → TTS** using state-of-the-art models:
- **Transcription**: [Whisper-small](https://openai.com/research/whisper) (OpenAI)
- **Synthesis**: [DeepSpeak-v1](https://huggingface.co/mahwizzzz/deepspeak-v1) (voice cloning)

![Pipeline](https://i.imgur.com/demo-pipeline.png)

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run pipeline
python main.py
```

**Input**: Urdu speech audio (.wav, .mp3, .flac, .ogg, .m4a)  
**Output**: Voice-cloned synthesized speech

## 📁 Features

- ✅ Multi-format audio input (MP3/WAV/FLAC/OGG/M4A)
- ✅ Voice cloning from reference audio  
- ✅ Automatic CUDA/CPU detection
- ✅ Configurable quality/speed
- ✅ Text editing callback for corrections/translation

## 💻 Usage Examples

### 1. Full Pipeline (ASR → Edit → TTS)
```python
from main import UrduSpeechPipeline

pipeline = UrduSpeechPipeline()

# Transcribe → Add text → Voice clone original speaker
result_text = pipeline.process_audio(
    "input_urdu.mp3",           # Input speech  
    "output_cloned.wav",        # Voice-cloned output
    modify_text=lambda t: t + " یہ اضافی متن ہے!"
)
```

### 2. ASR Only
```python
from main import UrduTranscriber
transcriber = UrduTranscriber()
text = transcriber.transcribe("urdu_speech.mp3")
print(text)
```

### 3. TTS Only (Voice Cloning)
```python
from main import UrduTTS
tts = UrduTTS()
tts.synthesize(
    "نوائے پاکستان!", 
    "speaker_reference.wav",  # 3-10s sample
    "cloned_output.wav"
)
```

## ⚙️ Configuration

Edit dataclasses in `main.py`:
```python
TTSConfig(num_steps=20, model_device="cuda")  # Faster/High quality
TranscriberConfig(model_size="base")          # Better accuracy
```

## 🔧 Dependencies

```bash
pip install -r requirements.txt
# On Apple Silicon M1/M2: pip install --extra-index-url https://download.pytorch.org/whl/cpu torch
```

**Note**: `irodori-tts` requires [ffmpeg](https://ffmpeg.org/).

## 🧪 Testing

1. Download Urdu speech sample
2. `python main.py`
3. Check `synthesized_output.wav`

## 🤝 Contributing

1. Fork & clone
2. `pip install -r requirements.txt -e .`
3. Add features/tests
4. PR welcome!

## 📄 License
MIT

