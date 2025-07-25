# Wakanda Whisper

Wakanda Whisper is a framework for automatic speech recognition (ASR) for African languages using OpenAI Whisper.

## Installation

```bash
pip install wakanda-whisper
```

## Usage

```python
import  wakanda_whisper

model = wakanda_whisper.from_pretrained("WakandaAI/wakanda-whisper-small-rw-v1")
result = model.transcribe("path/to/audio.wav")

print(result['text'].strip())
```