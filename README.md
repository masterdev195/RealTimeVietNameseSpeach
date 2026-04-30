# RealTimeVNSpeech

Vietnamese speech-to-text for uploaded audio files. The backend uses FastAPI and Faster-Whisper, and the frontend is a simple HTML page for uploading audio and viewing sentence-level results.

## Tech Stack & Models
- Backend: FastAPI + Uvicorn.
- ASR model: Faster-Whisper (Whisper large-v3) running on CPU with int8.
- VAD: Silero VAD (snakers4/silero-vad) for speech detection in realtime flow.
- Audio IO: torchaudio + PyAV (via faster-whisper decode).

## Features
- Upload audio and get sentence-level transcript output.
- Vietnamese-first transcription, keeps English words as-is.
- Simple web UI for quick testing.

## Requirements
- Windows 10/11
- Python 3.11

## Setup (Windows PowerShell)
1) Create and activate venv
```powershell
python -m venv venv311
.\venv311\Scripts\Activate.ps1
```

2) Install dependencies (CPU-only torch/torchaudio)
```powershell
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

3) Run backend server
```powershell
python Backend/main.py
```

4) Open frontend
- Open `Frontend/indec.html` in a browser.
- Click "Xử lý file" to upload and transcribe.

## Notes
- `Frontend/indec.html` is the entry UI.
- The API endpoint is `POST /transcribe-file`.