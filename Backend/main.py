import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
import asyncio
import sys
import os
import tempfile
import traceback

try:
    import torchaudio

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda _: None
except Exception:
    pass

from processor import SpeechProcessor
from services.language_utils import clean_subtitle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("--- ĐANG KHỞI ĐỘNG HỆ THỐNG PHỤ ĐỀ ---")

try:
    print("1. Đang nạp bộ lọc giọng nói (VAD)...")
    # Ưu tiên độ chính xác cho bài toán file upload.
    processor = SpeechProcessor(model_size="large-v3") 
    print("2. Đã nạp xong Model Whisper!")
except Exception as e:
    print(f"❌ LỖI KHI NẠP MODEL: {e}")
    sys.exit(1)


@app.post("/transcribe-file")
async def transcribe_file(
    audio_file: UploadFile = File(...)
):
    file_name = audio_file.filename or "audio"
    extension = os.path.splitext(file_name)[1] or ".wav"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
            content = await audio_file.read()
            if not content:
                raise HTTPException(status_code=400, detail="File âm thanh trống")
            tmp_file.write(content)
            tmp_path = tmp_file.name
            print(f"[Upload] filename={file_name}, bytes={len(content)}")

        try:
            if tmp_path and os.path.exists(tmp_path):
                print(f"[Upload] tmp_size={os.path.getsize(tmp_path)} bytes")
            results = processor.transcribe_file(file_path=tmp_path)
            mode = "transcript"
        except Exception as transcribe_error:
            print(f"⚠️ Lỗi nhận diện, chuyển sang transcript thường: {transcribe_error}")
            traceback.print_exc()
            results = processor.transcribe_file_basic(file_path=tmp_path)
            mode = "transcript-basic"

        return {
            "file_name": file_name,
            "segments": results,
            "total_segments": len(results),
            "mode": mode
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Không thể xử lý file âm thanh: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Đã kết nối với trình duyệt!")
    buffer = []
    
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            buffer.append(audio_chunk)

            if len(buffer) > 15: # Xử lý mỗi ~3-4 giây
                audio_to_process = np.concatenate(buffer)
                buffer = buffer[-7:] 

                if processor.is_speech(audio_to_process):
                    # Chạy nhận diện trong thread riêng để không treo socket
                    loop = asyncio.get_event_loop()
                    results = await loop.run_in_executor(None, processor.transcribe, audio_to_process)
                    
                    if results:
                        await websocket.send_text(json.dumps(results))
                        print(f"🎤 AI: {clean_subtitle(results[0]['text'])}")

    except WebSocketDisconnect:
        print("🔌 Trình duyệt đã ngắt kết nối.")
    except Exception as e:
        print(f"⚠️ Lỗi trong quá trình nhận diện: {e}")

# ĐÂY LÀ DÒNG QUAN TRỌNG NHẤT ĐỂ GIỮ SERVER KHÔNG THOÁT
if __name__ == "__main__":
    print("3. Đang khởi chạy Server tại cổng 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")