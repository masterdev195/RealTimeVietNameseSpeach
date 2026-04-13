import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from processor import SpeechProcessor
import numpy as np
import json
import asyncio
import sys

app = FastAPI()

print("--- ĐANG KHỞI ĐỘNG HỆ THỐNG PHỤ ĐỀ ---")

try:
    print("1. Đang nạp bộ lọc giọng nói (VAD)...")
    # Khởi tạo processor - Máy i7 Gen 11 hãy dùng "small" cho an toàn
    processor = SpeechProcessor(model_size="small") 
    print("2. Đã nạp xong Model Whisper!")
except Exception as e:
    print(f"❌ LỖI KHI NẠP MODEL: {e}")
    sys.exit(1)
def clean_subtitle(text):
    if not text: return ""
    # Viết hoa chữ cái đầu
    text = text[0].upper() + text[1:]
    # Loại bỏ khoảng trắng thừa
    text = " ".join(text.split())
    # Nếu câu quá ngắn (dưới 3 từ) mà không phải câu chào, thường là rác
    if len(text.split()) < 3 and text not in ["Chào bạn.", "Cảm ơn."]:
        return ""
    return text
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