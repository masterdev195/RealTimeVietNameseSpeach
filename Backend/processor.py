import numpy as np
import torch
import noisereduce as nr
from faster_whisper import WhisperModel

class SpeechProcessor:
    def __init__(self, model_size="Systran/faster-distil-whisper-large-v3"):
        # Tối ưu cho i7 Gen 11: 8 threads, compute int8
        self.model = WhisperModel(
            model_size, 
            device="cpu", 
            compute_type="int8", 
            cpu_threads=8
        )
        self.last_text = "" # Lưu câu trước ở đây
        
        # Load VAD để lọc im lặng
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', 
            model='silero_vad',
            trust_repo=True
        )
        (self.get_speech_timestamps, _, _, _, _) = utils

    def is_speech(self, audio_data, threshold=0.35):
        # Chuẩn hóa âm lượng
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
            
        audio_tensor = torch.from_numpy(audio_data).float()
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor, self.vad_model, sampling_rate=16000, threshold=threshold
        )
        return len(speech_timestamps) > 0

    def transcribe(self, audio_data):
        # 1. Lọc nhiễu kỹ thuật số
        audio_clean = nr.reduce_noise(y=audio_data, sr=16000, prop_decrease=0.8)
        
        # 2. Nhận diện kèm timestamps
        # Initial prompt giúp định hướng phụ đề tiếng Việt chuẩn xác
        prompt = "Phụ đề tiếng Việt cho người khiếm thính, nội dung chính xác, đầy đủ dấu."
        
        segments, _ = self.model.transcribe(
            audio_clean, 
            language="vi", 
            beam_size=2, 
            initial_prompt=prompt,
            word_timestamps=True, # Quan trọng để làm phụ đề
            no_speech_threshold=0.6, 
            condition_on_previous_text=True,
            log_prob_threshold=-1.0
        )
        full_text =""
        results = []
        for segment in segments:
            full_text += segment.text
            results.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })
        self.last_text = full_text[-100:]
        return results
    def clean_text(self, current_text):
    # Loại bỏ các ký tự lạ hoặc các câu ảo giác (hallucinations)
        hallucinations = ["Cảm ơn", "Thank you", "Hãy đăng ký", "Vietsub bởi"]
        for h in hallucinations:
            if h in current_text and len(current_text) < len(h) + 5:
                return ""
                
        # Xử lý dấu câu lộn xộn ở đầu câu
        current_text = current_text.lstrip(".,!?- ")
        
        return current_text