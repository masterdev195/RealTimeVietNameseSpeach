import numpy as np
import torch
import noisereduce as nr
from faster_whisper import WhisperModel

from services import (
    clean_transcript_text,
    guess_text_language,
    load_audio,
    normalize_audio_for_file,
    normalize_peak,
    segments_from_word_timestamps,
)


class SpeechProcessor:
    def __init__(self, model_size="large-v3"):
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=8,
        )
        self.last_text = ""
        self.vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        (self.get_speech_timestamps, _, _, _, _) = utils

    def is_speech(self, audio_data, threshold=0.35):
        audio_tensor = torch.from_numpy(normalize_peak(audio_data)).float()
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor, self.vad_model, sampling_rate=16000, threshold=threshold
        )
        return len(speech_timestamps) > 0

    def transcribe(self, audio_data):
        audio_clean = nr.reduce_noise(y=audio_data, sr=16000, prop_decrease=0.8)
        prompt = "Phụ đề tiếng Việt chuẩn. Giữ nguyên từ/cụm tiếng Anh xuất hiện trong lời nói. Không dịch."

        segments, _ = self.model.transcribe(
            audio_clean,
            task="transcribe",
            language="vi",
            beam_size=2,
            initial_prompt=prompt,
            word_timestamps=True,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            log_prob_threshold=-1.0,
        )

        full_text = ""
        results = []
        for segment in segments:
            full_text += segment.text
            results.append(
                {
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip(),
                }
            )

        self.last_text = full_text[-100:]
        return results

    def transcribe_file(self, file_path):
        audio_data, sample_rate = load_audio(file_path, target_sr=16000)
        audio_data = normalize_audio_for_file(audio_data, sample_rate=sample_rate, noise_decrease=0.5)
        duration_sec = len(audio_data) / float(sample_rate)
        print(f"[Transcribe] audio duration: {duration_sec:.2f}s")

        prompt = "Tạo phụ đề chính xác theo lời nói gốc. Ưu tiên tiếng Việt, giữ nguyên các từ/câu tiếng Anh, không dịch nội dung."
        segments, info = self.model.transcribe(
            audio_data,
            task="transcribe",
            language="vi",
            beam_size=5,
            best_of=5,
            initial_prompt=prompt,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 200},
            word_timestamps=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.7,
            log_prob_threshold=-1.2,
            temperature=0,
        )

        detected_language = getattr(info, "language", "unknown")
        results = segments_from_word_timestamps(segments, detected_language, guess_text_language)
        if results:
            print(f"[Transcribe] segments: {len(results)}, last end: {results[-1]['end']:.2f}s")
        return results

    def transcribe_file_basic(self, file_path):
        audio_data, _ = load_audio(file_path, target_sr=16000)
        audio_data = normalize_audio_for_file(audio_data, sample_rate=16000, noise_decrease=0.5)
        prompt = "Tạo phụ đề chính xác theo lời nói gốc. Ưu tiên tiếng Việt, giữ nguyên các từ/câu tiếng Anh, không dịch nội dung."

        segments, info = self.model.transcribe(
            audio_data,
            task="transcribe",
            language="vi",
            beam_size=5,
            best_of=5,
            initial_prompt=prompt,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 200},
            word_timestamps=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.7,
            log_prob_threshold=-1.2,
            temperature=0,
        )

        results = []
        for segment in segments:
            text = (segment.text or "").strip()
            if not text:
                continue
            results.append(
                {
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": text,
                    "language": guess_text_language(text),
                    "detected_language": getattr(info, "language", "unknown"),
                    "speaker_id": 0,
                    "speaker": "Người nói 1",
                }
            )

        return results

    def clean_text(self, current_text):
        return clean_transcript_text(current_text)
