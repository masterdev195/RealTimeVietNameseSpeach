import numpy as np
import noisereduce as nr
import torchaudio

try:
    from faster_whisper.audio import decode_audio
except Exception:
    decode_audio = None


def load_audio(file_path, target_sr=16000):
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)

        return waveform.squeeze(0).numpy().astype(np.float32), target_sr
    except Exception as torchaudio_error:
        if decode_audio is None:
            raise RuntimeError(
                f"Không đọc được audio bằng torchaudio ({torchaudio_error}) và không có decode_audio fallback."
            )

        audio = decode_audio(file_path, sampling_rate=target_sr)
        return np.asarray(audio, dtype=np.float32), target_sr


def normalize_audio_for_file(audio_data, sample_rate=16000, noise_decrease=0.5):
    audio = np.asarray(audio_data, dtype=np.float32)
    if audio.size == 0:
        return audio

    audio = nr.reduce_noise(y=audio, sr=sample_rate, prop_decrease=noise_decrease)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio


def normalize_peak(audio_data):
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        return audio_data / max_val
    return audio_data
