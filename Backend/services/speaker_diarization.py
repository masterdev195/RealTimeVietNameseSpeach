import importlib
import os
import inspect
import functools

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda _: None


class SpeakerDiarizer:
    def __init__(self):
        self.speaker_encoder = None
        self.diarization_pipeline = None

    def merge_segments_for_diarization(self, segments, min_duration=2.0, max_duration=8.0, max_gap=0.8):
        if not segments:
            return []

        merged = []
        current = None
        buffer_indices = []

        for idx, seg in enumerate(segments):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            text = (seg.get("text") or "").strip()

            if current is None:
                current = {"start": start, "end": end, "text": text}
                buffer_indices = [idx]
                continue

            gap = start - float(current["end"])
            duration = float(current["end"]) - float(current["start"])
            end_sentence = text.endswith((".", "?", "!", "…"))

            should_flush = gap > max_gap or duration >= max_duration or (end_sentence and duration >= min_duration)

            if should_flush:
                current["source_indices"] = buffer_indices
                merged.append(current)
                current = {"start": start, "end": end, "text": text}
                buffer_indices = [idx]
                continue

            current["end"] = end
            current["text"] = (current["text"] + " " + text).strip()
            buffer_indices.append(idx)

        if current is not None:
            current["source_indices"] = buffer_indices
            merged.append(current)

        return merged

    def create_time_windows(self, audio_len, sample_rate, window_sec=4.0, hop_sec=3.0):
        if sample_rate <= 0 or audio_len <= 0:
            return []

        total_duration = float(audio_len) / float(sample_rate)
        if total_duration <= 0:
            return []

        windows = []
        start = 0.0
        while start < total_duration:
            end = min(start + window_sec, total_duration)
            windows.append({"start": start, "end": end, "text": ""})
            if end >= total_duration:
                break
            start += hop_sec

        return windows

    def propagate_labels_to_segments(self, segments, labeled_chunks):
        if not segments:
            return segments
        if not labeled_chunks:
            for item in segments:
                item["speaker_id"] = -1
                item["speaker"] = "Không rõ"
            return segments

        chunk_idx = 0
        for seg in segments:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", 0.0))

            while chunk_idx < len(labeled_chunks) and float(labeled_chunks[chunk_idx].get("end", 0.0)) <= seg_start:
                chunk_idx += 1

            best_chunk = None
            best_overlap = 0.0
            for cand_idx in (chunk_idx - 1, chunk_idx):
                if cand_idx < 0 or cand_idx >= len(labeled_chunks):
                    continue
                chunk = labeled_chunks[cand_idx]
                ch_start = float(chunk.get("start", 0.0))
                ch_end = float(chunk.get("end", 0.0))
                overlap = min(seg_end, ch_end) - max(seg_start, ch_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_chunk = chunk

            if best_chunk is None or best_overlap <= 0:
                seg["speaker_id"] = -1
                seg["speaker"] = "Không rõ"
                continue

            speaker_id = int(best_chunk.get("speaker_id", -1))
            seg["speaker_id"] = speaker_id
            seg["speaker"] = "Không rõ" if speaker_id < 0 else f"Người nói {speaker_id + 1}"

        return segments

    def assign_speakers_semantic(self, segments, audio_data, sample_rate=16000, num_speakers=None):
        merged = self.merge_segments_for_diarization(segments)
        used_windows = False
        if num_speakers is not None and num_speakers > 1 and len(merged) < num_speakers:
            merged = self.create_time_windows(len(audio_data), sample_rate)
            used_windows = True

        if not merged:
            return segments

        labeled_chunks = self.assign_speakers(
            segments=merged,
            audio_data=audio_data,
            sample_rate=sample_rate,
            num_speakers=num_speakers,
        )
        if num_speakers is not None and num_speakers > 1:
            unique_labels = {
                int(item.get("speaker_id", -1))
                for item in (labeled_chunks or [])
                if int(item.get("speaker_id", -1)) >= 0
            }
            if len(unique_labels) < num_speakers and not used_windows:
                merged = self.create_time_windows(len(audio_data), sample_rate)
                labeled_chunks = self.assign_speakers(
                    segments=merged,
                    audio_data=audio_data,
                    sample_rate=sample_rate,
                    num_speakers=num_speakers,
                )
        labeled_chunks = self.smooth_speaker_labels(labeled_chunks)
        return self.propagate_labels_to_segments(segments, labeled_chunks)

    def _ensure_hf_hub_compat(self):
        try:
            import huggingface_hub as hf
        except Exception:
            return

        if getattr(hf.hf_hub_download, "_patched_use_auth_token", False):
            return

        signature = inspect.signature(hf.hf_hub_download)
        if "use_auth_token" in signature.parameters:
            return

        has_token_param = "token" in signature.parameters

        original_hf_hub_download = hf.hf_hub_download

        @functools.wraps(original_hf_hub_download)
        def patched_hf_hub_download(*args, use_auth_token=None, token=None, **kwargs):
            if token is None and use_auth_token is not None:
                token = use_auth_token
            if has_token_param:
                return original_hf_hub_download(*args, token=token, **kwargs)
            return original_hf_hub_download(*args, **kwargs)

        patched_hf_hub_download._patched_use_auth_token = True
        hf.hf_hub_download = patched_hf_hub_download

    def _ensure_diarization_pipeline(self):
        if self.diarization_pipeline is not None:
            return

        self._ensure_hf_hub_compat()

        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]
        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda _: None

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            print("[Diarization] HF_TOKEN missing - skip pyannote pipeline.")
            self.diarization_pipeline = False
            return

        try:
            from pyannote.audio import Pipeline
        except Exception as exc:
            print(f"[Diarization] pyannote.audio not available - {exc}")
            self.diarization_pipeline = False
            return

        try:
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token
            )
        except Exception:
            print("[Diarization] Failed to load pyannote pipeline - fallback to embedding.")
            self.diarization_pipeline = False
            return

    def _ensure_speaker_encoder(self):
        if self.speaker_encoder is not None:
            return

        self._ensure_hf_hub_compat()

        try:
            # speechbrain đôi khi vẫn gọi API cũ của torchaudio.
            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: ["soundfile"]
            if not hasattr(torchaudio, "set_audio_backend"):
                torchaudio.set_audio_backend = lambda _: None

            speaker_module = importlib.import_module("speechbrain.inference.speaker")
            encoder_classifier = getattr(speaker_module, "EncoderClassifier")
            fetching_module = importlib.import_module("speechbrain.utils.fetching")
            local_strategy = getattr(fetching_module, "LocalStrategy", None)
        except Exception:
            self.speaker_encoder = False
            return

        try:
            kwargs = {}
            if local_strategy is not None:
                kwargs["local_strategy"] = local_strategy.COPY

            self.speaker_encoder = encoder_classifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"},
                **kwargs
            )
        except Exception:
            self.speaker_encoder = False
            return

    def _segment_embedding(self, segment_audio):
        if segment_audio.size == 0:
            return None

        self._ensure_speaker_encoder()

        if self.speaker_encoder is False:
            wav = torch.from_numpy(segment_audio).float().unsqueeze(0)
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=16000,
                n_mfcc=20,
                melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
            )
            mfcc = mfcc_transform(wav).squeeze(0).numpy()
            if mfcc.ndim != 2 or mfcc.shape[1] < 4:
                return None

            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)

            stft = torch.stft(
                wav.squeeze(0),
                n_fft=400,
                hop_length=160,
                win_length=400,
                window=torch.hann_window(400, device=wav.device),
                return_complex=True
            ).abs() + 1e-8
            freqs = torch.linspace(0.0, 8000.0, stft.shape[0]).unsqueeze(1)
            spec_sum = torch.sum(stft, dim=0, keepdim=True) + 1e-8
            centroid = torch.sum(freqs * stft, dim=0, keepdim=True) / spec_sum
            bandwidth = torch.sqrt(torch.sum(((freqs - centroid) ** 2) * stft, dim=0, keepdim=True) / spec_sum)

            centroid_mean = float(torch.mean(centroid))
            centroid_std = float(torch.std(centroid))
            bandwidth_mean = float(torch.mean(bandwidth))
            bandwidth_std = float(torch.std(bandwidth))

            zcr = np.mean(np.abs(np.diff(np.sign(segment_audio)))) / 2.0
            rms = float(np.sqrt(np.mean(np.square(segment_audio)) + 1e-10))

            emb = np.concatenate([
                mfcc_mean.astype(np.float32),
                mfcc_std.astype(np.float32),
                np.array([
                    zcr,
                    np.log(rms + 1e-10),
                    centroid_mean / 8000.0,
                    centroid_std / 8000.0,
                    bandwidth_mean / 8000.0,
                    bandwidth_std / 8000.0,
                ], dtype=np.float32)
            ])
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb

        tensor_audio = torch.from_numpy(segment_audio).float().unsqueeze(0)

        with torch.no_grad():
            emb = self.speaker_encoder.encode_batch(tensor_audio)

        emb = emb.squeeze(0).squeeze(0)
        emb = F.normalize(emb, p=2, dim=0)
        return emb.cpu().numpy()

    def _simple_features(self, segment_audio, sample_rate=16000):
        if segment_audio.size == 0:
            return None

        wav = torch.from_numpy(segment_audio).float().unsqueeze(0)
        rms = float(torch.sqrt(torch.mean(wav ** 2) + 1e-10))
        zcr = float(torch.mean(torch.abs(torch.diff(torch.sign(wav)))) / 2.0)

        stft = torch.stft(
            wav.squeeze(0),
            n_fft=400,
            hop_length=160,
            win_length=400,
            window=torch.hann_window(400, device=wav.device),
            return_complex=True
        ).abs() + 1e-8

        freqs = torch.linspace(0.0, float(sample_rate) / 2.0, stft.shape[0]).unsqueeze(1)
        spec_sum = torch.sum(stft, dim=0, keepdim=True) + 1e-8
        centroid = torch.sum(freqs * stft, dim=0, keepdim=True) / spec_sum
        centroid_mean = float(torch.mean(centroid)) / (float(sample_rate) / 2.0)

        feats = np.array([
            np.log(rms + 1e-10),
            zcr,
            centroid_mean,
        ], dtype=np.float32)

        if not np.isfinite(feats).all():
            return None

        return feats

    def _kmeans_cosine(self, embeddings, k, max_iters=25):
        if len(embeddings) == 0:
            return []

        k = max(1, min(k, len(embeddings)))
        centroids = [embeddings[0].copy()]
        if k > 1:
            for _ in range(1, k):
                distances = []
                for emb in embeddings:
                    min_dist = min(1.0 - float(np.dot(emb, c)) for c in centroids)
                    distances.append(min_dist)
                next_idx = int(np.argmax(distances))
                centroids.append(embeddings[next_idx].copy())
        labels = np.zeros(len(embeddings), dtype=int)

        for _ in range(max_iters):
            changed = False

            for i, emb in enumerate(embeddings):
                sims = [float(np.dot(emb, c)) for c in centroids]
                new_label = int(np.argmax(sims))
                if labels[i] != new_label:
                    labels[i] = new_label
                    changed = True

            for c_idx in range(k):
                group = [embeddings[i] for i in range(len(embeddings)) if labels[i] == c_idx]
                if not group:
                    # Cụm rỗng: chọn điểm xa nhất so với centroid hiện tại để tái khởi tạo.
                    distances = [float(np.dot(embeddings[i], centroids[labels[i]])) for i in range(len(embeddings))]
                    candidate_idx = int(np.argmin(distances))
                    labels[candidate_idx] = c_idx
                    centroids[c_idx] = embeddings[candidate_idx].copy()
                    changed = True
                    continue
                centroid = np.mean(group, axis=0)
                norm = np.linalg.norm(centroid)
                centroids[c_idx] = centroid / norm if norm > 0 else centroid

            if not changed:
                break

        return labels.tolist()

    def _estimate_num_speakers(self, embeddings, max_speakers=4):
        n = len(embeddings)
        if n <= 1:
            return 1

        max_k = max(1, min(max_speakers, n))
        if max_k == 1:
            return 1

        pair_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                pair_sims.append(float(np.dot(embeddings[i], embeddings[j])))

        if not pair_sims:
            return 1

        spread = float(np.std(pair_sims))
        low_quantile = float(np.quantile(pair_sims, 0.2))

        if spread < 0.05 and low_quantile > 0.78:
            return 1

        # Với hội thoại đủ dài, ưu tiên thử tách >=2 người để tránh dồn hết về 1 nhãn.
        min_k = 1
        if n >= 4 and (spread >= 0.04 or low_quantile <= 0.76):
            min_k = 2

        best_k = min_k
        best_score = -1e9

        for k in range(min_k, max_k + 1):
            labels = self._kmeans_cosine(embeddings, k)
            unique = sorted(set(labels))

            centroids = []
            cohesion = 0.0
            for cluster_id in unique:
                group = [embeddings[i] for i in range(n) if labels[i] == cluster_id]
                centroid = np.mean(group, axis=0)
                norm = np.linalg.norm(centroid)
                centroid = centroid / norm if norm > 0 else centroid
                centroids.append(centroid)
                cohesion += float(np.mean([np.dot(e, centroid) for e in group]))

            cohesion /= max(len(unique), 1)

            separation = 0.0
            if len(centroids) > 1:
                sep_vals = []
                for i in range(len(centroids)):
                    for j in range(i + 1, len(centroids)):
                        sep_vals.append(1.0 - float(np.dot(centroids[i], centroids[j])))
                separation = float(np.mean(sep_vals)) if sep_vals else 0.0

            cluster_sizes = [sum(1 for lb in labels if lb == cluster_id) for cluster_id in unique]
            smallest_cluster = min(cluster_sizes) if cluster_sizes else 0
            tiny_cluster_penalty = 0.06 if (k > 1 and smallest_cluster <= 1) else 0.0

            score = cohesion + (0.65 * separation) - (0.02 * (k - 1)) - tiny_cluster_penalty

            if score > best_score:
                best_score = score
                best_k = k

        return best_k

    def _assign_with_pyannote(self, segments, audio_data, sample_rate=16000, num_speakers=None):
        if not segments:
            return segments

        self._ensure_diarization_pipeline()
        if self.diarization_pipeline is False:
            return None

        waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
        diarization = self.diarization_pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=num_speakers if num_speakers and num_speakers > 0 else None
        )

        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append((float(turn.start), float(turn.end), speaker))

        if not turns:
            print("[Diarization] pyannote returned no speaker turns.")
            return None

        speaker_map = {}
        for _, _, speaker in turns:
            if speaker not in speaker_map:
                speaker_map[speaker] = len(speaker_map)

        for item in segments:
            seg_start = float(item.get("start", 0))
            seg_end = float(item.get("end", 0))
            overlaps = {}

            for turn_start, turn_end, speaker in turns:
                if turn_end <= seg_start or turn_start >= seg_end:
                    continue
                overlap = min(seg_end, turn_end) - max(seg_start, turn_start)
                overlaps[speaker] = overlaps.get(speaker, 0.0) + overlap

            if not overlaps:
                item["speaker_id"] = -1
                item["speaker"] = "Không rõ"
                continue

            best_speaker = max(overlaps.items(), key=lambda pair: pair[1])[0]
            speaker_idx = speaker_map[best_speaker]
            item["speaker_id"] = speaker_idx
            item["speaker"] = f"Người nói {speaker_idx + 1}"

        unique_ids = sorted({item.get("speaker_id") for item in segments if item.get("speaker_id") is not None})
        print(f"[Diarization] pyannote assigned speakers: {unique_ids}")

        return segments

    def assign_speakers(self, segments, audio_data, sample_rate=16000, num_speakers=None):
        print(f"[Diarization] request num_speakers={num_speakers}")
        diarized = self._assign_with_pyannote(
            segments=segments,
            audio_data=audio_data,
            sample_rate=sample_rate,
            num_speakers=num_speakers
        )
        if diarized is not None:
            print("[Diarization] using pyannote pipeline.")
            return diarized

        print("[Diarization] fallback to embedding-based clustering.")

        prepared = []
        prepared_features = []
        min_embed_samples = int(0.45 * sample_rate)
        min_fallback_samples = int(0.2 * sample_rate)

        for idx, item in enumerate(segments):
            start_sample = max(int(item["start"] * sample_rate), 0)
            end_sample = min(int(item["end"] * sample_rate), len(audio_data))
            if end_sample <= start_sample:
                item["speaker_id"] = 0
                item["speaker"] = "Người nói 1"
                continue

            segment_audio = audio_data[start_sample:end_sample]
            if len(segment_audio) < min_embed_samples:
                # Mở rộng cửa sổ lấy mẫu quanh đoạn ngắn để đủ dữ liệu embedding.
                needed = max(min_embed_samples - len(segment_audio), 0)
                pad_left = needed // 2
                pad_right = needed - pad_left
                start_pad = max(start_sample - pad_left, 0)
                end_pad = min(end_sample + pad_right, len(audio_data))
                segment_audio = audio_data[start_pad:end_pad]

            if len(segment_audio) < min_fallback_samples:
                item["speaker_id"] = -1
                item["speaker"] = "Không rõ"
                continue

            emb = self._segment_embedding(segment_audio)
            if emb is not None and np.isfinite(emb).all():
                prepared.append((idx, emb))

            feats = self._simple_features(segment_audio, sample_rate=sample_rate)
            if feats is not None:
                prepared_features.append((idx, feats))

        if not prepared:
            for item in segments:
                if "speaker" not in item:
                    item["speaker_id"] = 0
                    item["speaker"] = "Người nói 1"
            return segments

        indices = [x[0] for x in prepared]
        embeddings = [x[1] for x in prepared]

        if num_speakers is not None and num_speakers > 0:
            labels = self._kmeans_cosine(embeddings, num_speakers)
        else:
            guessed = self._estimate_num_speakers(embeddings, max_speakers=4)
            labels = self._kmeans_cosine(embeddings, guessed)

        if num_speakers is not None and num_speakers > 1:
            unique_labels = set(int(lb) for lb in labels) if labels else set()
            if len(unique_labels) < num_speakers and len(prepared_features) >= num_speakers:
                feat_indices = [x[0] for x in prepared_features]
                feat_values = np.stack([x[1] for x in prepared_features], axis=0)
                feat_values = feat_values - np.mean(feat_values, axis=0, keepdims=True)
                norms = np.linalg.norm(feat_values, axis=1, keepdims=True) + 1e-8
                feat_values = feat_values / norms
                labels = self._kmeans_cosine(feat_values.tolist(), num_speakers)
                indices = feat_indices

        for i, seg_idx in enumerate(indices):
            speaker_idx = int(labels[i])
            segments[seg_idx]["speaker_id"] = speaker_idx
            segments[seg_idx]["speaker"] = f"Người nói {speaker_idx + 1}"

        for item in segments:
            if "speaker" not in item:
                item["speaker_id"] = 0
                item["speaker"] = "Người nói 1"

        return segments

    def smooth_speaker_labels(self, segments):
        if not segments:
            return segments

        labels = [int(item.get("speaker_id", 0)) for item in segments]

        for i in range(1, len(labels) - 1):
            prev_label = labels[i - 1]
            curr_label = labels[i]
            next_label = labels[i + 1]

            if prev_label >= 0 and next_label >= 0 and prev_label == next_label and curr_label != prev_label:
                labels[i] = prev_label

        for i, label in enumerate(labels):
            if label < 0:
                segments[i]["speaker_id"] = -1
                segments[i]["speaker"] = "Không rõ"
            else:
                segments[i]["speaker_id"] = label
                segments[i]["speaker"] = f"Người nói {label + 1}"

        return segments
