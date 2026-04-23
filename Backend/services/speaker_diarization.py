import importlib

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


class SpeakerDiarizer:
    def __init__(self):
        self.speaker_encoder = None

    def _ensure_speaker_encoder(self):
        if self.speaker_encoder is not None:
            return

        try:
            # speechbrain đôi khi vẫn gọi API cũ của torchaudio.
            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: ["soundfile"]
            if not hasattr(torchaudio, "set_audio_backend"):
                torchaudio.set_audio_backend = lambda _: None

            speaker_module = importlib.import_module("speechbrain.inference.speaker")
            encoder_classifier = getattr(speaker_module, "EncoderClassifier")
        except Exception:
            self.speaker_encoder = False
            return

        self.speaker_encoder = encoder_classifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"}
        )

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

    def _kmeans_cosine(self, embeddings, k, max_iters=25):
        if len(embeddings) == 0:
            return []

        k = max(1, min(k, len(embeddings)))
        centroids = [embeddings[i].copy() for i in np.linspace(0, len(embeddings) - 1, k, dtype=int)]
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

        best_k = 1
        best_score = -1e9

        for k in range(1, max_k + 1):
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

            score = cohesion + (0.55 * separation) - (0.03 * (k - 1))

            if score > best_score:
                best_score = score
                best_k = k

        return best_k

    def assign_speakers(self, segments, audio_data, sample_rate=16000, num_speakers=None):
        prepared = []

        for idx, item in enumerate(segments):
            start_sample = max(int(item["start"] * sample_rate), 0)
            end_sample = min(int(item["end"] * sample_rate), len(audio_data))
            if end_sample <= start_sample:
                item["speaker_id"] = 0
                item["speaker"] = "Người nói 1"
                continue

            segment_audio = audio_data[start_sample:end_sample]
            if len(segment_audio) < int(0.35 * sample_rate):
                item["speaker_id"] = -1
                item["speaker"] = "Không rõ"
                continue

            emb = self._segment_embedding(segment_audio)
            if emb is None:
                item["speaker_id"] = -1
                item["speaker"] = "Không rõ"
                continue

            prepared.append((idx, emb))

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

            if prev_label == next_label and curr_label != prev_label:
                labels[i] = prev_label

        for i, label in enumerate(labels):
            segments[i]["speaker_id"] = label
            segments[i]["speaker"] = f"Người nói {label + 1}"

        return segments
