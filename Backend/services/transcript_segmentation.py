def segments_from_word_timestamps(whisper_segments, detected_language, language_detector):
    chunks = []
    punctuation = {".", "?", "!", "…"}

    for segment in whisper_segments:
        words = getattr(segment, "words", None) or []
        if not words:
            text = (segment.text or "").strip()
            if not text:
                continue
            chunks.append({
                "start": round(float(segment.start), 2),
                "end": round(float(segment.end), 2),
                "text": text,
                "language": language_detector(text),
                "detected_language": detected_language
            })
            continue

        current_tokens = []
        chunk_start = None
        last_end = None

        for word in words:
            token = (getattr(word, "word", "") or "")
            w_start = getattr(word, "start", None)
            w_end = getattr(word, "end", None)

            if w_start is None or w_end is None:
                continue

            if chunk_start is None:
                chunk_start = float(w_start)
                last_end = float(w_end)
                current_tokens.append(token)
                continue

            gap = float(w_start) - float(last_end)
            duration = float(w_end) - float(chunk_start)
            end_sentence = token.strip().endswith(tuple(punctuation))

            if (gap > 0.7 or duration > 4.2) and current_tokens:
                text = "".join(current_tokens).strip()
                if text:
                    chunks.append({
                        "start": round(chunk_start, 2),
                        "end": round(float(last_end), 2),
                        "text": text,
                        "language": language_detector(text),
                        "detected_language": detected_language
                    })
                current_tokens = [token]
                chunk_start = float(w_start)
                last_end = float(w_end)
                continue

            current_tokens.append(token)
            last_end = float(w_end)

            if end_sentence and current_tokens:
                text = "".join(current_tokens).strip()
                if text:
                    chunks.append({
                        "start": round(chunk_start, 2),
                        "end": round(float(last_end), 2),
                        "text": text,
                        "language": language_detector(text),
                        "detected_language": detected_language
                    })
                current_tokens = []
                chunk_start = None
                last_end = None

        if current_tokens and chunk_start is not None and last_end is not None:
            text = "".join(current_tokens).strip()
            if text:
                chunks.append({
                    "start": round(chunk_start, 2),
                    "end": round(float(last_end), 2),
                    "text": text,
                    "language": language_detector(text),
                    "detected_language": detected_language
                })

    return chunks
