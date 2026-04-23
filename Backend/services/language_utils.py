import re

_VIETNAMESE_CHARS = "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"


def guess_text_language(text):
    if not text:
        return "unknown"

    lowered = text.lower()
    if any(ch in lowered for ch in _VIETNAMESE_CHARS):
        return "vi"

    tokens = re.findall(r"[a-zA-Z']+", lowered)
    if not tokens:
        return "unknown"

    en_hints = {"the", "and", "is", "are", "you", "we", "hello", "thank", "for", "to", "of", "in"}
    score = sum(1 for token in tokens if token in en_hints)
    if score >= 1:
        return "en"

    return "unknown"


def clean_subtitle(text):
    if not text:
        return ""

    text = text[0].upper() + text[1:]
    text = " ".join(text.split())

    if len(text.split()) < 3 and text not in ["Chào bạn.", "Cảm ơn."]:
        return ""

    return text


def clean_transcript_text(current_text):
    hallucinations = ["Cảm ơn", "Thank you", "Hãy đăng ký", "Vietsub bởi"]
    for candidate in hallucinations:
        if candidate in current_text and len(current_text) < len(candidate) + 5:
            return ""

    return current_text.lstrip(".,!?- ")
