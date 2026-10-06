import re

# Whisper was trained on subtitled videos, so on music, jingles and silence it
# often "recognizes" credits and sign-offs nobody said. These are cut out of the text.

# Phrases that are a hallucination only when they are the whole recognized text.
_WHOLE = {
    "продолжение следует",
    "спасибо за просмотр",
    "подписывайтесь на канал",
    "подписывайтесь на наш канал",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
}

_PATTERNS = [
    # "Редактор субтитров А.Семкин Корректор А.Егорова": names start with a capital (case-sensitive part).
    r"корректор\s+(?-i:(?:[А-ЯЁA-Z]\.\s?)?[А-ЯЁA-Z][\w-]*)",
    r"редактор\s+субтитров\b(?:\s+(?-i:(?:[А-ЯЁA-Z]\.\s?)?[А-ЯЁA-Z][\w-]*)){0,2}",
    r"субтитры\s+(?:сделал|сделаны|создавал|создал|подогнал|делал|добавил|подготовил|предоставил)\w*\s+"
    r"(?:«[^»]*»|\"[^\"]*\"|\S+)",
    r"subtitles?\s+(?:by|created by)\s+\S+(?:\s+\S+)?",
    r"amara\.org\S*",
    r"untertitel\w*\s+(?:von|der|des|im auftrag)\s+[^.!?…\n]*",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip(" \t\n.,!?…-—«»\"'")).lower()


def clean_hallucinations(text: str) -> str:
    """Remove typical Whisper hallucinations; returns "" if nothing real is left."""
    if _normalize(text) in _WHOLE:
        return ""
    for pattern in _COMPILED:
        text = pattern.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[\s.,!?…-]+", "", text)  # punctuation orphaned by a removed phrase
    return text if re.search(r"\w", text) else ""
