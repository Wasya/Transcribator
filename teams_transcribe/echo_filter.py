"""Detection of microphone echo: remote voices leaking from the speakers into the mic.

Without headphones the microphone hears the other participants, so their words
show up twice - once on the system channel (correct) and once attributed to "Я".
The two engines segment the same speech differently, so single utterances rarely
match one-to-one; instead each mic utterance is compared against the concatenated
system text of the surrounding time window, and counts as echo when (almost) all
of its words appear there in order.
"""
import re
from datetime import timedelta
from difflib import SequenceMatcher

from teams_transcribe.transcript_store import Utterance

_NON_WORD = re.compile(r"[^\w\s]+", re.UNICODE)


def normalize_words(text: str) -> list[str]:
    text = _NON_WORD.sub(" ", text.lower().replace("ё", "е"))
    return text.split()


def containment(container, part, *, min_block: int = 1) -> float:
    """Fraction of `part` (a sequence of words or characters) found in `container`
    in order, counting only matching runs of at least min_block elements (0..1)."""
    if not part:
        return 0.0
    matcher = SequenceMatcher(None, container, part, autojunk=False)
    matched = sum(b.size for b in matcher.get_matching_blocks() if b.size >= min_block)
    return matched / len(part)


def find_mic_echo(
    utterances: list[Utterance],
    *,
    window: float = 3.0,
    min_ratio: float = 0.8,
    min_char_ratio: float = 0.85,
    min_words: int = 3,
) -> set[int]:
    """Return id()s of mic utterances that merely repeat what the system channel heard.

    window         - seconds of slack around the mic utterance when collecting system text
    min_ratio      - required share of mic words found in that system text
    min_char_ratio - fallback for garbled echo ("что это так" vs "что это такое"):
                     share of mic characters found in runs of 3+ characters
    min_words      - shorter mic utterances ("да", "угу") are only dropped on an exact match
    """
    system = sorted(
        (u for u in utterances if u.speaker_key[0] == "system" and u.is_final),
        key=lambda u: u.timestamp,
    )
    if not system:
        return set()
    slack = timedelta(seconds=window)
    echo: set[int] = set()
    for u in utterances:
        if u.speaker_key[0] != "mic" or not u.is_final:
            continue
        words = normalize_words(u.text)
        if not words:
            continue
        lo = u.timestamp - slack
        hi = u.timestamp + timedelta(seconds=u.duration or 0.0) + slack
        nearby: list[str] = []
        for s in system:
            s_end = s.timestamp + timedelta(seconds=s.duration or 0.0)
            if s_end < lo:
                continue
            if s.timestamp > hi:
                break
            nearby.extend(normalize_words(s.text))
        if not nearby:
            continue
        if len(words) < min_words:
            if containment(nearby, words) == 1.0:
                echo.add(id(u))
        elif containment(nearby, words) >= min_ratio:
            echo.add(id(u))
        elif containment(" ".join(nearby), " ".join(words), min_block=3) >= min_char_ratio:
            echo.add(id(u))
    return echo
