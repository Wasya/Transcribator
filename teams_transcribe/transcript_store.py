import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

SpeakerKey = tuple[str, Optional[int]]


@dataclass
class Utterance:
    speaker_key: SpeakerKey
    timestamp: datetime
    is_final: bool
    text: str


class TranscriptStore:
    """Thread-safe in-memory store of utterances and speaker display names."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._utterances: list[Utterance] = []
        self._speaker_names: dict[SpeakerKey, str] = {}
        self._system_speaker_order: list[int] = []

    def register_mic(self) -> None:
        """Ensure the mic speaker key has a default display name. Idempotent."""
        with self._lock:
            self._speaker_names.setdefault(("mic", None), "Я")

    def get_or_create_system_speaker_name(self, speaker_id: int) -> tuple[str, bool]:
        """Return (name, is_new) for a system-channel speaker_id, assigning a
        default 'Собеседник N' name the first time this speaker_id is seen."""
        with self._lock:
            key: SpeakerKey = ("system", speaker_id)
            if key not in self._speaker_names:
                self._system_speaker_order.append(speaker_id)
                n = len(self._system_speaker_order)
                self._speaker_names[key] = f"Собеседник {n}"
                return self._speaker_names[key], True
            return self._speaker_names[key], False

    def rename(self, speaker_key: SpeakerKey, new_name: str) -> None:
        with self._lock:
            self._speaker_names[speaker_key] = new_name

    def speaker_name(self, speaker_key: SpeakerKey) -> str:
        with self._lock:
            return self._speaker_names.get(speaker_key, str(speaker_key))

    def add_utterance(self, speaker_key: SpeakerKey, timestamp: datetime, is_final: bool, text: str) -> None:
        with self._lock:
            self._utterances.append(Utterance(speaker_key, timestamp, is_final, text))

    def final_utterances(self) -> list[Utterance]:
        """All is_final=True utterances, in insertion order."""
        with self._lock:
            return [u for u in self._utterances if u.is_final]
