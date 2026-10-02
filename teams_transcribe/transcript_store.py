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
    start: Optional[float] = None  # seconds from the start of the source's audio
    duration: Optional[float] = None


class TranscriptStore:
    """Thread-safe in-memory store of utterances and speaker display names."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._utterances: list[Utterance] = []
        self._speaker_names: dict[SpeakerKey, str] = {}
        self._system_speaker_order: list[int] = []
        self._auto_named: set[SpeakerKey] = set()

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
                self._auto_named.add(key)
                return self._speaker_names[key], True
            return self._speaker_names[key], False

    def rename(self, speaker_key: SpeakerKey, new_name: str) -> None:
        with self._lock:
            self._speaker_names[speaker_key] = new_name
            self._auto_named.discard(speaker_key)

    def speaker_name(self, speaker_key: SpeakerKey) -> str:
        with self._lock:
            return self._speaker_names.get(speaker_key, str(speaker_key))

    def add_utterance(
        self,
        speaker_key: SpeakerKey,
        timestamp: datetime,
        is_final: bool,
        text: str,
        start: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> None:
        with self._lock:
            self._utterances.append(Utterance(speaker_key, timestamp, is_final, text, start, duration))

    def system_speakers(self) -> list[tuple[SpeakerKey, str]]:
        with self._lock:
            return [(k, n) for k, n in self._speaker_names.items() if k[0] == "system"]

    def system_utterances(self) -> list[Utterance]:
        """Final utterances of the system channel (the ones diarization applies to)."""
        with self._lock:
            return [u for u in self._utterances if u.is_final and u.speaker_key[0] == "system"]

    def apply_diarization(self, assignments: dict[int, int]) -> None:
        """Re-label system utterances after offline diarization.

        assignments maps id(utterance) -> new speaker id. Speakers get fresh default
        names in order of first appearance, except that a name the user typed for the
        old (live) speaker an utterance used to belong to is carried over to the new
        speaker that most often replaces it.
        """
        with self._lock:
            targets = [u for u in self._utterances if id(u) in assignments]
            targets.sort(key=lambda u: u.timestamp)
            votes: dict[int, dict[SpeakerKey, int]] = {}
            for u in targets:
                new_id = assignments[id(u)]
                votes.setdefault(new_id, {}).setdefault(u.speaker_key, 0)
                votes[new_id][u.speaker_key] += 1
            old_names = dict(self._speaker_names)
            old_auto = set(self._auto_named)
            for key in [k for k in self._speaker_names if k[0] == "system"]:
                del self._speaker_names[key]
                self._auto_named.discard(key)
            self._system_speaker_order = []
            for u in targets:
                new_id = assignments[id(u)]
                if new_id not in self._system_speaker_order:
                    self._system_speaker_order.append(new_id)
            # A custom (user-typed) name goes to the single new speaker that took over
            # most of that old speaker's utterances.
            carried: dict[int, str] = {}
            old_keys = {k for v in votes.values() for k in v}
            for old_key in old_keys:
                if old_key not in old_names or old_key in old_auto:
                    continue
                winner = max(votes, key=lambda nid: votes[nid].get(old_key, 0))
                carried.setdefault(winner, old_names[old_key])
            for n, new_id in enumerate(self._system_speaker_order, start=1):
                key: SpeakerKey = ("system", new_id)
                if new_id in carried:
                    self._speaker_names[key] = carried[new_id]
                else:
                    self._speaker_names[key] = f"Собеседник {n}"
                    self._auto_named.add(key)
            for u in targets:
                u.speaker_key = ("system", assignments[id(u)])

    def final_utterances(self) -> list[Utterance]:
        """All is_final=True utterances, in insertion order."""
        with self._lock:
            return [u for u in self._utterances if u.is_final]
