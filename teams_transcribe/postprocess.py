import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from teams_transcribe.audio_utils import Resampler, TARGET_RATE, pcm16_to_float32
from teams_transcribe.transcript_store import TranscriptStore, Utterance

# (start, end, speaker label) turns produced by a diarization pipeline.
Turn = tuple[float, float, str]


def read_wav_16k(path: Path) -> np.ndarray:
    """Read a mono PCM16 WAV written by the session and resample it to 16 kHz float32."""
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    return Resampler(rate).process(pcm16_to_float32(raw))


def assign_speakers(utterances: list[Utterance], turns: list[Turn]) -> dict[int, int]:
    """Map id(utterance) -> speaker number (0, 1, ... by first appearance in time).

    Each utterance goes to the speaker whose turns overlap it the most; with no
    overlap it falls back to the temporally nearest turn.
    """
    if not turns:
        return {}
    order: dict[str, int] = {}
    for _s, _e, label in sorted(turns):
        order.setdefault(label, len(order))

    assignments: dict[int, int] = {}
    for u in utterances:
        if u.start is None:
            continue
        u_start = u.start
        u_end = u.start + (u.duration or 0.0)
        overlap: dict[str, float] = {}
        for s, e, label in turns:
            ov = min(u_end, e) - max(u_start, s)
            if ov > 0:
                overlap[label] = overlap.get(label, 0.0) + ov
        if overlap:
            label = max(overlap, key=overlap.get)
        else:
            mid = (u_start + u_end) / 2
            label = min(turns, key=lambda t: 0.0 if t[0] <= mid <= t[1] else min(abs(t[0] - mid), abs(t[1] - mid)))[2]
        assignments[id(u)] = order[label]
    return assignments


def run_diarization(audio_16k: np.ndarray, hf_token: Optional[str], device: str) -> list[Turn]:
    """pyannote diarization through WhisperX (needs a Hugging Face token)."""
    import torch  # noqa: F401  (first, so CUDA DLLs resolve before other native libs)
    from whisperx.diarize import DiarizationPipeline

    pipeline = DiarizationPipeline(token=hf_token, device=device)
    frame = pipeline(audio_16k)
    return [(float(r.start), float(r.end), str(r.speaker)) for r in frame.itertuples()]


def diarize_system_wav(
    wav_path: Path,
    store: TranscriptStore,
    hf_token: Optional[str],
    device: str,
    progress: Optional[Callable[[str], None]] = None,
    diarizer: Callable[[np.ndarray, Optional[str], str], list[Turn]] = run_diarization,
) -> int:
    """Variant A: label the system channel by speaker using the recorded system.wav.

    Returns the number of distinct speakers found (0 if nothing could be done).
    """
    utterances = store.system_utterances()
    if not utterances:
        return 0
    if progress:
        progress("Чтение записи системного звука…")
    audio = read_wav_16k(wav_path)
    if progress:
        progress("Разбор по голосам (pyannote)…")
    turns = diarizer(audio, hf_token, device)
    assignments = assign_speakers(utterances, turns)
    if not assignments:
        return 0
    store.apply_diarization(assignments)
    return len(set(assignments.values()))
