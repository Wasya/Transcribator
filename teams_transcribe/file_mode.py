"""Offline transcription of an audio file through the same pipeline as a live session.

Purpose: reproducible experiments. The same file, run with different models or
settings, makes differences in the result attributable to the change under test
instead of to a different broadcast/call.
"""
import difflib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from teams_transcribe.session import DIARIZE_BOTH, DIARIZE_LIVE, DIARIZE_OFF, DIARIZE_POST
from teams_transcribe.transcript_store import TranscriptStore

# Timestamps in the export then read as elapsed time in the file: [00:01:23].
BASE_TIME = datetime(2000, 1, 1, 0, 0, 0)
CHUNK_SAMPLES = 2048


def load_audio_16k(path: Path) -> np.ndarray:
    """Decode any audio file (wav, mp3, m4a, ogg, ...) to 16 kHz mono float32."""
    import av

    container = av.open(str(path))
    try:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        chunks: list[np.ndarray] = []
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    finally:
        container.close()
    if not chunks:
        raise ValueError(f"В файле нет звука: {path}")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def seed_everything(seed: int) -> None:
    """Best effort: fix the random choices Whisper makes when it retries a poor segment
    at a higher temperature, so repeated runs of one file give the same text."""
    np.random.seed(seed)
    try:
        import ctranslate2

        ctranslate2.set_random_seed(seed)
    except Exception:  # noqa: BLE001 - not installed: nothing to seed
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:  # noqa: BLE001
        pass


@dataclass
class FileRunOptions:
    model: str
    device: str
    compute_type: str
    language: str = "ru"
    diarization: str = DIARIZE_POST
    hf_token: Optional[str] = None
    max_segment: Optional[float] = None
    use_context: bool = False
    seed: int = 0
    vad_threshold: Optional[float] = 0.35  # None = Whisper's internal speech filter off


@dataclass
class FileRunResult:
    store: TranscriptStore
    audio_seconds: float
    wall_seconds: float
    segments: int
    words: int
    speakers: int
    options: FileRunOptions
    notes: list[str] = field(default_factory=list)

    @property
    def realtime_factor(self) -> float:
        return self.wall_seconds / self.audio_seconds if self.audio_seconds else 0.0

    def text(self) -> str:
        utterances = sorted(self.store.final_utterances(), key=lambda u: u.timestamp)
        return " ".join(u.text for u in utterances)


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def transcribe_audio(
    audio: np.ndarray,
    opts: FileRunOptions,
    progress: Callable[[str], None] = lambda _t: None,
    engine=None,
    identifier=None,
    diarizer=None,
) -> FileRunResult:
    """Run the live pipeline (segmenter -> Whisper -> optional speaker labelling) on a
    16 kHz mono signal. engine/identifier/diarizer can be injected (tests)."""
    from teams_transcribe.postprocess import diarize_audio, run_diarization
    from teams_transcribe.whisper_stream import WhisperStream, get_engine

    notes: list[str] = []
    seed_everything(opts.seed)
    if engine is None:
        progress(f"Загрузка модели {opts.model} ({opts.device}, {opts.compute_type})…")
        engine = get_engine(opts.model, opts.device, opts.compute_type)
    engine.vad_threshold = opts.vad_threshold
    live_ids = opts.diarization in (DIARIZE_LIVE, DIARIZE_BOTH)
    post_ids = opts.diarization in (DIARIZE_POST, DIARIZE_BOTH)
    if live_ids and identifier is None:
        from teams_transcribe.speaker_id import OnlineSpeakerIdentifier, get_embedder

        progress("Загрузка модели распознавания голосов…")
        identifier = OnlineSpeakerIdentifier(get_embedder(opts.hf_token, opts.device))

    store = TranscriptStore()
    count = 0

    def on_result(start, duration, is_final, speaker_id, text):
        nonlocal count
        sid = speaker_id if speaker_id is not None else 0
        key = ("system", sid)
        store.get_or_create_system_speaker_name(sid)
        store.add_utterance(key, BASE_TIME + timedelta(seconds=start), True, text, start, duration)
        count += 1

    errors: list[Exception] = []
    stream = WhisperStream(
        engine, language=opts.language, sample_rate=16000, on_result=on_result,
        on_error=errors.append, speaker_identifier=identifier if live_ids else None,
        label="file", use_context=opts.use_context, max_segment=opts.max_segment,
    )
    began = time.monotonic()
    stream.start()
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    step = CHUNK_SAMPLES * 2
    next_report = 0.1
    for i in range(0, len(pcm), step):
        stream.send(pcm[i:i + step])
        done = i / len(pcm)
        if done >= next_report:
            progress(f"Нарезка и распознавание: {int(done * 100)}%  (в очереди {stream.backlog_seconds:.0f} с)")
            next_report += 0.1
    progress("Доработка очереди распознавания…")
    stream.stop()
    if errors:
        raise RuntimeError(f"Ошибка распознавания: {errors[0]}") from errors[0]

    if post_ids:
        if opts.hf_token or diarizer is not None:
            progress("Разбор по голосам (вариант A)…")
            diarize_audio(audio, store, opts.hf_token, opts.device, diarizer=diarizer or run_diarization)
        else:
            notes.append("Разбор по голосам пропущен: не задан HF_TOKEN")
    wall = time.monotonic() - began

    speakers = len({u.speaker_key for u in store.final_utterances()})
    result = FileRunResult(
        store=store, audio_seconds=len(audio) / 16000, wall_seconds=wall, segments=count,
        words=0, speakers=speakers, options=opts, notes=notes,
    )
    result.words = count_words(result.text())
    return result


_PREFIX = re.compile(r"^\s*\[\d{1,2}:\d{2}(?::\d{2})?\]\s*[^:\n]{1,40}:\s*")


def load_reference_text(path: Path) -> str:
    """Reference text: a plain text file, or an earlier transcript.txt (the
    "[hh:mm:ss] Имя:" line prefixes are stripped)."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return " ".join(_PREFIX.sub("", line) for line in lines)


def compare_texts(reference: str, hypothesis: str) -> dict:
    """Word-level comparison. 'wer' is approximate: (substituted + deleted + inserted
    words) / reference words, derived from the diff alignment."""
    ref = re.findall(r"\w+", reference.lower())
    hyp = re.findall(r"\w+", hypothesis.lower())
    matcher = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    subs = dels = inss = 0
    missing: list[str] = []
    extra: list[str] = []
    for tag, a1, a2, b1, b2 in matcher.get_opcodes():
        if tag == "replace":
            subs += min(a2 - a1, b2 - b1)
            dels += max(0, (a2 - a1) - (b2 - b1))
            inss += max(0, (b2 - b1) - (a2 - a1))
        elif tag == "delete":
            dels += a2 - a1
        elif tag == "insert":
            inss += b2 - b1
        if tag in ("delete", "replace") and a2 - a1 >= 3:
            missing.append(" ".join(ref[a1:a2]))
        if tag in ("insert", "replace") and b2 - b1 >= 3:
            extra.append(" ".join(hyp[b1:b2]))
    n = max(1, len(ref))
    return {
        "similarity": matcher.ratio(),
        "wer": (subs + dels + inss) / n,
        "reference_words": len(ref),
        "hypothesis_words": len(hyp),
        "missing": missing,
        "extra": extra,
    }
