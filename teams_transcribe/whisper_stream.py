import gc
import logging
import queue
import threading
import time
from datetime import datetime
from typing import Callable, Optional

from teams_transcribe.audio_utils import Resampler, Segmenter, normalize_level, pcm16_to_float32
from teams_transcribe.hallucinations import clean_hallucinations

log = logging.getLogger("teams_transcribe")


class PromptContext:
    """Tail of the text recognized so far, passed to Whisper as initial_prompt.

    Segments are cut every few seconds, often mid-sentence; with the previous words
    as context Whisper finishes a split word correctly and keeps spelling of names
    consistent. Context older than max_age seconds (a pause, another topic) is not used.
    """

    def __init__(self, max_chars: int = 200, max_age: float = 12.0):
        self._max_chars = max_chars
        self._max_age = max_age
        self._tail = ""
        self._last_end = 0.0

    def get(self, segment_start: float) -> Optional[str]:
        if not self._tail or segment_start - self._last_end > self._max_age:
            return None
        return self._tail

    def update(self, text: str, segment_end: float) -> None:
        tail = text[-self._max_chars:]
        if len(text) > self._max_chars and " " in tail:
            tail = tail.split(" ", 1)[1]  # do not start the context with a cut-off word
        self._tail = tail
        self._last_end = segment_end

    def reset(self) -> None:
        self._tail = ""


def strip_prompt_echo(text: str, prompt: Optional[str]) -> str:
    """Whisper sometimes answers a prompt by repeating it (typically on music/noise):
    drop output that is just a piece of the prompt."""
    if not prompt or not text:
        return text
    t = " ".join(text.lower().split())
    p = " ".join(prompt.lower().split())
    if len(t) >= 10 and t in p:
        return ""
    return text


class WhisperEngine:
    """One shared faster-whisper model (the recognizer inside WhisperX).

    transcribe() is serialized with a lock, so the mic and system streams can
    share a single model instance on one GPU.
    """

    def __init__(self, model_name: str, device: str, compute_type: str):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._lock = threading.Lock()
        # Whisper's own speech detector (applied inside each segment). None = off.
        self.vad_threshold: Optional[float] = 0.35

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.model_name, self.device, self.compute_type)

    def load(self) -> None:
        """Load the model; a no-op if it is already loaded."""
        with self._lock:
            if self._model is not None:
                return
            if self.device == "cuda":
                try:
                    # Importing torch first lets ctranslate2 find the CUDA DLLs bundled with it.
                    import torch  # noqa: F401
                except ImportError:
                    pass
            from faster_whisper import WhisperModel

            kwargs = dict(device=self.device, compute_type=self.compute_type)
            try:
                # A model downloaded earlier loads straight from the local cache,
                # skipping Hugging Face's online revision check (slow or hanging
                # on a poor connection).
                self._model = WhisperModel(self.model_name, local_files_only=True, **kwargs)
            except Exception:  # noqa: BLE001 - not cached yet: download it
                self._model = WhisperModel(self.model_name, **kwargs)
            self._warm_up()

    def _warm_up(self) -> None:
        """Run one tiny decode so CUDA kernels/allocations are set up during
        "Загрузка моделей", not on the first real phrase of the call."""
        import numpy as np

        try:
            segments, _info = self._model.transcribe(
                np.zeros(16000, dtype=np.float32), language="en", beam_size=1,
                vad_filter=False, without_timestamps=True,
            )
            list(segments)
        except Exception:  # noqa: BLE001 - warm-up is only an optimization
            pass

    def transcribe(self, audio, language: str, prompt: Optional[str] = None) -> str:
        lang = None if language == "multi" else language
        audio = normalize_level(audio)
        with self._lock:
            segments, _info = self._model.transcribe(
                audio,
                language=lang,
                # Beam search is ~2x slower; on a CPU that is the difference between
                # keeping up with the call and falling ever further behind.
                beam_size=5 if self.device == "cuda" else 1,
                # Radio and calls have music/noise under the voice: with the default 0.5
                # this detector drops quiet or masked speech from inside a segment.
                vad_filter=self.vad_threshold is not None,
                vad_parameters={"threshold": self.vad_threshold} if self.vad_threshold is not None else None,
                condition_on_previous_text=False,
                initial_prompt=prompt,
            )
            # Whisper itself flags passages it thinks contain no speech (music, noise):
            # drop those it is also unsure about, then cut stock hallucinated phrases.
            kept = [s.text.strip() for s in segments if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)]
            return strip_prompt_echo(clean_hallucinations(" ".join(kept)), prompt)


_engine_cache_lock = threading.Lock()
_cached_engine: Optional[WhisperEngine] = None


def release_gpu_memory() -> None:
    """Collect unreachable models and hand torch's cached CUDA blocks back to the
    driver, so the next model (ctranslate2 has its own allocator) can use them.
    With 8 GB cards Windows otherwise silently spills into system RAM, which
    makes recognition many times slower."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def get_engine(model_name: str, device: str, compute_type: str) -> WhisperEngine:
    """Return a loaded WhisperEngine, reusing the one from the previous session
    when the (model, device, compute_type) triple is unchanged. Only one engine
    is kept, and the old one is freed before the new one loads, so switching
    models does not pile up copies in (GPU) memory."""
    global _cached_engine
    with _engine_cache_lock:
        engine = _cached_engine
        if engine is None or engine.key != (model_name, device, compute_type):
            _cached_engine = engine = None
            release_gpu_memory()
            engine = WhisperEngine(model_name, device, compute_type)
            _cached_engine = engine
    engine.load()
    return engine


class WhisperStream:
    """Recognizer for one mono PCM16 source, matching the DeepgramStream interface.

    Audio is cut into utterances at pauses; each finished utterance is transcribed
    on a worker thread and reported as a final result (no interim results).

    Latency = waiting for the pause (or MAX_SEGMENT of non-stop speech) + the
    recognition itself. MAX_SEGMENT is kept short so continuous speech still
    shows up every few seconds; the GUI/export glue the pieces back into one
    line per speaker (transcript_store.merge_utterances).
    """

    MAX_SEGMENT = 8.0  # seconds

    def __init__(
        self,
        engine: WhisperEngine,
        *,
        language: str,
        sample_rate: int,
        on_result: Callable[[float, float, bool, Optional[int], str], None],
        on_error: Optional[Callable[[Exception], None]] = None,
        speaker_identifier: Optional[Callable] = None,
        label: str = "audio",
        use_context: bool = False,
        max_segment: Optional[float] = None,
    ):
        self._engine = engine
        self._language = language
        self._on_result = on_result
        self._on_error = on_error
        self._identifier = speaker_identifier
        self._label = label
        self._last_stats_log = time.monotonic()
        self._resampler = Resampler(sample_rate)
        self._segmenter = Segmenter(max_segment=max_segment or self.MAX_SEGMENT)
        self._jobs: "queue.Queue" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._started_at: Optional[datetime] = None
        self._backlog_lock = threading.Lock()
        self._backlog = 0.0  # seconds of audio queued or being recognized
        # Off by default: on test recordings the prompt did not reliably help and once
        # made Whisper skip whole phrases (see WHISPER_USE_CONTEXT in .env.example).
        self._use_context = use_context
        self._context = PromptContext()

    @property
    def started_at(self) -> Optional[datetime]:
        return self._started_at

    @property
    def backlog_seconds(self) -> float:
        with self._backlog_lock:
            return self._backlog

    def _enqueue(self, segment) -> None:
        segment.queued_at = time.monotonic()
        with self._backlog_lock:
            self._backlog += segment.end - segment.start
        self._jobs.put(segment)

    def start(self) -> None:
        self._started_at = datetime.now()
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()

    def send(self, pcm16_bytes: bytes) -> None:
        samples = self._resampler.process(pcm16_to_float32(pcm16_bytes))
        for segment in self._segmenter.feed(samples):
            self._enqueue(segment)
        now = time.monotonic()
        if now - self._last_stats_log >= 10.0:
            self._last_stats_log = now
            st = self._segmenter.pop_stats()
            # Tells apart "nobody spoke / too quiet" from "speech was there but not recognized".
            log.info(
                "level %s: %.0fs heard, speech frames %.0f%%, rms mean %.4f max %.4f, threshold %.4f (floor %.4f)",
                self._label, st["seconds"], st["speech_pct"], st["rms_mean"], st["rms_max"],
                st["threshold"], st["noise_floor"],
            )

    def stop(self, stall_timeout: float = 120.0) -> None:
        """Flush the pending utterance and wait for the worker to finish the whole
        backlog, however long that takes (a slow CPU may be minutes behind). Only a
        worker that makes no progress at all for stall_timeout seconds is abandoned."""
        for segment in self._segmenter.flush():
            self._enqueue(segment)
        self._jobs.put(None)
        worker = self._worker
        self._worker = None
        if worker is None:
            return
        last_backlog = self.backlog_seconds
        last_change = time.monotonic()
        while worker.is_alive():
            worker.join(timeout=min(2.0, stall_timeout))
            backlog = self.backlog_seconds
            if backlog != last_backlog:
                last_backlog, last_change = backlog, time.monotonic()
            elif time.monotonic() - last_change > stall_timeout:
                log.warning("recognizer made no progress for %.0fs; %.0fs of audio left unrecognized",
                            stall_timeout, backlog)
                break

    def _work(self) -> None:
        while True:
            segment = self._jobs.get()
            if segment is None:
                return
            try:
                began = time.monotonic()
                prompt = self._context.get(segment.start) if self._use_context else None
                text = self._engine.transcribe(segment.audio, self._language, prompt)
                done = time.monotonic()
                # Where the delay goes: "wait" = queued behind other work, "decode" = the
                # model itself; the pause that ended the utterance comes on top of both.
                log.info(
                    "whisper segment: audio=%.1fs wait=%.2fs decode=%.2fs chars=%d",
                    segment.end - segment.start, began - segment.queued_at, done - began, len(text),
                )
                if not text:
                    self._context.reset()
                    continue
                self._context.update(text, segment.end)
                speaker_id = self._identifier(segment.audio) if self._identifier else None
                self._on_result(segment.start, segment.end - segment.start, True, speaker_id, text)
            except Exception as exc:
                if self._on_error is not None:
                    self._on_error(exc)
            finally:
                with self._backlog_lock:
                    self._backlog = max(0.0, self._backlog - (segment.end - segment.start))
