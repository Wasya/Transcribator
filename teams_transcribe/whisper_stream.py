import queue
import threading
from datetime import datetime
from typing import Callable, Optional

from teams_transcribe.audio_utils import Resampler, Segmenter, pcm16_to_float32


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

    def load(self) -> None:
        if self.device == "cuda":
            try:
                # Importing torch first lets ctranslate2 find the CUDA DLLs bundled with it.
                import torch  # noqa: F401
            except ImportError:
                pass
        from faster_whisper import WhisperModel

        self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)

    def transcribe(self, audio, language: str) -> str:
        lang = None if language == "multi" else language
        with self._lock:
            segments, _info = self._model.transcribe(
                audio,
                language=lang,
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            return " ".join(s.text.strip() for s in segments).strip()


class WhisperStream:
    """Recognizer for one mono PCM16 source, matching the DeepgramStream interface.

    Audio is cut into utterances at pauses; each finished utterance is transcribed
    on a worker thread and reported as a final result (no interim results).
    """

    def __init__(
        self,
        engine: WhisperEngine,
        *,
        language: str,
        sample_rate: int,
        on_result: Callable[[float, float, bool, Optional[int], str], None],
        on_error: Optional[Callable[[Exception], None]] = None,
        speaker_identifier: Optional[Callable] = None,
    ):
        self._engine = engine
        self._language = language
        self._on_result = on_result
        self._on_error = on_error
        self._identifier = speaker_identifier
        self._resampler = Resampler(sample_rate)
        self._segmenter = Segmenter()
        self._jobs: "queue.Queue" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._started_at: Optional[datetime] = None

    @property
    def started_at(self) -> Optional[datetime]:
        return self._started_at

    def start(self) -> None:
        self._started_at = datetime.now()
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()

    def send(self, pcm16_bytes: bytes) -> None:
        samples = self._resampler.process(pcm16_to_float32(pcm16_bytes))
        for segment in self._segmenter.feed(samples):
            self._jobs.put(segment)

    def stop(self) -> None:
        """Flush the pending utterance, wait for the worker to finish the backlog."""
        for segment in self._segmenter.flush():
            self._jobs.put(segment)
        self._jobs.put(None)
        if self._worker is not None:
            self._worker.join(timeout=300)
            self._worker = None

    def _work(self) -> None:
        while True:
            segment = self._jobs.get()
            if segment is None:
                return
            try:
                text = self._engine.transcribe(segment.audio, self._language)
                if not text:
                    continue
                speaker_id = self._identifier(segment.audio) if self._identifier else None
                self._on_result(segment.start, segment.end - segment.start, True, speaker_id, text)
            except Exception as exc:
                if self._on_error is not None:
                    self._on_error(exc)
