import queue
import threading
import time
import wave
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Optional

import pyaudiowpatch as pyaudio
from teams_transcribe.audio_capture import DeviceInfo, MicCapture, SystemCapture
from teams_transcribe.transcript_store import SpeakerKey, TranscriptStore

# Diarization modes for the WhisperX backend.
DIARIZE_OFF = "off"
DIARIZE_POST = "post"  # variant A: pyannote over system.wav after Stop
DIARIZE_LIVE = "live"  # variant B: online voice clustering while recording
DIARIZE_BOTH = "both"  # B during the call, then A to refine the final transcript

BACKEND_DEEPGRAM = "deepgram"
BACKEND_WHISPERX = "whisperx"


def _open_wav_writer(path: Path, sample_rate: int) -> wave.Wave_write:
    writer = wave.open(str(path), "wb")
    writer.setnchannels(1)
    writer.setsampwidth(2)  # 16-bit PCM, matches pyaudio.paInt16
    writer.setframerate(sample_rate)
    return writer


def _endpointing_for_language(language: str) -> int:
    return 100 if language == "multi" else 10


@dataclass
class SessionOptions:
    backend: str = BACKEND_DEEPGRAM
    language: str = "ru"
    record_audio: bool = False
    output_dir: Optional[Path] = None
    deepgram_api_key: Optional[str] = None
    # WhisperX backend only:
    hf_token: Optional[str] = None
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    diarization: str = DIARIZE_POST


@dataclass
class UIEvent:
    kind: str
    speaker_key: SpeakerKey
    speaker_name: str
    text: str = ""
    is_final: bool = False


class TranscriptionSession:
    """Owns mic + system capture, both Deepgram connections, and the TranscriptStore
    for one recording session. Call start()/stop() from the GUI thread only."""

    def __init__(
        self,
        mic_device: DeviceInfo,
        system_device: DeviceInfo,
        options: SessionOptions,
    ):
        self.store = TranscriptStore()
        self.events: "queue.Queue[UIEvent]" = queue.Queue()
        self.options = options
        self._mic_device = mic_device
        self._system_device = system_device
        self._language = options.language
        self._output_dir = options.output_dir
        whisper = options.backend == BACKEND_WHISPERX
        # Variant A needs the system audio on disk; the WAV is deleted after
        # post-processing unless the user asked to keep raw audio.
        self.needs_diarization_pass = whisper and options.diarization in (DIARIZE_POST, DIARIZE_BOTH)
        self._record_audio = options.record_audio
        self._record_system_wav = options.record_audio or self.needs_diarization_pass
        self._client = None
        self._engine = None
        self._identifier = None
        self._mic_queue: "queue.Queue[bytes]" = queue.Queue()
        self._system_queue: "queue.Queue[bytes]" = queue.Queue()
        self._pa: Optional[pyaudio.PyAudio] = None
        self._mic_capture: Optional[MicCapture] = None
        self._system_capture: Optional[SystemCapture] = None
        self._mic_stream: Optional[Any] = None
        self._system_stream: Optional[Any] = None
        self._mic_wav: Optional[wave.Wave_write] = None
        self._system_wav: Optional[wave.Wave_write] = None
        self._sender_threads: list[threading.Thread] = []
        self._stop_senders = threading.Event()
        # Copied from the streams at start(), so results that still trickle in
        # after stop() has released the streams can be timestamped.
        self._mic_started_at = None
        self._system_started_at = None

    def _mic_result(self, start, duration, is_final, speaker_id, text):
        key: SpeakerKey = ("mic", None)
        name = self.store.speaker_name(key)
        ts = self._mic_started_at + timedelta(seconds=start)
        self.store.add_utterance(key, ts, is_final, text, start, duration)
        self.events.put(UIEvent(kind="utterance", speaker_key=key, speaker_name=name, text=text, is_final=is_final))

    def _system_result(self, start, duration, is_final, speaker_id, text):
        sid = speaker_id if speaker_id is not None else 0
        key: SpeakerKey = ("system", sid)
        name, is_new = self.store.get_or_create_system_speaker_name(sid)
        if is_new:
            self.events.put(UIEvent(kind="new_speaker", speaker_key=key, speaker_name=name))
        ts = self._system_started_at + timedelta(seconds=start)
        self.store.add_utterance(key, ts, is_final, text, start, duration)
        self.events.put(UIEvent(kind="utterance", speaker_key=key, speaker_name=name, text=text, is_final=is_final))

    def _sender_loop(
        self,
        chunk_queue: "queue.Queue[bytes]",
        stream: Any,
        wav_writer: Optional[wave.Wave_write],
    ) -> None:
        while not self._stop_senders.is_set():
            try:
                chunk = chunk_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                stream.send(chunk)
                if wav_writer is not None:
                    wav_writer.writeframes(chunk)
            except Exception as exc:
                self._on_stream_error(exc)
                break

    def _on_stream_error(self, exc: Exception) -> None:
        self.events.put(
            UIEvent(kind="error", speaker_key=("error", None), speaker_name="", text=str(exc))
        )

    @staticmethod
    def _drain_queue(
        chunk_queue: "queue.Queue[bytes]",
        stream: Any,
        wav_writer: Optional[wave.Wave_write],
    ) -> None:
        """Non-blocking drain of whatever's left in chunk_queue, sending each
        chunk (and writing it to wav_writer, if open) before the connection is
        torn down, so trailing audio isn't silently dropped on stop(). Must
        only be called after the corresponding sender thread has been joined,
        so nothing else is concurrently pulling from chunk_queue or calling
        the unsynchronized stream.send()."""
        while True:
            try:
                chunk = chunk_queue.get_nowait()
            except queue.Empty:
                return
            try:
                stream.send(chunk)
                if wav_writer is not None:
                    wav_writer.writeframes(chunk)
            except Exception:
                return

    def prepare(self, progress: Optional[Callable[[str], None]] = None) -> None:
        """Heavy, GUI-independent setup (model loading / client creation). Safe to
        call from a worker thread; must complete before start()."""
        opts = self.options
        say = progress or (lambda _text: None)
        if opts.backend == BACKEND_DEEPGRAM:
            from deepgram import DeepgramClient

            say("Подключение к Deepgram…")
            self._client = DeepgramClient(api_key=opts.deepgram_api_key)
            return
        from teams_transcribe.whisper_stream import get_engine

        # Models are cached across sessions (process-wide), so a second "Start"
        # with the same settings skips the slow load.
        say(f"Загрузка модели распознавания речи ({opts.whisper_model}, {opts.whisper_device})…")
        self._engine = get_engine(opts.whisper_model, opts.whisper_device, opts.whisper_compute_type)
        if opts.diarization in (DIARIZE_LIVE, DIARIZE_BOTH):
            from teams_transcribe.speaker_id import OnlineSpeakerIdentifier, get_embedder

            say("Загрузка модели распознавания голосов…")
            # The identifier itself (speaker centroids) is fresh for every session.
            self._identifier = OnlineSpeakerIdentifier(get_embedder(opts.hf_token, opts.whisper_device))

    def _make_stream(self, device: DeviceInfo, *, diarize: bool, endpointing: int, on_result):
        if self.options.backend == BACKEND_DEEPGRAM:
            from teams_transcribe.deepgram_stream import DeepgramStream

            return DeepgramStream(
                self._client, model="nova-3", language=self._language,
                sample_rate=device.sample_rate, diarize=diarize,
                endpointing=endpointing, on_result=on_result,
                on_error=self._on_stream_error,
            )
        from teams_transcribe.whisper_stream import WhisperStream

        return WhisperStream(
            self._engine, language=self._language, sample_rate=device.sample_rate,
            on_result=on_result, on_error=self._on_stream_error,
            speaker_identifier=self._identifier if diarize else None,
            label="system" if diarize else "mic",
        )

    def start(self) -> None:
        self.store.register_mic()
        self._pa = pyaudio.PyAudio()
        endpointing = _endpointing_for_language(self._language)

        if self._record_audio or self._record_system_wav:
            assert self._output_dir is not None, "output_dir is required when recording audio"
        if self._record_audio:
            self._mic_wav = _open_wav_writer(self._output_dir / "mic.wav", self._mic_device.sample_rate)
        if self._record_system_wav:
            self._system_wav = _open_wav_writer(self._output_dir / "system.wav", self._system_device.sample_rate)

        self._mic_stream = self._make_stream(
            self._mic_device, diarize=False, endpointing=endpointing, on_result=self._mic_result
        )
        self._mic_stream.start()
        self._mic_started_at = self._mic_stream.started_at

        self._system_stream = self._make_stream(
            self._system_device, diarize=True, endpointing=endpointing, on_result=self._system_result
        )
        self._system_stream.start()
        self._system_started_at = self._system_stream.started_at

        self._mic_capture = MicCapture(self._pa, self._mic_device, self._mic_queue)
        self._mic_capture.start()
        self._system_capture = SystemCapture(self._pa, self._system_device, self._system_queue)
        self._system_capture.start()

        self._stop_senders.clear()
        t1 = threading.Thread(
            target=self._sender_loop, args=(self._mic_queue, self._mic_stream, self._mic_wav), daemon=True
        )
        t2 = threading.Thread(
            target=self._sender_loop, args=(self._system_queue, self._system_stream, self._system_wav), daemon=True
        )
        t1.start()
        t2.start()
        self._sender_threads = [t1, t2]

    def set_muted(self, muted: bool) -> None:
        if self._mic_capture is not None:
            self._mic_capture.set_muted(muted)

    @property
    def mic_level(self) -> float:
        return self._mic_capture.level if self._mic_capture is not None else 0.0

    @property
    def system_level(self) -> float:
        return self._system_capture.level if self._system_capture is not None else 0.0

    @property
    def backlog_seconds(self) -> float:
        """Speech waiting to be recognized (WhisperX only; 0 for Deepgram): how far
        the transcript lags behind real time."""
        return sum(
            getattr(s, "backlog_seconds", 0.0) for s in (self._mic_stream, self._system_stream) if s is not None
        )

    def rename_speaker(self, speaker_key: SpeakerKey, new_name: str) -> None:
        self.store.rename(speaker_key, new_name)

    def stop(self) -> None:
        # Stop audio capture first so no new chunks get enqueued while we
        # tear down the senders.
        if self._mic_capture is not None:
            self._mic_capture.stop()
        if self._system_capture is not None:
            self._system_capture.stop()

        # Signal the sender loops to stop and wait for them to actually
        # finish pulling from the queues before we touch those queues
        # ourselves - otherwise the drain below and a still-live sender
        # thread could concurrently pull from the same queue.Queue and both
        # call the unsynchronized stream.send() on the same socket.
        self._stop_senders.set()
        for t in self._sender_threads:
            t.join(timeout=2)

        # Now that the sender threads are guaranteed done, drain whatever's
        # left in the queues (little to nothing after the join, but anything
        # queued between a sender's last get() and the capture stop still
        # needs to go out) so trailing audio isn't silently discarded.
        if self._mic_stream is not None:
            self._drain_queue(self._mic_queue, self._mic_stream, self._mic_wav)
        if self._system_stream is not None:
            self._drain_queue(self._system_queue, self._system_stream, self._system_wav)

        # Brief grace period so trailing is_final results the API is still
        # computing have a chance to arrive before we close the connections.
        if self.options.backend == BACKEND_DEEPGRAM and (
            self._mic_stream is not None or self._system_stream is not None
        ):
            time.sleep(0.5)

        if self._mic_stream is not None:
            self._mic_stream.stop()
        if self._system_stream is not None:
            self._system_stream.stop()
        if self._mic_wav is not None:
            self._mic_wav.close()
            self._mic_wav = None
        if self._system_wav is not None:
            self._system_wav.close()
            self._system_wav = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
        # The finished session object outlives stop() (the GUI keeps it for
        # re-export), so drop every reference to the models here; otherwise a
        # model replaced in the cache would stay in GPU memory.
        self._mic_stream = None
        self._system_stream = None
        self._engine = None
        self._identifier = None
        self._client = None
