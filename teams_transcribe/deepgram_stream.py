import threading
from datetime import datetime
from typing import Callable, Optional

from deepgram import DeepgramClient
from deepgram.core.events import EventType


class DeepgramStream:
    """Wraps one Deepgram live connection for a single mono audio source."""

    def __init__(
        self,
        client: DeepgramClient,
        *,
        model: str,
        language: str,
        sample_rate: int,
        diarize: bool,
        endpointing: int,
        on_result: Callable[[float, float, bool, Optional[int], str], None],
    ):
        self._client = client
        self._model = model
        self._language = language
        self._sample_rate = sample_rate
        self._diarize = diarize
        self._endpointing = endpointing
        self._on_result = on_result
        self._connection_ctx = None
        self._connection = None
        self._started_at: Optional[datetime] = None
        self._listen_thread: Optional[threading.Thread] = None

    @property
    def started_at(self) -> Optional[datetime]:
        return self._started_at

    def start(self) -> None:
        self._connection_ctx = self._client.listen.v1.connect(
            model=self._model,
            language=self._language,
            encoding="linear16",
            sample_rate=self._sample_rate,
            channels=1,
            diarize=self._diarize,
            smart_format=True,
            interim_results=True,
            endpointing=self._endpointing,
        )
        self._connection = self._connection_ctx.__enter__()
        self._started_at = datetime.now()

        def on_message(result):
            # Defensive: an unexpected/malformed message or an exception raised
            # inside _on_result must not kill the SDK's listener thread silently.
            try:
                if getattr(result, "type", None) != "Results":
                    return
                alt = result.channel.alternatives[0]
                text = alt.transcript
                if not text:
                    return
                speaker_id = None
                if alt.words:
                    speaker_id = alt.words[0].speaker
                self._on_result(result.start, result.duration, bool(result.is_final), speaker_id, text)
            except Exception:
                pass

        self._connection.on(EventType.MESSAGE, on_message)

        self._listen_thread = threading.Thread(target=self._connection.start_listening, daemon=True)
        self._listen_thread.start()

    def send(self, pcm16_bytes: bytes) -> None:
        if self._connection is not None:
            self._connection.send_media(pcm16_bytes)

    def stop(self) -> None:
        if self._connection_ctx is not None:
            self._connection_ctx.__exit__(None, None, None)
            self._connection_ctx = None
            self._connection = None
