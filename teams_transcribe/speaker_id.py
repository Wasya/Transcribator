import threading
from typing import Callable, Optional

import numpy as np

from teams_transcribe.audio_utils import TARGET_RATE

EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"


class OnlineSpeakerClusterer:
    """Incremental cosine-similarity clustering of voice embeddings.

    Each embedding joins the most similar known speaker (if similarity >= threshold)
    or opens a new one. Speaker ids are 0, 1, 2... in order of first appearance.
    """

    def __init__(self, threshold: float = 0.5, max_speakers: int = 12):
        self._threshold = threshold
        self._max_speakers = max_speakers
        self._centroids: list[np.ndarray] = []
        self._counts: list[int] = []

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else v

    def nearest(self, embedding: np.ndarray) -> tuple[Optional[int], float]:
        if not self._centroids:
            return None, -1.0
        e = self._unit(embedding)
        sims = [float(np.dot(e, self._unit(c))) for c in self._centroids]
        best = int(np.argmax(sims))
        return best, sims[best]

    def assign(self, embedding: np.ndarray, update: bool = True) -> int:
        best, sim = self.nearest(embedding)
        e = self._unit(embedding)
        if best is not None and (sim >= self._threshold or len(self._centroids) >= self._max_speakers):
            if update:
                n = self._counts[best]
                self._centroids[best] = (self._centroids[best] * n + e) / (n + 1)
                self._counts[best] = n + 1
            return best
        self._centroids.append(e.copy())
        self._counts.append(1)
        return len(self._centroids) - 1


class PyannoteEmbedder:
    """Voice embeddings via pyannote (lazy-loaded, thread-safe)."""

    def __init__(self, hf_token: Optional[str], device: str):
        self._hf_token = hf_token
        self._device = device
        self._inference = None
        self._lock = threading.Lock()

    @property
    def key(self) -> tuple[Optional[str], str]:
        return (self._hf_token, self._device)

    def load(self) -> None:
        """Load the embedding model; a no-op if it is already loaded."""
        with self._lock:
            if self._inference is not None:
                return
            import torch  # noqa: F401  (imported first so CUDA DLLs resolve before other native libs)
            from pyannote.audio import Inference, Model

            model = Model.from_pretrained(EMBEDDING_MODEL, token=self._hf_token)
            inference = Inference(model, window="whole")
            inference.to(torch.device(self._device))
            self._inference = inference

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        import torch

        with self._lock:
            waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
            emb = self._inference({"waveform": waveform, "sample_rate": TARGET_RATE})
        return np.asarray(emb, dtype=np.float32).reshape(-1)


_embedder_cache_lock = threading.Lock()
_cached_embedder: Optional[PyannoteEmbedder] = None


def get_embedder(hf_token: Optional[str], device: str) -> PyannoteEmbedder:
    """Return a loaded PyannoteEmbedder, reusing the previous session's instance
    when token and device are unchanged (see whisper_stream.get_engine)."""
    global _cached_embedder
    with _embedder_cache_lock:
        embedder = _cached_embedder
        if embedder is None or embedder.key != (hf_token, device):
            embedder = PyannoteEmbedder(hf_token, device)
            _cached_embedder = embedder
    embedder.load()
    return embedder


class OnlineSpeakerIdentifier:
    """Maps an audio segment to a speaker id. Short segments (too little voice
    for a reliable embedding) are attributed to the previous speaker without
    updating any centroid."""

    def __init__(
        self,
        embed: Callable[[np.ndarray], np.ndarray],
        clusterer: Optional[OnlineSpeakerClusterer] = None,
        min_seconds: float = 1.2,
    ):
        self._embed = embed
        self._clusterer = clusterer or OnlineSpeakerClusterer()
        self._min_samples = int(min_seconds * TARGET_RATE)
        self._last: Optional[int] = None

    def __call__(self, audio: np.ndarray) -> Optional[int]:
        if len(audio) < self._min_samples:
            return self._last
        speaker = self._clusterer.assign(self._embed(audio))
        self._last = speaker
        return speaker
