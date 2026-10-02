from dataclasses import dataclass
from typing import Optional

import numpy as np

TARGET_RATE = 16000


def pcm16_to_float32(pcm16_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0


class Resampler:
    """Stateful mono resampler to 16 kHz (box-filter anti-aliasing + linear interpolation).

    Good enough for speech recognition; keeps phase across chunks so consecutive
    chunks join without clicks.
    """

    def __init__(self, source_rate: int, target_rate: int = TARGET_RATE):
        self._ratio = source_rate / target_rate
        self._passthrough = source_rate == target_rate
        self._tail = np.zeros(0, dtype=np.float32)
        self._pos = 0.0  # position of the next output sample, in tail-relative input samples

    def process(self, samples: np.ndarray) -> np.ndarray:
        if self._passthrough:
            return samples
        buf = np.concatenate([self._tail, samples])
        width = max(1, int(round(self._ratio)))
        if width > 1:
            kernel = np.ones(width, dtype=np.float32) / width
            filtered = np.convolve(buf, kernel, mode="same")
        else:
            filtered = buf
        count = int(np.floor((len(filtered) - 1 - self._pos) / self._ratio)) + 1
        if count <= 0:
            self._tail = buf
            return np.zeros(0, dtype=np.float32)
        positions = self._pos + np.arange(count) * self._ratio
        out = np.interp(positions, np.arange(len(filtered)), filtered).astype(np.float32)
        next_pos = self._pos + count * self._ratio
        keep_from = min(int(np.floor(next_pos)), len(buf))
        self._tail = buf[keep_from:]
        self._pos = next_pos - keep_from
        return out


@dataclass
class Segment:
    audio: np.ndarray  # float32, 16 kHz mono
    start: float  # seconds from the start of the stream's audio
    end: float


class Segmenter:
    """Energy-based speech segmenter: splits a 16 kHz mono stream into utterances at pauses.

    Feed arbitrary-sized chunks with feed(); finished segments are returned as they
    complete. flush() emits whatever speech is still pending (call it on stop).
    """

    FRAME = 480  # 30 ms at 16 kHz

    def __init__(
        self,
        *,
        min_silence: float = 0.7,
        min_speech: float = 0.25,
        max_segment: float = 20.0,
        preroll: float = 0.3,
        min_threshold: float = 0.006,
        noise_factor: float = 3.0,
    ):
        self._min_silence_frames = max(1, int(min_silence * TARGET_RATE / self.FRAME))
        self._min_speech_frames = max(1, int(min_speech * TARGET_RATE / self.FRAME))
        self._max_frames = int(max_segment * TARGET_RATE / self.FRAME)
        self._preroll_frames = int(preroll * TARGET_RATE / self.FRAME)
        self._min_threshold = min_threshold
        self._noise_factor = noise_factor
        self._noise_floor = 0.0
        self._pending = np.zeros(0, dtype=np.float32)
        self._frame_index = 0  # index of the next frame to be consumed
        self._preroll: list[np.ndarray] = []
        self._frames: list[np.ndarray] = []
        self._start_frame = 0
        self._in_speech = False
        self._silence_run = 0
        self._speech_frames = 0

    def _is_speech(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(frame * frame)))
        threshold = max(self._min_threshold, self._noise_factor * self._noise_floor)
        speech = rms > threshold
        if not speech:
            self._noise_floor = rms if self._noise_floor == 0.0 else 0.95 * self._noise_floor + 0.05 * rms
        return speech

    def _emit(self, trailing_silence: int) -> Optional[Segment]:
        frames = self._frames
        if trailing_silence:
            frames = frames[: len(frames) - trailing_silence + min(trailing_silence, 3)]
        speech = self._speech_frames
        self._frames = []
        self._in_speech = False
        self._silence_run = 0
        self._speech_frames = 0
        if speech < self._min_speech_frames or not frames:
            return None
        audio = np.concatenate(frames)
        start = self._start_frame * self.FRAME / TARGET_RATE
        return Segment(audio=audio, start=start, end=start + len(audio) / TARGET_RATE)

    def feed(self, samples: np.ndarray) -> list[Segment]:
        self._pending = np.concatenate([self._pending, samples])
        out: list[Segment] = []
        while len(self._pending) >= self.FRAME:
            frame = self._pending[: self.FRAME]
            self._pending = self._pending[self.FRAME:]
            speech = self._is_speech(frame)
            if not self._in_speech:
                if speech:
                    self._in_speech = True
                    self._frames = list(self._preroll) + [frame]
                    self._start_frame = self._frame_index - len(self._preroll)
                    self._preroll = []
                    self._silence_run = 0
                    self._speech_frames = 1
                else:
                    self._preroll.append(frame)
                    if len(self._preroll) > self._preroll_frames:
                        self._preroll.pop(0)
            else:
                self._frames.append(frame)
                if speech:
                    self._silence_run = 0
                    self._speech_frames += 1
                else:
                    self._silence_run += 1
                if self._silence_run >= self._min_silence_frames:
                    seg = self._emit(self._silence_run)
                    if seg:
                        out.append(seg)
                elif len(self._frames) >= self._max_frames:
                    seg = self._emit(0)
                    if seg:
                        out.append(seg)
            self._frame_index += 1
        return out

    def flush(self) -> list[Segment]:
        if self._in_speech:
            seg = self._emit(self._silence_run)
            return [seg] if seg else []
        return []
