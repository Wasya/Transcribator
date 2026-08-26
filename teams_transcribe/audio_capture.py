import array
import queue
import threading
from dataclasses import dataclass

import pyaudiowpatch as pyaudio

CHUNK_FRAMES = 1024


@dataclass
class DeviceInfo:
    index: int
    name: str
    sample_rate: int
    channels: int


def list_input_devices(p: "pyaudio.PyAudio") -> list[DeviceInfo]:
    """Regular WASAPI microphones (loopback pseudo-devices excluded)."""
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    loopback_indices = {d["index"] for d in p.get_loopback_device_info_generator()}
    devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["hostApi"] != wasapi["index"]:
            continue
        if info["maxInputChannels"] <= 0:
            continue
        if info["index"] in loopback_indices:
            continue
        devices.append(
            DeviceInfo(
                index=info["index"],
                name=info["name"],
                sample_rate=int(info["defaultSampleRate"]),
                channels=1,
            )
        )
    return devices


def list_loopback_devices(p: "pyaudio.PyAudio") -> list[DeviceInfo]:
    """WASAPI loopback devices (system audio output, captured as input)."""
    return [
        DeviceInfo(
            index=d["index"],
            name=d["name"],
            sample_rate=int(d["defaultSampleRate"]),
            channels=d["maxInputChannels"],
        )
        for d in p.get_loopback_device_info_generator()
    ]


def _downmix_to_mono(pcm16_bytes: bytes, channels: int) -> bytes:
    """Average N interleaved int16 channels down to mono int16 PCM."""
    if channels == 1:
        return pcm16_bytes
    samples = array.array("h")
    samples.frombytes(pcm16_bytes)
    frame_count = len(samples) // channels
    mono = array.array("h", (0 for _ in range(frame_count)))
    for i in range(frame_count):
        total = 0
        for c in range(channels):
            total += samples[i * channels + c]
        mono[i] = int(total / channels)
    return mono.tobytes()


class MicCapture:
    """Captures mono PCM16 audio from a microphone in a background (callback) thread."""

    def __init__(self, p: "pyaudio.PyAudio", device: DeviceInfo, chunk_queue: "queue.Queue[bytes]"):
        self._p = p
        self._device = device
        self._queue = chunk_queue
        self._stream = None
        self._muted = threading.Event()

    def start(self) -> None:
        def callback(in_data, frame_count, time_info, status):
            if self._muted.is_set():
                in_data = b"\x00" * len(in_data)
            self._queue.put(in_data)
            return (None, pyaudio.paContinue)

        self._stream = self._p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._device.sample_rate,
            input=True,
            input_device_index=self._device.index,
            frames_per_buffer=CHUNK_FRAMES,
            stream_callback=callback,
        )
        self._stream.start_stream()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

    def set_muted(self, muted: bool) -> None:
        if muted:
            self._muted.set()
        else:
            self._muted.clear()


class SystemCapture:
    """Captures loopback (system output) audio, downmixed to mono PCM16, in a background thread."""

    def __init__(self, p: "pyaudio.PyAudio", device: DeviceInfo, chunk_queue: "queue.Queue[bytes]"):
        self._p = p
        self._device = device
        self._queue = chunk_queue
        self._stream = None

    def start(self) -> None:
        def callback(in_data, frame_count, time_info, status):
            mono = _downmix_to_mono(in_data, self._device.channels)
            self._queue.put(mono)
            return (None, pyaudio.paContinue)

        self._stream = self._p.open(
            format=pyaudio.paInt16,
            channels=self._device.channels,
            rate=self._device.sample_rate,
            input=True,
            input_device_index=self._device.index,
            frames_per_buffer=CHUNK_FRAMES,
            stream_callback=callback,
        )
        self._stream.start_stream()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
