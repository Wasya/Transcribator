# TeamsTranscribe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Windows tkinter GUI app (`TeamsTranscribe.py`) that transcribes a live conversation (e.g. an MS Teams call) in real time via Deepgram, separating "Я" (microphone) from remote participants (system audio, diarized), with live speaker renaming and export to txt/json/per-speaker files.

**Architecture:** Two independent Deepgram live-streaming connections (microphone, mono, `diarize=False`; system-audio loopback, mono after downmix, `diarize=True`) run in background threads. A `TranscriptionSession` orchestrator owns capture + Deepgram connections + an in-memory `TranscriptStore`, and exposes a thread-safe event queue that a tkinter `App` polls on the main thread via `.after()`. On Stop, the store is exported to a session folder.

**Tech Stack:** Python 3.12, `deepgram-sdk` 7.7.1 (already installed), `python-dotenv` (already installed), `PyAudioWPatch` 0.2.12.8 (new — WASAPI loopback capture), `tkinter` (stdlib).

**Spec:** `docs/superpowers/specs/2026-08-26-teams-transcribe-design.md`

## Global Constraints

- Project is **not** a git repository — every task's "commit" step is replaced by "no commit (no git repo); move to the next task."
- No pytest/unit-test framework in this project — every task's verification step is a manual `python -c` snippet or a short throwaway script run via Bash, with the exact expected output stated.
- Name-sanitization rule (from spec): strip characters invalid in Windows file/folder names (`< > : " / \ | ? *` and ASCII control chars 0x00–0x1F), then truncate to **20** characters, then append `[DD-MM-YYYY]_[HH-MM]`.
- Default name prefixes: `DeepGram` for `ListenSteam.py`'s output file, `DeepGramMeeting` for TeamsTranscribe's session folder.
- Deepgram model: `nova-3`. Language codes exposed in the GUI: `ru`, `en`, `multi` (the last enables Deepgram's multilingual code-switching, confirmed valid for streaming with `nova-3`).
- Endpointing: `100` ms when language is `multi` (per Deepgram's guidance for code-switching), `10` ms otherwise — matches `ListenSteam.py`'s existing default.
- Audio format sent to Deepgram: `encoding="linear16"`, 16-bit PCM, mono, at each source's own native device sample rate (no resampling between mic and system audio — confirmed necessary because on the dev machine mic defaults to 44100 Hz and the loopback device to 48000 Hz).
- Confirmed via manual spike (this session): `PyAudioWPatch` callback-mode streams (`stream_callback=`) with `format=pyaudio.paInt16` work correctly for both a regular WASAPI microphone (mono, 44100 Hz) and a WASAPI loopback device (2ch, 48000 Hz, only produces non-silent data while something is actually playing — 0 bytes is not an error, it's silence). Loopback pseudo-devices also appear in the plain device enumeration and must be filtered out by comparing against `get_loopback_device_info_generator()`, and mic enumeration should be restricted to the WASAPI host API (`get_host_api_info_by_type(pyaudio.paWASAPI)`) to avoid duplicate MME/DirectSound entries for the same physical device.

---

### Task 1: Shared filename-sanitization module + refactor `ListenSteam.py`

**Files:**
- Create: `naming.py`
- Modify: `ListenSteam.py` (replace `INVALID_FILENAME_CHARS`, `OUTFILE_NAME_MAX_LENGTH`, `build_output_filename` with calls into `naming.py`)

**Interfaces:**
- Produces: `naming.sanitize_name_component(raw: str) -> str`, `naming.build_timestamped_name(prefix: str | None, default_prefix: str, when: datetime | None = None) -> str` (returns a name **without** extension, e.g. `"DeepGram26-08-2026_11-09"`).

- [ ] **Step 1: Write `naming.py`**

```python
import re
from datetime import datetime

INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
NAME_MAX_LENGTH = 20


def sanitize_name_component(raw: str) -> str:
    """Strip characters invalid in Windows file/folder names and truncate to NAME_MAX_LENGTH."""
    return INVALID_FILENAME_CHARS.sub("", raw)[:NAME_MAX_LENGTH]


def build_timestamped_name(prefix: str | None, default_prefix: str, when: datetime | None = None) -> str:
    """Build '<prefix><DD-MM-YYYY>_<HH-MM>' (no extension).

    If prefix is falsy, uses default_prefix verbatim. Otherwise prefix is
    sanitized and truncated to NAME_MAX_LENGTH first.
    """
    when = when or datetime.now()
    timestamp = when.strftime("%d-%m-%Y_%H-%M")
    clean_prefix = sanitize_name_component(prefix) if prefix else default_prefix
    return f"{clean_prefix}{timestamp}"
```

- [ ] **Step 2: Verify manually**

Run:
```bash
./.venv/Scripts/python.exe -c "
from datetime import datetime
from naming import build_timestamped_name, sanitize_name_component

when = datetime(2026, 8, 26, 11, 9)
assert build_timestamped_name(None, 'DeepGram', when) == 'DeepGram26-08-2026_11-09'
assert build_timestamped_name('', 'DeepGram', when) == 'DeepGram26-08-2026_11-09'
assert build_timestamped_name('My<>:\"/\\\\|?*Radio_Session_Name_TooLong', 'DeepGram', when) == 'MyRadio_Session_Name26-08-2026_11-09'
assert sanitize_name_component('a<b>c') == 'abc'
print('OK')
"
```
Expected: `OK` printed, no assertion errors.

- [ ] **Step 3: Refactor `ListenSteam.py` to use `naming.py`**

Replace the current block:
```python
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

OUTFILE_NAME_MAX_LENGTH = 20


def build_output_filename(out_file: str | None) -> str:
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")

    if out_file:
        prefix = INVALID_FILENAME_CHARS.sub("", out_file)[:OUTFILE_NAME_MAX_LENGTH]
    else:
        prefix = "DeepGram"

    return f"{prefix}{timestamp}.txt"
```

with:
```python
from naming import build_timestamped_name


def build_output_filename(out_file: str | None) -> str:
    return f"{build_timestamped_name(out_file, 'DeepGram')}.txt"
```

Remove the now-unused `import re` and `OUTFILE_NAME_MAX_LENGTH` help text reference in `argparse` stays the same (it still refers to "обрезается до 20 символов" which is still true, now enforced by `naming.NAME_MAX_LENGTH`).

- [ ] **Step 4: Verify `ListenSteam.py` still produces the same filenames**

Run:
```bash
./.venv/Scripts/python.exe -c "
import ListenSteam
" 2>&1 | head -5
```
Expected: no `ImportError`/`SyntaxError` (the script will raise on `DEEPGRAM_API_KEY` only if `.env` is missing, or hang trying to connect — importing top-level code will actually start the stream, so instead just check it parses):
```bash
./.venv/Scripts/python.exe -c "import ast; ast.parse(open('ListenSteam.py', encoding='utf-8').read()); print('parses OK')"
```
Expected: `parses OK`.

- [ ] **Step 5: No commit (no git repo in this project) — proceed to Task 2.**

---

### Task 2: `TranscriptStore` — speaker registry and utterance log

**Files:**
- Create: `teams_transcribe/__init__.py` (empty)
- Create: `teams_transcribe/transcript_store.py`

**Interfaces:**
- Produces: `SpeakerKey = tuple[str, int | None]`, `Utterance` dataclass (`speaker_key`, `timestamp: datetime`, `is_final: bool`, `text: str`), `TranscriptStore` with methods `register_mic() -> None`, `get_or_create_system_speaker_name(speaker_id: int) -> tuple[str, bool]` (name, is_new), `rename(speaker_key: SpeakerKey, new_name: str) -> None`, `speaker_name(speaker_key: SpeakerKey) -> str`, `add_utterance(speaker_key: SpeakerKey, timestamp: datetime, is_final: bool, text: str) -> None`, `final_utterances() -> list[Utterance]`.

- [ ] **Step 1: Write `teams_transcribe/transcript_store.py`**

```python
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


class TranscriptStore:
    """Thread-safe in-memory store of utterances and speaker display names."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._utterances: list[Utterance] = []
        self._speaker_names: dict[SpeakerKey, str] = {}
        self._system_speaker_order: list[int] = []

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
                return self._speaker_names[key], True
            return self._speaker_names[key], False

    def rename(self, speaker_key: SpeakerKey, new_name: str) -> None:
        with self._lock:
            self._speaker_names[speaker_key] = new_name

    def speaker_name(self, speaker_key: SpeakerKey) -> str:
        with self._lock:
            return self._speaker_names.get(speaker_key, str(speaker_key))

    def add_utterance(self, speaker_key: SpeakerKey, timestamp: datetime, is_final: bool, text: str) -> None:
        with self._lock:
            self._utterances.append(Utterance(speaker_key, timestamp, is_final, text))

    def final_utterances(self) -> list[Utterance]:
        """All is_final=True utterances, in insertion order."""
        with self._lock:
            return [u for u in self._utterances if u.is_final]
```

- [ ] **Step 2: Verify manually**

Run:
```bash
./.venv/Scripts/python.exe -c "
from datetime import datetime
from teams_transcribe.transcript_store import TranscriptStore

s = TranscriptStore()
s.register_mic()
assert s.speaker_name(('mic', None)) == 'Я'

name0, is_new0 = s.get_or_create_system_speaker_name(0)
assert name0 == 'Собеседник 1' and is_new0 is True
name0b, is_new0b = s.get_or_create_system_speaker_name(0)
assert name0b == 'Собеседник 1' and is_new0b is False
name1, is_new1 = s.get_or_create_system_speaker_name(1)
assert name1 == 'Собеседник 2' and is_new1 is True

s.rename(('system', 1), 'Иван')
assert s.speaker_name(('system', 1)) == 'Иван'

s.add_utterance(('mic', None), datetime(2026, 8, 26, 11, 0, 0), True, 'привет')
s.add_utterance(('system', 1), datetime(2026, 8, 26, 11, 0, 1), False, 'interim text')
s.add_utterance(('system', 1), datetime(2026, 8, 26, 11, 0, 2), True, 'и тебе привет')

finals = s.final_utterances()
assert len(finals) == 2, finals
assert finals[0].text == 'привет'
assert finals[1].text == 'и тебе привет'
assert finals[1].speaker_key == ('system', 1)
print('OK')
"
```
Expected: `OK` printed, no assertion errors.

- [ ] **Step 3: No commit (no git repo) — proceed to Task 3.**

---

### Task 3: `exporter.py` — write session output files

**Files:**
- Create: `teams_transcribe/exporter.py`

**Interfaces:**
- Consumes: `teams_transcribe.transcript_store.TranscriptStore` (from Task 2), `naming.sanitize_name_component` (from Task 1).
- Produces: `export_session(store: TranscriptStore, output_dir: pathlib.Path) -> None`, writing `transcript.txt`, `transcript.json`, `speakers/<Имя>.txt` into `output_dir`.

- [ ] **Step 1: Write `teams_transcribe/exporter.py`**

```python
import json
from pathlib import Path

from naming import sanitize_name_component
from teams_transcribe.transcript_store import TranscriptStore


def export_session(store: TranscriptStore, output_dir: Path) -> None:
    """Write transcript.txt, transcript.json and speakers/<Имя>.txt into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    utterances = store.final_utterances()

    txt_lines: list[str] = []
    json_entries: list[dict] = []
    per_speaker: dict[str, list[str]] = {}

    for u in utterances:
        name = store.speaker_name(u.speaker_key)
        time_str = u.timestamp.strftime("%H:%M:%S")
        txt_lines.append(f"[{time_str}] {name}: {u.text}")
        json_entries.append(
            {
                "speaker_source": u.speaker_key[0],
                "speaker_id": u.speaker_key[1],
                "speaker_name": name,
                "timestamp": u.timestamp.isoformat(),
                "is_final": u.is_final,
                "text": u.text,
            }
        )
        per_speaker.setdefault(name, []).append(u.text)

    (output_dir / "transcript.txt").write_text("\n".join(txt_lines), encoding="utf-8")
    (output_dir / "transcript.json").write_text(
        json.dumps(json_entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    speakers_dir = output_dir / "speakers"
    speakers_dir.mkdir(exist_ok=True)
    for name, texts in per_speaker.items():
        safe_name = sanitize_name_component(name) or "Speaker"
        (speakers_dir / f"{safe_name}.txt").write_text("\n".join(texts), encoding="utf-8")
```

- [ ] **Step 2: Verify manually**

Run:
```bash
./.venv/Scripts/python.exe -c "
import shutil
from datetime import datetime
from pathlib import Path

from teams_transcribe.transcript_store import TranscriptStore
from teams_transcribe.exporter import export_session

s = TranscriptStore()
s.register_mic()
s.get_or_create_system_speaker_name(0)
s.add_utterance(('mic', None), datetime(2026, 8, 26, 11, 0, 0), True, 'Привет всем')
s.add_utterance(('system', 0), datetime(2026, 8, 26, 11, 0, 5), True, 'Привет, как дела?')

out = Path('_export_test_tmp')
if out.exists():
    shutil.rmtree(out)
export_session(s, out)

txt = (out / 'transcript.txt').read_text(encoding='utf-8')
assert '[11:00:00] Я: Привет всем' in txt
assert '[11:00:05] Собеседник 1: Привет, как дела?' in txt

import json
data = json.loads((out / 'transcript.json').read_text(encoding='utf-8'))
assert len(data) == 2
assert data[0]['speaker_name'] == 'Я'

assert (out / 'speakers' / 'Я.txt').read_text(encoding='utf-8') == 'Привет всем'
assert (out / 'speakers' / 'Собеседник 1.txt').read_text(encoding='utf-8') == 'Привет, как дела?'

shutil.rmtree(out)
print('OK')
"
```
Expected: `OK` printed, no assertion errors, and the temp folder is removed by the script itself.

- [ ] **Step 3: No commit (no git repo) — proceed to Task 4.**

---

### Task 4: `audio_capture.py` — device enumeration + mic/system capture

**Files:**
- Create: `teams_transcribe/audio_capture.py`

**Interfaces:**
- Produces: `DeviceInfo` dataclass (`index: int`, `name: str`, `sample_rate: int`, `channels: int`), `list_input_devices(p) -> list[DeviceInfo]`, `list_loopback_devices(p) -> list[DeviceInfo]`, `MicCapture(p, device: DeviceInfo, chunk_queue: queue.Queue[bytes])` with `.start()`, `.stop()`, `.set_muted(bool)`, `SystemCapture(p, device: DeviceInfo, chunk_queue: queue.Queue[bytes])` with `.start()`, `.stop()`.

- [ ] **Step 1: Write `teams_transcribe/audio_capture.py`**

```python
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
```

- [ ] **Step 2: Add `PyAudioWPatch` to the venv and `requirements.txt`**

Run:
```bash
./.venv/Scripts/python.exe -m pip install PyAudioWPatch -q
./.venv/Scripts/python.exe -m pip freeze | grep -i pyaudio
```
Expected: prints `PyAudioWPatch==0.2.12.8` (or newer). Then add the exact pinned line to `requirements.txt` (alongside the existing `deepgram-sdk` and `python-dotenv` lines).

- [ ] **Step 3: Verify device listing and downmix logic manually**

Run:
```bash
./.venv/Scripts/python.exe -c "
import pyaudiowpatch as pyaudio
from teams_transcribe.audio_capture import list_input_devices, list_loopback_devices, _downmix_to_mono

p = pyaudio.PyAudio()
mics = list_input_devices(p)
loops = list_loopback_devices(p)
assert len(mics) >= 1, mics
assert len(loops) >= 1, loops
assert all('[Loopback]' not in m.name for m in mics)
p.terminate()

stereo = bytes()
import struct
stereo = b''.join(struct.pack('<hh', 100, 300) for _ in range(4))
mono = _downmix_to_mono(stereo, 2)
import array
vals = array.array('h'); vals.frombytes(mono)
assert list(vals) == [200, 200, 200, 200], list(vals)
print('mics:', [m.name for m in mics])
print('loopback:', [l.name for l in loops])
print('OK')
"
```
Expected: `OK` printed, mic list contains no `[Loopback]`-tagged entries, downmix of interleaved (100, 300) pairs equals 200 per frame.

- [ ] **Step 4: Verify live capture manually (real audio)**

Run:
```bash
timeout 6 ./.venv/Scripts/python.exe -u -c "
import time, pyaudiowpatch as pyaudio
from teams_transcribe.audio_capture import list_input_devices, list_loopback_devices, MicCapture, SystemCapture
import queue

p = pyaudio.PyAudio()
mic_dev = list_input_devices(p)[0]
sys_dev = list_loopback_devices(p)[0]

mic_q, sys_q = queue.Queue(), queue.Queue()
mic = MicCapture(p, mic_dev, mic_q)
sysc = SystemCapture(p, sys_dev, sys_q)
mic.start(); sysc.start()
time.sleep(3)
mic.stop(); sysc.stop()
p.terminate()
print('mic chunks:', mic_q.qsize(), 'system chunks:', sys_q.qsize())
print('OK')
"
```
Expected: `OK` printed with `mic chunks` > 0 (speak or make noise near the mic during the 3s window if it reads 0 — silence still queues chunks of zero bytes, so this should be > 0 regardless).

- [ ] **Step 5: No commit (no git repo) — proceed to Task 5.**

---

### Task 5: `deepgram_stream.py` — one Deepgram live connection wrapper

**Files:**
- Create: `teams_transcribe/deepgram_stream.py`

**Interfaces:**
- Consumes: `deepgram.DeepgramClient` (from `deepgram-sdk`, already a dependency).
- Produces: `DeepgramStream(client, *, model: str, language: str, sample_rate: int, diarize: bool, endpointing: int, on_result: Callable[[float, float, bool, int | None, str], None])` with `.start() -> None`, `.send(pcm16_bytes: bytes) -> None`, `.stop() -> None`, and property `.started_at -> datetime | None`. `on_result` is called with `(start_seconds, duration_seconds, is_final, speaker_id_or_None, text)` for every non-empty transcript result.

- [ ] **Step 1: Write `teams_transcribe/deepgram_stream.py`**

```python
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
```

- [ ] **Step 2: Verify against the real Deepgram API using the example stream**

Run (uses the `.env` key already configured in this project, and the public example stream URL already used by `ListenSteam.py`, capped at ~8s):
```bash
timeout 12 ./.venv/Scripts/python.exe -u -c "
import time, threading, httpx
from dotenv import load_dotenv
import os
load_dotenv()
from deepgram import DeepgramClient
from teams_transcribe.deepgram_stream import DeepgramStream

results = []

def on_result(start, duration, is_final, speaker_id, text):
    results.append((is_final, speaker_id, text))
    print('RESULT', is_final, speaker_id, repr(text))

client = DeepgramClient(api_key=os.environ['DEEPGRAM_API_KEY'])
stream = DeepgramStream(client, model='nova-3', language='en', sample_rate=16000, diarize=False, endpointing=10, on_result=on_result)
stream.start()

def feed():
    with httpx.stream('GET', 'https://static.deepgram.com/examples/Bueller-Life-moves-pretty-fast.wav', follow_redirects=True) as r:
        for chunk in r.iter_bytes():
            stream.send(chunk)
            time.sleep(0.02)

t = threading.Thread(target=feed, daemon=True)
t.start()
time.sleep(8)
stream.stop()
assert len(results) > 0, 'no results received from Deepgram'
print('OK, got', len(results), 'results')
"
```
Expected: several `RESULT ...` lines printed with non-empty transcript text, then `OK, got N results` with `N > 0`. (This sends a WAV file's bytes directly as if they were raw PCM, so the transcript text itself will likely be garbled/wrong — that's fine, this step only proves the connection, callback wiring, and `on_result` plumbing work end-to-end against the real API, not transcription accuracy.)

- [ ] **Step 3: No commit (no git repo) — proceed to Task 6.**

---

### Task 6: `session.py` — orchestrates capture + Deepgram + store

**Files:**
- Create: `teams_transcribe/session.py`

**Interfaces:**
- Consumes: `teams_transcribe.audio_capture.{DeviceInfo, MicCapture, SystemCapture}` (Task 4), `teams_transcribe.deepgram_stream.DeepgramStream` (Task 5), `teams_transcribe.transcript_store.{TranscriptStore, SpeakerKey}` (Task 2).
- Produces: `UIEvent` dataclass (`kind: str` — `"utterance"` or `"new_speaker"`, `speaker_key: SpeakerKey`, `speaker_name: str`, `text: str = ""`, `is_final: bool = False`), `TranscriptionSession(api_key: str, mic_device: DeviceInfo, system_device: DeviceInfo, language: str, record_audio: bool = False, output_dir: pathlib.Path | None = None)` with `.store: TranscriptStore`, `.events: queue.Queue[UIEvent]`, `.start() -> None`, `.stop() -> None`, `.set_muted(bool) -> None`, `.rename_speaker(speaker_key: SpeakerKey, new_name: str) -> None`. When `record_audio=True`, `output_dir` must already exist (the caller creates it) and `mic.wav`/`system.wav` are written into it as the session runs.

- [ ] **Step 1: Write `teams_transcribe/session.py`**

```python
import queue
import threading
import wave
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pyaudiowpatch as pyaudio
from deepgram import DeepgramClient

from teams_transcribe.audio_capture import DeviceInfo, MicCapture, SystemCapture
from teams_transcribe.deepgram_stream import DeepgramStream
from teams_transcribe.transcript_store import SpeakerKey, TranscriptStore


def _open_wav_writer(path: Path, sample_rate: int) -> wave.Wave_write:
    writer = wave.open(str(path), "wb")
    writer.setnchannels(1)
    writer.setsampwidth(2)  # 16-bit PCM, matches pyaudio.paInt16
    writer.setframerate(sample_rate)
    return writer


def _endpointing_for_language(language: str) -> int:
    return 100 if language == "multi" else 10


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
        api_key: str,
        mic_device: DeviceInfo,
        system_device: DeviceInfo,
        language: str,
        record_audio: bool = False,
        output_dir: Optional[Path] = None,
    ):
        self.store = TranscriptStore()
        self.events: "queue.Queue[UIEvent]" = queue.Queue()
        self._client = DeepgramClient(api_key=api_key)
        self._mic_device = mic_device
        self._system_device = system_device
        self._language = language
        self._record_audio = record_audio
        self._output_dir = output_dir
        self._mic_queue: "queue.Queue[bytes]" = queue.Queue()
        self._system_queue: "queue.Queue[bytes]" = queue.Queue()
        self._pa: Optional[pyaudio.PyAudio] = None
        self._mic_capture: Optional[MicCapture] = None
        self._system_capture: Optional[SystemCapture] = None
        self._mic_stream: Optional[DeepgramStream] = None
        self._system_stream: Optional[DeepgramStream] = None
        self._mic_wav: Optional[wave.Wave_write] = None
        self._system_wav: Optional[wave.Wave_write] = None
        self._sender_threads: list[threading.Thread] = []
        self._stop_senders = threading.Event()

    def _mic_result(self, start, duration, is_final, speaker_id, text):
        key: SpeakerKey = ("mic", None)
        name = self.store.speaker_name(key)
        ts = self._mic_stream.started_at + timedelta(seconds=start)
        self.store.add_utterance(key, ts, is_final, text)
        self.events.put(UIEvent(kind="utterance", speaker_key=key, speaker_name=name, text=text, is_final=is_final))

    def _system_result(self, start, duration, is_final, speaker_id, text):
        sid = speaker_id if speaker_id is not None else 0
        key: SpeakerKey = ("system", sid)
        name, is_new = self.store.get_or_create_system_speaker_name(sid)
        if is_new:
            self.events.put(UIEvent(kind="new_speaker", speaker_key=key, speaker_name=name))
        ts = self._system_stream.started_at + timedelta(seconds=start)
        self.store.add_utterance(key, ts, is_final, text)
        self.events.put(UIEvent(kind="utterance", speaker_key=key, speaker_name=name, text=text, is_final=is_final))

    def _sender_loop(
        self,
        chunk_queue: "queue.Queue[bytes]",
        stream: DeepgramStream,
        wav_writer: Optional[wave.Wave_write],
    ) -> None:
        while not self._stop_senders.is_set():
            try:
                chunk = chunk_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            stream.send(chunk)
            if wav_writer is not None:
                wav_writer.writeframes(chunk)

    def start(self) -> None:
        self.store.register_mic()
        self._pa = pyaudio.PyAudio()
        endpointing = _endpointing_for_language(self._language)

        if self._record_audio:
            assert self._output_dir is not None, "output_dir is required when record_audio=True"
            self._mic_wav = _open_wav_writer(self._output_dir / "mic.wav", self._mic_device.sample_rate)
            self._system_wav = _open_wav_writer(self._output_dir / "system.wav", self._system_device.sample_rate)

        self._mic_stream = DeepgramStream(
            self._client, model="nova-3", language=self._language,
            sample_rate=self._mic_device.sample_rate, diarize=False,
            endpointing=endpointing, on_result=self._mic_result,
        )
        self._mic_stream.start()

        self._system_stream = DeepgramStream(
            self._client, model="nova-3", language=self._language,
            sample_rate=self._system_device.sample_rate, diarize=True,
            endpointing=endpointing, on_result=self._system_result,
        )
        self._system_stream.start()

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

    def rename_speaker(self, speaker_key: SpeakerKey, new_name: str) -> None:
        self.store.rename(speaker_key, new_name)

    def stop(self) -> None:
        self._stop_senders.set()
        for t in self._sender_threads:
            t.join(timeout=2)
        if self._mic_capture is not None:
            self._mic_capture.stop()
        if self._system_capture is not None:
            self._system_capture.stop()
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
```

- [ ] **Step 2: Verify manually against real devices + Deepgram (short real session)**

Run (speak into the mic during the 5s window for a non-empty result; system-audio silence is fine, it just won't produce text):
```bash
timeout 15 ./.venv/Scripts/python.exe -u -c "
import time, os
from dotenv import load_dotenv
load_dotenv()
import pyaudiowpatch as pyaudio
from teams_transcribe.audio_capture import list_input_devices, list_loopback_devices
from teams_transcribe.session import TranscriptionSession

p = pyaudio.PyAudio()
mic_dev = list_input_devices(p)[0]
sys_dev = list_loopback_devices(p)[0]
p.terminate()

session = TranscriptionSession(os.environ['DEEPGRAM_API_KEY'], mic_dev, sys_dev, 'ru')
session.start()
time.sleep(5)
session.set_muted(True)
time.sleep(1)
session.set_muted(False)
time.sleep(2)
session.stop()

import queue
events = []
try:
    while True:
        events.append(session.events.get_nowait())
except queue.Empty:
    pass

print('total events:', len(events))
for e in events[:10]:
    print(e.kind, e.speaker_key, e.speaker_name, e.is_final, repr(e.text))
print('final utterances stored:', len(session.store.final_utterances()))
print('OK')
"
```
Expected: `OK` at the end, `total events` >= 0 (0 is only acceptable if you stayed completely silent — speak during the window to get a non-zero count and confirm the pipeline), no unhandled exceptions/tracebacks.

- [ ] **Step 3: Verify `record_audio=True` writes non-empty WAV files**

Run (speak into the mic during the 3s window):
```bash
timeout 12 ./.venv/Scripts/python.exe -u -c "
import shutil, time, os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import pyaudiowpatch as pyaudio
from teams_transcribe.audio_capture import list_input_devices, list_loopback_devices
from teams_transcribe.session import TranscriptionSession

p = pyaudio.PyAudio()
mic_dev = list_input_devices(p)[0]
sys_dev = list_loopback_devices(p)[0]
p.terminate()

out = Path('_session_wav_test_tmp')
if out.exists():
    shutil.rmtree(out)
out.mkdir()

session = TranscriptionSession(os.environ['DEEPGRAM_API_KEY'], mic_dev, sys_dev, 'ru', record_audio=True, output_dir=out)
session.start()
time.sleep(3)
session.stop()

mic_wav = out / 'mic.wav'
system_wav = out / 'system.wav'
assert mic_wav.exists() and mic_wav.stat().st_size > 44, mic_wav.stat().st_size
assert system_wav.exists() and system_wav.stat().st_size >= 44, system_wav.stat().st_size

shutil.rmtree(out)
print('OK')
"
```
Expected: `OK` printed (44 bytes is a bare WAV header with zero audio frames — `mic.wav` must exceed that since you spoke during the window; `system.wav` may legitimately sit at exactly the header size if nothing was playing through speakers).

- [ ] **Step 4: No commit (no git repo) — proceed to Task 7.**

---

### Task 7: `gui.py` — tkinter application

**Files:**
- Create: `teams_transcribe/gui.py`

**Interfaces:**
- Consumes: `teams_transcribe.audio_capture.{list_input_devices, list_loopback_devices}` (Task 4), `teams_transcribe.session.TranscriptionSession` (Task 6), `teams_transcribe.exporter.export_session` (Task 3), `naming.build_timestamped_name` (Task 1).
- Produces: `App(tk.Tk)` — the whole GUI; `run(api_key: str) -> None` helper that constructs and mainloops an `App`.

- [ ] **Step 1: Write `teams_transcribe/gui.py`**

```python
import queue
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

import pyaudiowpatch as pyaudio

from naming import build_timestamped_name
from teams_transcribe.audio_capture import list_input_devices, list_loopback_devices
from teams_transcribe.exporter import export_session
from teams_transcribe.session import TranscriptionSession

LANGUAGES = [("Русский", "ru"), ("Английский", "en"), ("Мультиязычный (code-switching)", "multi")]

CONSENT_NOTICE = (
    "Это приложение записывает и расшифровывает весь разговор, включая\n"
    "голоса удалённых участников. Получение согласия участников на запись\n"
    "и соответствие политике вашей компании/законодательству — на вас."
)


class App(tk.Tk):
    def __init__(self, api_key: str):
        super().__init__()
        self.title("TeamsTranscribe")
        self._api_key = api_key
        self._session: TranscriptionSession | None = None
        self._muted = False
        self._speaker_rows: dict[tuple, tk.Entry] = {}
        self._interim_line_start: str | None = None

        self._output_dir: Path | None = None

        self._pa = pyaudio.PyAudio()
        self._mic_devices = list_input_devices(self._pa)
        self._system_devices = list_loopback_devices(self._pa)

        self._build_widgets()
        messagebox.showinfo("Перед началом", CONSENT_NOTICE)

    def _build_widgets(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Label(top, text="Микрофон:").grid(row=0, column=0, sticky="w")
        self.mic_combo = ttk.Combobox(
            top, values=[d.name for d in self._mic_devices], state="readonly", width=40
        )
        if self._mic_devices:
            self.mic_combo.current(0)
        self.mic_combo.grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(top, text="Системный звук:").grid(row=1, column=0, sticky="w")
        self.system_combo = ttk.Combobox(
            top, values=[d.name for d in self._system_devices], state="readonly", width=40
        )
        if self._system_devices:
            self.system_combo.current(0)
        self.system_combo.grid(row=1, column=1, sticky="w", padx=4)

        ttk.Label(top, text="Язык распознавания:").grid(row=2, column=0, sticky="w")
        self.lang_combo = ttk.Combobox(
            top, values=[label for label, _ in LANGUAGES], state="readonly", width=40
        )
        self.lang_combo.current(0)
        self.lang_combo.grid(row=2, column=1, sticky="w", padx=4)

        ttk.Label(top, text="Имя сессии (опц.):").grid(row=3, column=0, sticky="w")
        self.session_name_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.session_name_var, width=42).grid(row=3, column=1, sticky="w", padx=4)

        self.record_wav_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            top, text="Сохранять сырое аудио (WAV)", variable=self.record_wav_var
        ).grid(row=4, column=0, columnspan=2, sticky="w")

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", padx=8, pady=4)
        self.start_button = ttk.Button(buttons, text="Начать", command=self._on_start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="Стоп", command=self._on_stop, state="disabled")
        self.stop_button.pack(side="left", padx=4)
        self.mute_button = ttk.Button(buttons, text="Заглушить микрофон", command=self._on_mute_toggle, state="disabled")
        self.mute_button.pack(side="left", padx=4)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        self.transcript_box = scrolledtext.ScrolledText(body, width=70, height=25, state="disabled")
        self.transcript_box.pack(side="left", fill="both", expand=True)

        speakers_frame = ttk.Frame(body)
        speakers_frame.pack(side="left", fill="y", padx=(8, 0))
        ttk.Label(speakers_frame, text="Участники:").pack(anchor="w")
        self.speakers_container = ttk.Frame(speakers_frame)
        self.speakers_container.pack(fill="y")

    def _on_start(self) -> None:
        if not self._mic_devices or not self._system_devices:
            messagebox.showerror("Ошибка", "Не найдено устройство микрофона или системного звука.")
            return
        mic = self._mic_devices[self.mic_combo.current()]
        system = self._system_devices[self.system_combo.current()]
        language = LANGUAGES[self.lang_combo.current()][1]
        record_audio = self.record_wav_var.get()

        session_name = build_timestamped_name(self.session_name_var.get(), "DeepGramMeeting")
        self._output_dir = Path(session_name)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._session = TranscriptionSession(
            self._api_key, mic, system, language,
            record_audio=record_audio, output_dir=self._output_dir,
        )
        try:
            self._session.start()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            messagebox.showerror("Ошибка подключения к Deepgram", str(exc))
            self._session = None
            return

        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.mute_button.config(state="normal")
        self.after(100, self._poll_events)

    def _on_stop(self) -> None:
        if self._session is None:
            return
        session = self._session
        self._session = None
        session.stop()

        export_session(session.store, self._output_dir)

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.mute_button.config(state="disabled")
        messagebox.showinfo("Готово", f"Транскрипт сохранён в:\n{self._output_dir.resolve()}")

    def _on_mute_toggle(self) -> None:
        self._muted = not self._muted
        if self._session is not None:
            self._session.set_muted(self._muted)
        self.mute_button.config(text="Включить микрофон" if self._muted else "Заглушить микрофон")

    def _poll_events(self) -> None:
        if self._session is None:
            return
        try:
            while True:
                event = self._session.events.get_nowait()
                self._handle_event(event)
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _handle_event(self, event) -> None:
        if event.kind == "new_speaker":
            self._add_speaker_row(event.speaker_key, event.speaker_name)
        elif event.kind == "utterance":
            self._update_transcript(event)

    def _add_speaker_row(self, speaker_key, default_name: str) -> None:
        if speaker_key in self._speaker_rows:
            return
        row = ttk.Frame(self.speakers_container)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=default_name + ":").pack(side="left")
        var = tk.StringVar(value=default_name)

        def on_change(*_args, key=speaker_key, var=var):
            if self._session is not None:
                self._session.rename_speaker(key, var.get())

        entry = tk.Entry(row, textvariable=var, width=16)
        entry.pack(side="left", padx=4)
        var.trace_add("write", on_change)
        self._speaker_rows[speaker_key] = entry

    def _update_transcript(self, event) -> None:
        self.transcript_box.config(state="normal")
        if self._interim_line_start is not None:
            self.transcript_box.delete(self._interim_line_start, "end")
            self._interim_line_start = None

        line = f"{event.speaker_name}: {event.text}\n"
        if event.is_final:
            self.transcript_box.insert("end", line)
        else:
            self._interim_line_start = self.transcript_box.index("end-1c")
            self.transcript_box.insert("end", line)

        self.transcript_box.see("end")
        self.transcript_box.config(state="disabled")


def run(api_key: str) -> None:
    app = App(api_key)
    app.mainloop()
```

- [ ] **Step 2: Verify the module imports and builds widgets without a live session**

Run:
```bash
./.venv/Scripts/python.exe -c "
import tkinter as tk
from teams_transcribe.gui import App

app = App('fake-key-for-widget-test')
app.update()
assert app.mic_combo['values'], 'no mic devices listed'
assert app.system_combo['values'], 'no loopback devices listed'
app.destroy()
print('OK')
"
```
Expected: `OK` printed (a window may flash briefly and close), no traceback. Note: this only checks widget construction and device listing — it does not call `_on_start`, since that would open a real Deepgram connection with a fake key.

- [ ] **Step 3: No commit (no git repo) — proceed to Task 8.**

---

### Task 8: Entry point, `requirements.txt`, `README.md`

**Files:**
- Create: `TeamsTranscribe.py`
- Modify: `requirements.txt`
- Modify: `README.md`

**Interfaces:**
- Consumes: `teams_transcribe.gui.run` (Task 7).

- [ ] **Step 1: Write `TeamsTranscribe.py`**

```python
import os
import sys

from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")

if __name__ == "__main__":
    if not DEEPGRAM_API_KEY:
        print("DEEPGRAM_API_KEY не найден. Создайте .env на основе .env.example.", file=sys.stderr)
        sys.exit(1)

    from teams_transcribe.gui import run

    run(DEEPGRAM_API_KEY)
```

- [ ] **Step 2: Confirm `requirements.txt` contains all three dependencies**

`requirements.txt` should now read (order not important, exact versions as installed in this venv):
```
deepgram-sdk==7.7.1
python-dotenv==1.2.3
PyAudioWPatch==0.2.12.8
```

- [ ] **Step 3: Verify the entry point parses and the missing-key path works**

Run:
```bash
./.venv/Scripts/python.exe -c "import ast; ast.parse(open('TeamsTranscribe.py', encoding='utf-8').read()); print('parses OK')"
```
Expected: `parses OK`.

```bash
DEEPGRAM_API_KEY= ./.venv/Scripts/python.exe -c "
import subprocess, sys
r = subprocess.run([sys.executable, 'TeamsTranscribe.py'], capture_output=True, text=True, env={'PATH': __import__('os').environ['PATH'], 'DEEPGRAM_API_KEY': ''})
print('returncode:', r.returncode)
print('stderr:', r.stderr.strip())
assert r.returncode == 1
assert 'DEEPGRAM_API_KEY' in r.stderr
print('OK')
"
```
Expected: `OK` printed, confirming the app exits cleanly with a clear message when the key is missing (this test intentionally clears the env var for the subprocess so `.env` in the current directory would still be picked up by `load_dotenv()` if present — if this project's real `.env` has a valid key, the subprocess will instead try to open the GUI; if that happens, this manual check can be skipped since Task 7 Step 2 already covers GUI construction, and the missing-key branch can instead be verified by temporarily renaming `.env`).

- [ ] **Step 4: Update `README.md`**

Add a new section after the existing `ListenSteam.py` section (mirroring its style), covering:
- What `TeamsTranscribe.py` does (GUI, dual capture, diarization, live renaming, export).
- The consent/legal notice (reuse wording close to the spec's legal section).
- How to run it: `.venv\Scripts\python.exe TeamsTranscribe.py`.
- What gets exported and where: a session folder named `DeepGramMeeting[DD-MM-YYYY]_[HH-MM]/` (or a custom prefix typed into the "Имя сессии" field, sanitized/truncated the same way as `ListenSteam.py`'s `--OutFile`), created at the moment you click "Начать" and containing `transcript.txt`, `transcript.json`, `speakers/<Имя>.txt` after "Стоп", plus `mic.wav`/`system.wav` if "Сохранять сырое аудио (WAV)" was checked.
- That it's Windows-only (WASAPI loopback via `PyAudioWPatch`).
- Update the "Требования"/"Структура проекта" sections to mention `PyAudioWPatch` and the new files.

- [ ] **Step 5: No commit (no git repo) — proceed to Task 9.**

---

### Task 9: Manual end-to-end verification

**Files:** none (verification only).

- [ ] **Step 1: Launch the app**

Run (not backgrounded — this is interactive):
```bash
./.venv/Scripts/python.exe TeamsTranscribe.py
```

- [ ] **Step 2: Manual checklist**

Using the GUI, confirm each of the following, using e.g. a YouTube video playing through speakers as a stand-in for "remote participants" and your own voice as "Я":

1. Consent notice dialog appears before the main window is usable.
2. Device dropdowns are pre-populated (mic and system-audio device) and language defaults to "Русский".
3. Clicking "Начать" enables "Стоп"/"Заглушить микрофон" and disables "Начать".
4. Speaking into the mic produces live text in the transcript pane labeled "Я".
5. Playing audio (a video with speech) through speakers produces live text labeled "Собеседник 1" (and "Собеседник 2" if a second distinct voice plays).
6. Renaming a "Собеседник N" field updates future transcript lines with the new name.
7. Clicking "Заглушить микрофон" stops new "Я" text from appearing while speaking; clicking it again resumes.
8. Clicking "Стоп" shows a confirmation dialog with an output folder path, and that folder exists with `transcript.txt`, `transcript.json`, and a `speakers/` subfolder containing one `.txt` file per participant (using whatever name was set, including any rename from step 6).
9. `transcript.txt` lines are in chronological order across both "Я" and "Собеседник N" entries.
10. Repeat with the "Сохранять сырое аудио (WAV)" checkbox ticked before "Начать": after "Стоп", the output folder additionally contains `mic.wav` and `system.wav`, and `mic.wav` plays back your recorded voice when opened.

- [ ] **Step 3: Report results back**

Note any checklist item that failed, with what was observed, before considering this plan complete.
