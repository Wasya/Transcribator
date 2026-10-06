import logging
import wave
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("teams_transcribe")

MP3_BITRATE = 96  # kbit/s, mono: transparent for speech, ~43 MB per hour
_CHUNK_FRAMES = 65536


def wav_to_mp3(wav_path: Path, mp3_path: Path, bitrate: int = MP3_BITRATE) -> None:
    """Encode a mono PCM16 WAV (as written by the session) to MP3 with LAME."""
    import lameenc

    with wave.open(str(wav_path), "rb") as wav:
        encoder = lameenc.Encoder()
        encoder.set_bit_rate(bitrate)
        encoder.set_in_sample_rate(wav.getframerate())
        encoder.set_channels(wav.getnchannels())
        encoder.set_quality(2)
        with open(mp3_path, "wb") as out:
            while True:
                frames = wav.readframes(_CHUNK_FRAMES)
                if not frames:
                    break
                out.write(encoder.encode(frames))
            out.write(encoder.flush())


def compress_session_audio(output_dir: Path, progress: Optional[Callable[[str], None]] = None) -> list[str]:
    """Replace mic.wav / system.wav in output_dir with mic.mp3 / system.mp3.

    A WAV is deleted only after its MP3 was written successfully; on any failure the
    WAV stays. Returns the names of the files that could not be converted.
    """
    failed: list[str] = []
    for name in ("mic", "system"):
        wav_path = output_dir / f"{name}.wav"
        if not wav_path.exists():
            continue
        if progress:
            progress(f"Сжатие аудио в MP3 ({name})…")
        mp3_path = output_dir / f"{name}.mp3"
        try:
            wav_to_mp3(wav_path, mp3_path)
            wav_path.unlink()
        except Exception:  # noqa: BLE001 - keep the WAV, report the failure
            log.exception("mp3 conversion of %s failed", wav_path.name)
            mp3_path.unlink(missing_ok=True)
            failed.append(wav_path.name)
    return failed
