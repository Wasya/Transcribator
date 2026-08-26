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
    used_filenames: dict[str, int] = {}
    for name, texts in per_speaker.items():
        safe_name = sanitize_name_component(name) or "Speaker"
        count = used_filenames.get(safe_name, 0)
        used_filenames[safe_name] = count + 1
        filename = f"{safe_name}.txt" if count == 0 else f"{safe_name}_{count + 1}.txt"
        (speakers_dir / filename).write_text("\n".join(texts), encoding="utf-8")
