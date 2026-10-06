import json
from pathlib import Path

from naming import sanitize_name_component
from teams_transcribe.echo_filter import find_mic_echo
from teams_transcribe.transcript_store import TranscriptStore, merge_utterances


def export_session(
    store: TranscriptStore,
    output_dir: Path,
    *,
    drop_mic_echo: bool = True,
    merge_gap: float = 3.0,
) -> int:
    """Write transcript.txt, transcript.json and speakers/<Имя>.txt into output_dir.

    With drop_mic_echo, mic utterances that only repeat what the system channel
    heard (remote voices leaking from the speakers) are left out of transcript.txt
    and the per-speaker files; transcript.json keeps them flagged "mic_echo": true.
    In the text files, consecutive utterances of one speaker separated by a pause
    of at most merge_gap seconds form one line (see merge_utterances);
    transcript.json always lists the individual utterances.
    Returns the number of utterances treated as echo.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    utterances = sorted(store.final_utterances(), key=lambda u: u.timestamp)
    echo = find_mic_echo(utterances) if drop_mic_echo else set()

    txt_lines: list[str] = []
    json_entries: list[dict] = []
    per_speaker: dict[str, list[str]] = {}

    for u in utterances:
        name = store.speaker_name(u.speaker_key)
        is_echo = id(u) in echo
        json_entries.append(
            {
                "speaker_source": u.speaker_key[0],
                "speaker_id": u.speaker_key[1],
                "speaker_name": name,
                "timestamp": u.timestamp.isoformat(),
                "is_final": u.is_final,
                "mic_echo": is_echo,
                "text": u.text,
            }
        )

    # Echo is removed before merging, so a remote speaker's line is not broken
    # up by the "Я" duplicates that used to sit between its parts.
    kept = [u for u in utterances if id(u) not in echo]
    for line in merge_utterances(kept, max_gap=merge_gap):
        name = store.speaker_name(line.speaker_key)
        txt_lines.append(f"[{line.timestamp:%H:%M:%S}] {name}: {line.text}")
        per_speaker.setdefault(name, []).append(line.text)

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
    return len(echo)
