"""Расшифровка звукового файла тем же конвейером, что и живая запись (WhisperX).

Нужна для воспроизводимых экспериментов: один и тот же файл можно прогнать с разными
моделями и настройками и сравнить результаты.

Примеры:
    python transcribe_file.py запись.mp3
    python transcribe_file.py запись.mp3 --models large-v3-turbo,large-v3 --diarization post
    python transcribe_file.py запись.mp3 --models small,large-v3 --reference эталон.txt
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

OUTPUT_ROOT = Path(__file__).resolve().parent / "Output"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", type=Path, help="звуковой файл (wav, mp3, m4a, ogg, ...)")
    p.add_argument("--models", help="модели через запятую (small, medium, large-v3-turbo, large-v3); по умолчанию — «авто»")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], help="устройство (по умолчанию из .env / авто)")
    p.add_argument("--compute-type", help="float16 | int8 | ... (по умолчанию по устройству)")
    p.add_argument("--language", default="ru", choices=["ru", "en", "de", "multi"], help="язык (по умолчанию ru)")
    p.add_argument("--diarization", default="post", choices=["off", "post", "live", "both"],
                   help="разбор по голосам: post — после распознавания (A), live — по ходу (B), both, off")
    p.add_argument("--max-segment", type=float, help="максимальная длина куска речи, с (по умолчанию 8)")
    p.add_argument("--context", action="store_true", help="подсказывать Whisper конец предыдущей фразы")
    p.add_argument("--vad-threshold", type=float, default=0.35, help="порог внутреннего детектора речи Whisper (0.35); 0 — выключить")
    p.add_argument("--seed", type=int, default=0, help="зерно случайных чисел для воспроизводимости (по умолчанию 0)")
    p.add_argument("--merge-gap", type=float, default=None, help="пауза склейки реплик одного голоса, с")
    p.add_argument("--reference", type=Path, help="эталонный текст (txt или transcript.txt) для сравнения")
    p.add_argument("--reference-offset", type=float, default=None,
                   help="на какой секунде звука начинается первая строка эталона (по умолчанию — с первой реплики)")
    p.add_argument("--out", type=Path, help="папка результатов (по умолчанию Output/File_<имя>_<время>)")
    return p.parse_args(argv)


def fmt_table(rows: list[list[str]]) -> str:
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows)


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    load_dotenv()

    if not args.audio.is_file():
        print(f"Файл не найден: {args.audio}", file=sys.stderr)
        return 2

    from teams_transcribe.config import load_settings, resolve_whisper_runtime
    from teams_transcribe.exporter import export_session
    from teams_transcribe.file_mode import (
        FileRunOptions, compare_texts, load_audio_16k, load_reference_text, load_reference_turns,
        score_diarization, transcribe_audio,
    )
    from teams_transcribe.transcript_store import DEFAULT_MERGE_GAP

    settings = load_settings()
    if args.device:
        settings.whisper_device = args.device
    if args.compute_type:
        settings.whisper_compute_type = args.compute_type
    if args.diarization != "off" and not settings.hf_token:
        print("Внимание: HF_TOKEN не задан — разбор по голосам будет пропущен (см. README).", file=sys.stderr)

    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else [None]
    reference = load_reference_text(args.reference) if args.reference else None
    ref_turns = load_reference_turns(args.reference) if args.reference else []
    score_voices = bool(ref_turns) and args.diarization != "off"
    merge_gap = DEFAULT_MERGE_GAP if args.merge_gap is None else args.merge_gap

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    base = args.out or OUTPUT_ROOT / f"File_{args.audio.stem[:20]}_{stamp}"
    base.mkdir(parents=True, exist_ok=True)

    print(f"Файл: {args.audio}")
    t0 = time.monotonic()
    audio = load_audio_16k(args.audio)
    print(f"Звук: {len(audio) / 16000:.0f} с (чтение {time.monotonic() - t0:.1f} с)")

    rows = [["модель", "устройство", "звук,с", "время,с", "x реал.", "кусков", "слов", "голосов"]]
    if reference is not None:
        rows[0] += ["схожесть", "WER~", "пропущено", "лишнего"]
    if score_voices:
        rows[0] += ["голосов в эталоне", "точность голосов", "чистота", "полнота"]
    summary = []
    details = []
    voice_details = []

    for requested in models:
        model, device, compute_type = resolve_whisper_runtime(settings, requested)
        opts = FileRunOptions(
            model=model, device=device, compute_type=compute_type, language=args.language,
            diarization=args.diarization, hf_token=settings.hf_token, max_segment=args.max_segment,
            use_context=args.context, seed=args.seed,
            vad_threshold=args.vad_threshold if args.vad_threshold > 0 else None,
        )
        print(f"\n=== {model} ({device}, {compute_type}), разбор по голосам: {args.diarization} ===")
        result = transcribe_audio(audio, opts, progress=lambda t: print("  " + t, flush=True))
        for note in result.notes:
            print(f"  ! {note}")

        out_dir = base / model.replace("/", "_")
        export_session(result.store, out_dir, drop_mic_echo=False, merge_gap=merge_gap)

        row = [model, device, f"{result.audio_seconds:.0f}", f"{result.wall_seconds:.0f}",
               f"{result.realtime_factor:.2f}", str(result.segments), str(result.words), str(result.speakers)]
        entry = {
            "model": model, "device": device, "compute_type": compute_type,
            "audio_seconds": result.audio_seconds, "wall_seconds": result.wall_seconds,
            "segments": result.segments, "words": result.words, "speakers": result.speakers,
            "options": {k: v for k, v in vars(opts).items() if k != "hf_token"}, "notes": result.notes,
        }
        if reference is not None:
            cmp = compare_texts(reference, result.text())
            row += [f"{cmp['similarity']:.3f}", f"{cmp['wer']:.3f}", str(len(cmp["missing"])), str(len(cmp["extra"]))]
            entry["comparison"] = cmp
            details.append((model, cmp))
        if score_voices:
            dia = score_diarization(result.store.final_utterances(), ref_turns, args.reference_offset)
            if dia is None:
                row += ["-", "-", "-", "-"]
            else:
                row += [str(dia["real"]), f"{dia['accuracy']:.3f}", f"{dia['purity']:.3f}", f"{dia['completeness']:.3f}"]
                entry["diarization"] = dia
                voice_details.append((model, dia))
        rows.append(row)
        summary.append(entry)
        print(f"  результат: {out_dir}")

    table = fmt_table(rows)
    print("\n" + table)
    text = [f"Файл: {args.audio}", f"Язык: {args.language}, разбор по голосам: {args.diarization}, seed: {args.seed}", "", table]
    for model, cmp in details:
        text.append(f"\n--- {model}: есть в эталоне, нет в результате (>=3 слов подряд):")
        text += [f"  - {m}" for m in cmp["missing"]] or ["  (нет)"]
        text.append(f"--- {model}: есть в результате, нет в эталоне (>=3 слов подряд):")
        text += [f"  + {m}" for m in cmp["extra"]] or ["  (нет)"]
    for model, dia in voice_details:
        text.append(f"\n--- {model}: разбор по голосам (найдено {dia['found']}, в эталоне {dia['real']}):")
        text.append("  точность — доля речи, правильно отнесённой к своему человеку; чистота < 1 — разные люди слиты в один "
                    "найденный голос; полнота < 1 — один человек раздроблен на несколько")
        for found, parts in dia["merged"]:
            text.append(f"  слиты: {found} = " + " + ".join(f"«{p}»" for p in parts))
        for real_name, parts in dia["split"]:
            text.append(f"  раздроблен: «{real_name}» → " + " + ".join(parts))
        if not dia["merged"] and not dia["split"]:
            text.append("  слияний и дроблений нет")
        text.append("  (Г1, Г2, … — найденные голоса в порядке появления)")
    (base / "summary.txt").write_text("\n".join(text), encoding="utf-8")
    (base / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for model, dia in voice_details:
        print(f"\n{model}: голоса — найдено {dia['found']} из {dia['real']}, точность {dia['accuracy']:.3f}"
              f" (чистота {dia['purity']:.3f}, полнота {dia['completeness']:.3f})")
        for found, parts in dia["merged"]:
            print(f"  слиты: {found} = " + " + ".join(f"«{p}»" for p in parts))
        for real_name, parts in dia["split"]:
            print(f"  раздроблен: «{real_name}» → " + " + ".join(parts))
    for model, cmp in details:
        print(f"\n{model}: пропущено {len(cmp['missing'])} фрагм., лишнего {len(cmp['extra'])} фрагм. (подробно — в summary.txt)")
    print(f"\nСводка: {base / 'summary.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
