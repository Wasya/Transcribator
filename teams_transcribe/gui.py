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
        self._interim_marks: dict[tuple, str] = {}

        self._output_dir: Path | None = None

        self._pa = pyaudio.PyAudio()
        self._mic_devices = list_input_devices(self._pa)
        self._system_devices = list_loopback_devices(self._pa)

        self._build_widgets()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
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

    @staticmethod
    def _resolve_output_dir(base_name: str) -> Path:
        """Return a Path for base_name, disambiguated with a _2, _3, ... suffix
        if a non-empty directory of that name already exists (minute-granularity
        timestamps in build_timestamped_name can collide on rapid restart)."""
        candidate = Path(base_name)
        if not candidate.exists() or not any(candidate.iterdir()):
            return candidate
        n = 2
        while True:
            candidate = Path(f"{base_name}_{n}")
            if not candidate.exists() or not any(candidate.iterdir()):
                return candidate
            n += 1

    def _on_start(self) -> None:
        if not self._mic_devices or not self._system_devices:
            messagebox.showerror("Ошибка", "Не найдено устройство микрофона или системного звука.")
            return

        for child in list(self.speakers_container.winfo_children()):
            child.destroy()
        self._speaker_rows.clear()
        self._interim_marks.clear()
        self._muted = False
        self.mute_button.config(text="Заглушить микрофон")

        mic = self._mic_devices[self.mic_combo.current()]
        system = self._system_devices[self.system_combo.current()]
        language = LANGUAGES[self.lang_combo.current()][1]
        record_audio = self.record_wav_var.get()

        session_name = build_timestamped_name(self.session_name_var.get(), "DeepGramMeeting")
        self._output_dir = self._resolve_output_dir(session_name)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._session = TranscriptionSession(
            self._api_key, mic, system, language,
            record_audio=record_audio, output_dir=self._output_dir,
        )
        try:
            self._session.start()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            try:
                self._session.stop()
            except Exception:
                # Best-effort teardown of a partially-open session; the
                # original error below is what matters to the user.
                pass
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
        try:
            try:
                session.stop()
            except Exception as exc:
                messagebox.showwarning("Предупреждение", f"Ошибка при остановке сессии: {exc}")
            export_session(session.store, self._output_dir)
            messagebox.showinfo("Готово", f"Транскрипт сохранён в:\n{self._output_dir.resolve()}")
        except Exception as exc:
            messagebox.showerror("Ошибка экспорта", str(exc))
        finally:
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.mute_button.config(state="disabled")

    def _on_close(self) -> None:
        if self._session is not None:
            if not messagebox.askyesno(
                "Завершить сеанс?",
                "Идёт запись. Закрыть окно сейчас и сохранить то, что уже записано?",
            ):
                return
            self._on_stop()
        self._pa.terminate()
        self.destroy()

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
        elif event.kind == "error":
            messagebox.showerror("Ошибка соединения", f"Сессия остановлена из-за ошибки:\n{event.text}")
            self._on_stop()

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

        text = f"{event.speaker_name}: {event.text}"
        mark = self._interim_marks.get(event.speaker_key)
        if mark is not None:
            # This speaker already has an in-progress line: replace just that
            # line's content in place (never touching the line's own trailing
            # newline), so other speakers' concurrently-updated lines are
            # untouched and marks never collide at the same buffer index.
            self.transcript_box.delete(f"{mark} linestart", f"{mark} lineend")
            self.transcript_box.insert(f"{mark} linestart", text)
            if event.is_final:
                self.transcript_box.mark_unset(mark)
                del self._interim_marks[event.speaker_key]
        else:
            insert_pos = self.transcript_box.index("end-1c")
            self.transcript_box.insert("end", text + "\n")
            if not event.is_final:
                mark_name = f"interim_{event.speaker_key[0]}_{event.speaker_key[1]}"
                self.transcript_box.mark_set(mark_name, insert_pos)
                self.transcript_box.mark_gravity(mark_name, "left")
                self._interim_marks[event.speaker_key] = mark_name

        self.transcript_box.see("end")
        self.transcript_box.config(state="disabled")


def run(api_key: str) -> None:
    app = App(api_key)
    app.mainloop()
