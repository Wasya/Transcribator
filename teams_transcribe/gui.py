import logging
import math
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox, scrolledtext, ttk

import pyaudiowpatch as pyaudio

from naming import build_timestamped_name
from teams_transcribe.audio_capture import LevelMonitor, list_input_devices, list_loopback_devices
from teams_transcribe.exporter import export_session
from teams_transcribe.config import Settings, resolve_whisper_runtime
from teams_transcribe.transcript_store import DEFAULT_MERGE_GAP, TranscriptStore, merge_utterances
from teams_transcribe.session import (
    BACKEND_DEEPGRAM,
    BACKEND_WHISPERX,
    DIARIZE_BOTH,
    DIARIZE_LIVE,
    DIARIZE_OFF,
    DIARIZE_POST,
    SessionOptions,
    TranscriptionSession,
)

log = logging.getLogger("teams_transcribe")

# Session folders live in <project root>/Output regardless of the working directory.
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "Output"

BACKENDS = [
    ("Deepgram (облако, нужен API-ключ)", BACKEND_DEEPGRAM),
    ("WhisperX (локально, бесплатно)", BACKEND_WHISPERX),
]

WHISPER_MODELS = [
    ("Авто (по устройству)", None),
    ("small — быстрая, для CPU", "small"),
    ("medium", "medium"),
    ("large-v3-turbo — для GPU", "large-v3-turbo"),
    ("large-v3 — максимум качества", "large-v3"),
]

DIARIZATION_MODES = [
    ("После «Стоп» (вариант A, точнее)", DIARIZE_POST),
    ("Во время звонка (вариант B)", DIARIZE_LIVE),
    ("B во время звонка + A после «Стоп»", DIARIZE_BOTH),
    ("Не разделять участников", DIARIZE_OFF),
]

LANGUAGES = [
    ("Русский", "ru"),
    ("Английский", "en"),
    ("Немецкий", "de"),
    ("Мультиязычный (code-switching)", "multi"),
]

LEVEL_FLOOR_DB = -60.0  # RMS at or below this shows as an empty meter


def level_to_percent(rms: float) -> int:
    """Map RMS (0..1) to a 0..100 meter position on a dB scale, so quiet speech is visible."""
    if rms <= 0.0:
        return 0
    db = 20.0 * math.log10(rms)
    return int(max(0.0, min(100.0, (db - LEVEL_FLOOR_DB) / -LEVEL_FLOOR_DB * 100.0)))


CONSENT_NOTICE = (
    "Это приложение записывает и расшифровывает весь разговор, включая\n"
    "голоса удалённых участников. Получение согласия участников на запись\n"
    "и соответствие политике вашей компании/законодательству — на вас."
)


class App(tk.Tk):
    def __init__(self, settings: Settings):
        super().__init__()
        self.title("TeamsTranscribe")
        self._settings = settings
        self._work_queue: "queue.Queue" = queue.Queue()
        self._busy = False
        self._activity_text = ""
        self._activity_began: float | None = None
        self._tick_job = None
        self._recording_began = 0.0
        self._heard_speech = False  # a recognized utterance has appeared in this session
        self._session: TranscriptionSession | None = None
        self._muted = False
        self._speaker_rows: dict[tuple, tk.Entry] = {}
        # The transcript box is a view of _view_store (current or last session):
        # final utterances, merged per speaker, plus each speaker's in-progress
        # (interim) text at the bottom. It is redrawn as a whole, so renaming a
        # speaker immediately relabels all of their earlier lines too.
        self._view_store: TranscriptStore | None = None
        self._interims: dict[tuple, str] = {}
        self._render_pending = False
        self._recording = False  # session started and not yet stopped

        self._output_dir: Path | None = None
        # The most recently finished session: its speakers stay editable and
        # "Сохранить заново" re-exports it with the new names.
        self._last_session: TranscriptionSession | None = None
        self._last_output_dir: Path | None = None
        self._monitor: LevelMonitor | None = None

        self._pa = pyaudio.PyAudio()
        self._mic_devices = list_input_devices(self._pa)
        self._system_devices = list_loopback_devices(self._pa)

        self._build_widgets()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._start_monitor()
        self.after(100, self._poll_levels)
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
        self.mic_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_device_change())
        self.mic_level = ttk.Progressbar(top, orient="horizontal", length=120, mode="determinate", maximum=100)
        self.mic_level.grid(row=0, column=2, sticky="w", padx=4)

        ttk.Label(top, text="Системный звук:").grid(row=1, column=0, sticky="w")
        self.system_combo = ttk.Combobox(
            top, values=[d.name for d in self._system_devices], state="readonly", width=40
        )
        if self._system_devices:
            self.system_combo.current(0)
        self.system_combo.grid(row=1, column=1, sticky="w", padx=4)
        self.system_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_device_change())
        self.system_level = ttk.Progressbar(top, orient="horizontal", length=120, mode="determinate", maximum=100)
        self.system_level.grid(row=1, column=2, sticky="w", padx=4)
        ttk.Label(
            top, text="← уровень звука: говорите в микрофон / включите звук в звонке — полоски должны двигаться",
            foreground="#555",
        ).grid(row=0, column=3, rowspan=2, sticky="w", padx=4)

        ttk.Label(top, text="Язык распознавания:").grid(row=2, column=0, sticky="w")
        self.lang_combo = ttk.Combobox(
            top, values=[label for label, _ in LANGUAGES], state="readonly", width=40
        )
        self.lang_combo.current(0)
        self.lang_combo.grid(row=2, column=1, sticky="w", padx=4)

        ttk.Label(top, text="Движок распознавания:").grid(row=3, column=0, sticky="w")
        self.backend_combo = ttk.Combobox(
            top, values=[label for label, _ in BACKENDS], state="readonly", width=40
        )
        self.backend_combo.current(0 if self._settings.deepgram_api_key else 1)
        self.backend_combo.grid(row=3, column=1, sticky="w", padx=4)
        self.backend_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_backend_widgets())

        ttk.Label(top, text="Модель WhisperX:").grid(row=4, column=0, sticky="w")
        self.model_combo = ttk.Combobox(
            top, values=[label for label, _ in WHISPER_MODELS], state="readonly", width=40
        )
        self.model_combo.current(0)
        self.model_combo.grid(row=4, column=1, sticky="w", padx=4)

        ttk.Label(top, text="Разбор по голосам:").grid(row=5, column=0, sticky="w")
        self.diar_combo = ttk.Combobox(
            top, values=[label for label, _ in DIARIZATION_MODES], state="readonly", width=40
        )
        self.diar_combo.current(0)
        self.diar_combo.grid(row=5, column=1, sticky="w", padx=4)

        ttk.Label(top, text="Имя сессии (опц.):").grid(row=6, column=0, sticky="w")
        self.session_name_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.session_name_var, width=42).grid(row=6, column=1, sticky="w", padx=4)

        self.record_wav_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            top, text="Сохранять сырое аудио (WAV)", variable=self.record_wav_var
        ).grid(row=7, column=0, columnspan=2, sticky="w")

        self.drop_echo_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            top,
            text="Убирать из транскрипта эхо микрофона (голоса собеседников из динамиков)",
            variable=self.drop_echo_var,
        ).grid(row=8, column=0, columnspan=2, sticky="w")

        merge_row = ttk.Frame(top)
        merge_row.grid(row=9, column=0, columnspan=4, sticky="w")
        ttk.Label(merge_row, text="Не начинать новую реплику, если тот же голос молчал не дольше").pack(side="left")
        self.merge_gap_var = tk.DoubleVar(value=DEFAULT_MERGE_GAP)
        ttk.Spinbox(
            merge_row, from_=0.0, to=30.0, increment=0.5, width=5, textvariable=self.merge_gap_var,
            command=self._schedule_render,
        ).pack(side="left", padx=4)
        ttk.Label(merge_row, text="с (0 — каждая фраза отдельной строкой)").pack(side="left")
        self.merge_gap_var.trace_add("write", lambda *_a: self._schedule_render())
        self._update_backend_widgets()

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", padx=8, pady=4)
        self.start_button = ttk.Button(buttons, text="Начать", command=self._on_start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="Стоп", command=self._on_stop, state="disabled")
        self.stop_button.pack(side="left", padx=4)
        self.mute_button = ttk.Button(buttons, text="Заглушить микрофон", command=self._on_mute_toggle, state="disabled")
        self.mute_button.pack(side="left", padx=4)
        self.reexport_button = ttk.Button(
            buttons, text="Сохранить заново", command=self._on_reexport, state="disabled"
        )
        self.reexport_button.pack(side="left", padx=(16, 4))

        self.status_var = tk.StringVar(value="")
        status_row = ttk.Frame(self)
        status_row.pack(fill="x", padx=8)
        # Moving bar = "the program is working, not frozen" (model loading, stopping, diarization).
        self.activity_bar = ttk.Progressbar(status_row, mode="indeterminate", length=140)
        ttk.Label(status_row, textvariable=self.status_var, foreground="#555").pack(side="left", fill="x")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        self.transcript_box = scrolledtext.ScrolledText(
            body, width=70, height=25, state="disabled", wrap="word"
        )
        self.transcript_box.pack(side="left", fill="both", expand=True)
        self._bold_font = tkfont.nametofont(self.transcript_box.cget("font")).copy()
        self._bold_font.configure(weight="bold")
        self.transcript_box.tag_configure("name", font=self._bold_font)
        self.transcript_box.tag_configure("time", foreground="#888")
        self.transcript_box.tag_configure("interim", foreground="#888")

        speakers_frame = ttk.Frame(body)
        speakers_frame.pack(side="left", fill="y", padx=(8, 0))
        ttk.Label(speakers_frame, text="Участники:").pack(anchor="w")
        self.speakers_container = ttk.Frame(speakers_frame)
        self.speakers_container.pack(fill="y")
        ttk.Label(
            speakers_frame,
            text="Имена можно менять и после «Стоп» —\nзатем нажмите «Сохранить заново».",
            foreground="#555",
        ).pack(anchor="w", pady=(8, 0))

    # --- level meters -----------------------------------------------------

    def _selected_device(self, devices, combo):
        return devices[combo.current()] if devices and combo.current() >= 0 else None

    def _start_monitor(self) -> None:
        self._stop_monitor()
        if self._session is not None:
            return  # the session's own captures feed the meters
        self._monitor = LevelMonitor(
            self._pa,
            self._selected_device(self._mic_devices, self.mic_combo),
            self._selected_device(self._system_devices, self.system_combo),
        )
        self._monitor.start()
        for name, exc in self._monitor.errors().items():
            log.warning("level monitor: %s device failed to open: %s", name, exc)

    def _stop_monitor(self) -> None:
        if self._monitor is not None:
            self._monitor.stop()
            self._monitor = None

    def _on_device_change(self) -> None:
        if self._session is None:
            self._start_monitor()

    def _poll_levels(self) -> None:
        source = self._session if self._session is not None else self._monitor
        mic = level_to_percent(source.mic_level) if source is not None else 0
        system = level_to_percent(source.system_level) if source is not None else 0
        self.mic_level["value"] = mic
        self.system_level["value"] = system
        session = self._session
        if session is not None and self._recording:
            elapsed = int(time.monotonic() - self._recording_began)
            count = len(session.store.final_utterances())
            text = f"● Идёт запись {elapsed // 60}:{elapsed % 60:02d}  ·  реплик: {count}"
            if not self._heard_speech:
                text += "  ·  ждём речь (текст появляется после паузы в разговоре)…"
            if session.options.backend == BACKEND_WHISPERX:
                lag = session.backlog_seconds
                if lag >= 3.0:
                    text += f"  ·  распознавание отстаёт на {lag:.0f} с — попробуйте модель полегче"
                elif lag > 0.5 and self._heard_speech:
                    text += "  ·  распознаю…"
            if self.status_var.get() != text:
                self.status_var.set(text)
        self.after(100, self._poll_levels)

    @staticmethod
    def _resolve_output_dir(base_name: str) -> Path:
        """Return a Path for base_name, disambiguated with a _2, _3, ... suffix
        if a non-empty directory of that name already exists (minute-granularity
        timestamps in build_timestamped_name can collide on rapid restart)."""
        candidate = OUTPUT_ROOT / base_name
        if not candidate.exists() or not any(candidate.iterdir()):
            return candidate
        n = 2
        while True:
            candidate = OUTPUT_ROOT / f"{base_name}_{n}"
            if not candidate.exists() or not any(candidate.iterdir()):
                return candidate
            n += 1

    def _update_backend_widgets(self) -> None:
        whisper = BACKENDS[self.backend_combo.current()][1] == BACKEND_WHISPERX
        state = "readonly" if whisper else "disabled"
        self.model_combo.config(state=state)
        self.diar_combo.config(state=state)

    def _set_status(self, text: str) -> None:
        """Static status text; also ends any running activity indicator."""
        self._end_activity()
        self.status_var.set(text)
        self.update_idletasks()

    def _begin_activity(self, text: str) -> None:
        """Long-running step: animated bar plus a live elapsed-seconds counter."""
        self._activity_text = text
        if self._activity_began is None:
            self._activity_began = time.monotonic()
            self.activity_bar.pack(side="right")
            self.activity_bar.start(12)
            self._tick_job = self.after(500, self._tick_activity)
        self._tick_activity(reschedule=False)

    def _set_activity_text(self, text: str) -> None:
        if self._activity_began is None:
            self._begin_activity(text)
        else:
            self._activity_text = text
            self._tick_activity(reschedule=False)

    def _end_activity(self) -> None:
        if self._activity_began is not None:
            self._activity_began = None
            self.activity_bar.stop()
            self.activity_bar.pack_forget()
        if self._tick_job is not None:
            self.after_cancel(self._tick_job)
            self._tick_job = None

    def _tick_activity(self, reschedule: bool = True) -> None:
        if self._activity_began is None:
            return
        elapsed = int(time.monotonic() - self._activity_began)
        self.status_var.set(f"{self._activity_text}  ({elapsed} с)")
        if reschedule:
            self._tick_job = self.after(500, self._tick_activity)

    def _setup_logging(self) -> None:
        for h in list(log.handlers):
            log.removeHandler(h)
            h.close()
        log.setLevel(logging.INFO)
        handler = logging.FileHandler(self._output_dir / "session.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(handler)

    def _build_options(self) -> SessionOptions | None:
        backend = BACKENDS[self.backend_combo.current()][1]
        language = LANGUAGES[self.lang_combo.current()][1]
        opts = SessionOptions(
            backend=backend, language=language, record_audio=self.record_wav_var.get(),
            output_dir=self._output_dir, deepgram_api_key=self._settings.deepgram_api_key,
            hf_token=self._settings.hf_token,
        )
        if backend == BACKEND_DEEPGRAM:
            if not opts.deepgram_api_key:
                messagebox.showerror(
                    "Нет API-ключа",
                    "DEEPGRAM_API_KEY не задан в .env.\nВыберите WhisperX или добавьте ключ.",
                )
                return None
            return opts
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            messagebox.showerror(
                "WhisperX не установлен",
                "Установите зависимости локального движка:\n"
                "  pip install -r requirements-whisperx.txt\n(см. README)",
            )
            return None
        opts.diarization = DIARIZATION_MODES[self.diar_combo.current()][1]
        if opts.diarization != DIARIZE_OFF and not opts.hf_token:
            if not messagebox.askyesno(
                "Нет HF_TOKEN",
                "Для разбора по голосам нужен токен Hugging Face (HF_TOKEN в .env, см. README).\n"
                "Продолжить без разбора по голосам?",
            ):
                return None
            opts.diarization = DIARIZE_OFF
        model_override = WHISPER_MODELS[self.model_combo.current()][1]
        opts.whisper_model, opts.whisper_device, opts.whisper_compute_type = resolve_whisper_runtime(
            self._settings, model_override
        )
        return opts

    def _on_start(self) -> None:
        if not self._mic_devices or not self._system_devices:
            messagebox.showerror("Ошибка", "Не найдено устройство микрофона или системного звука.")
            return

        session_name = build_timestamped_name(self.session_name_var.get(), "DeepGramMeeting")
        self._output_dir = self._resolve_output_dir(session_name)
        self._output_dir.mkdir(parents=True, exist_ok=True)  # also creates Output/
        options = self._build_options()
        if options is None:
            return
        self._setup_logging()
        safe = {k: v for k, v in vars(options).items() if k not in ("deepgram_api_key", "hf_token")}
        log.info("start: %s", safe)

        for child in list(self.speakers_container.winfo_children()):
            child.destroy()
        self._speaker_rows.clear()
        self._muted = False
        self.mute_button.config(text="Заглушить микрофон")
        self.reexport_button.config(state="disabled")
        self._last_session = None

        mic = self._mic_devices[self.mic_combo.current()]
        system = self._system_devices[self.system_combo.current()]
        self._session = TranscriptionSession(mic, system, options)
        self._view_store = self._session.store
        self._interims.clear()
        self._render_view()
        self.start_button.config(state="disabled")

        self._heard_speech = False
        if options.backend == BACKEND_WHISPERX:
            self._begin_activity(
                "Загрузка моделей… (при первом запуске они скачиваются — несколько ГБ, ход виден в окне консоли)"
            )
        else:
            self._begin_activity("Подключение…")
        session = self._session

        def prepare():
            try:
                began = time.monotonic()
                session.prepare(progress=lambda t: self._work_queue.put(("status", t, None)))
                log.info("models ready in %.1fs", time.monotonic() - began)
                self._work_queue.put(("prepared", session, None))
            except Exception as exc:  # noqa: BLE001
                log.exception("prepare failed")
                self._work_queue.put(("prepared", session, exc))

        self._start_background_work(prepare)

    def _start_background_work(self, target) -> None:
        """Run target on a daemon thread and poll self._work_queue until the
        work reports completion (which clears self._busy)."""
        already_polling = self._busy
        self._busy = True
        threading.Thread(target=target, daemon=True).start()
        if not already_polling:
            self.after(150, self._poll_work)

    def _finish_start(self, session: TranscriptionSession, error: Exception | None) -> None:
        self._busy = False
        if session is not self._session:
            return
        if error is None:
            # Release the devices held by the level monitor before the session opens them.
            self._stop_monitor()
            try:
                session.start()
            except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
                log.exception("start failed")
                error = exc
        if error is not None:
            try:
                session.stop()
            except Exception:
                # Best-effort teardown of a partially-open session; the
                # original error below is what matters to the user.
                pass
            if session.options.backend == BACKEND_DEEPGRAM:
                title = "Ошибка подключения к Deepgram"
            else:
                title = "Ошибка запуска WhisperX"
            messagebox.showerror(title, f"{error}\n\nПодробности: {self._output_dir / 'session.log'}")
            self._session = None
            self._set_status("")
            self.start_button.config(state="normal")
            self._start_monitor()
            return
        self._set_status("Идёт запись…")
        self._recording_began = time.monotonic()
        self._recording = True
        self.stop_button.config(state="normal")
        self.mute_button.config(state="normal")
        self.after(100, self._poll_events)

    def _poll_work(self) -> None:
        try:
            while True:
                kind, payload, extra = self._work_queue.get_nowait()
                if kind == "prepared":
                    self._finish_start(payload, extra)
                elif kind == "status":
                    self._set_activity_text(payload)
                elif kind == "stopped":
                    self._busy = False
                    stop_error, diar_error = extra
                    self._finish_stop(payload, stop_error, diar_error)
        except queue.Empty:
            pass
        if self._busy:
            self.after(150, self._poll_work)

    def _on_stop(self) -> None:
        if self._session is None:
            return
        session = self._session
        self._session = None
        self._recording = False
        self.stop_button.config(state="disabled")
        self.mute_button.config(state="disabled")
        self._begin_activity("Остановка, доработка последних реплик…")

        # session.stop() can block for a long time with WhisperX (it waits for
        # the recognizer to finish the backlog of segments), so it runs on a
        # worker thread together with the optional diarization pass; the GUI
        # keeps responding and shows progress via the "status" work events.
        wav = self._output_dir / "system.wav"
        opts = session.options
        status = lambda t: self._work_queue.put(("status", t, None))  # noqa: E731

        def work():
            stop_error = None
            diar_error = None
            try:
                session.stop()
            except Exception as exc:  # noqa: BLE001
                log.exception("stop failed")
                stop_error = exc
            if session.needs_diarization_pass and wav.exists():
                status("Разбор по голосам…")
                try:
                    from teams_transcribe.postprocess import diarize_system_wav

                    n = diarize_system_wav(
                        wav, session.store, opts.hf_token, opts.whisper_device, progress=status
                    )
                    log.info("diarization pass: %s speakers", n)
                except Exception as exc:  # noqa: BLE001
                    log.exception("diarization pass failed")
                    diar_error = exc
            self._work_queue.put(("stopped", session, (stop_error, diar_error)))

        self._start_background_work(work)

    def _show_trailing_events(self, session: TranscriptionSession) -> None:
        """Display utterances that arrived while the session was shutting down
        (_poll_events stops as soon as the session is detached)."""
        while True:
            try:
                event = session.events.get_nowait()
            except queue.Empty:
                return
            if event.kind in ("new_speaker", "utterance"):
                self._handle_event(event, session.store)

    def _finish_stop(
        self,
        session: TranscriptionSession,
        stop_error: Exception | None,
        diar_error: Exception | None,
    ) -> None:
        try:
            self._show_trailing_events(session)
            if stop_error is not None:
                messagebox.showwarning("Предупреждение", f"Ошибка при остановке сессии: {stop_error}")
            if diar_error is not None:
                messagebox.showwarning(
                    "Разбор по голосам не удался",
                    f"{diar_error}\n\nТранскрипт сохранён с разметкой, полученной во время звонка.",
                )
            elif session.needs_diarization_pass:
                # Speakers were re-labelled offline: rebuild the (still editable)
                # speaker list with the new names.
                self._rebuild_speaker_rows(session.store)
            # Whatever was still interim when the connection closed never became final.
            self._interims.clear()
            self._render_view()
            if session.needs_diarization_pass and not session.options.record_audio:
                (self._output_dir / "system.wav").unlink(missing_ok=True)
            self._last_session = session
            self._last_output_dir = self._output_dir
            echo_count = self._export(session.store, self._output_dir)
            note = f"\n\nУбрано реплик-эхо микрофона: {echo_count}" if echo_count else ""
            messagebox.showinfo("Готово", f"Транскрипт сохранён в:\n{self._output_dir.resolve()}{note}")
        except Exception as exc:
            log.exception("export failed")
            messagebox.showerror("Ошибка экспорта", str(exc))
        finally:
            self._set_status("")
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.mute_button.config(state="disabled")
            if self._last_session is not None:
                self.reexport_button.config(state="normal")
            self._start_monitor()

    def _merge_gap(self) -> float:
        try:
            return max(0.0, float(self.merge_gap_var.get()))
        except (tk.TclError, ValueError):  # the spinbox is mid-edit / not a number
            return DEFAULT_MERGE_GAP

    def _export(self, store: TranscriptStore, output_dir: Path) -> int:
        echo_count = export_session(
            store, output_dir, drop_mic_echo=self.drop_echo_var.get(), merge_gap=self._merge_gap()
        )
        log.info("export: %s mic utterances dropped as echo", echo_count)
        return echo_count

    def _on_reexport(self) -> None:
        """Re-save the last finished session after the user edited speaker names."""
        if self._last_session is None or self._last_output_dir is None:
            return
        try:
            self._export(self._last_session.store, self._last_output_dir)
            self._render_view()
            messagebox.showinfo("Готово", f"Транскрипт сохранён заново в:\n{self._last_output_dir.resolve()}")
        except Exception as exc:  # noqa: BLE001
            log.exception("re-export failed")
            messagebox.showerror("Ошибка экспорта", str(exc))

    def _rebuild_speaker_rows(self, store: TranscriptStore) -> None:
        for child in list(self.speakers_container.winfo_children()):
            child.destroy()
        self._speaker_rows.clear()
        for key, name in store.system_speakers():
            self._add_speaker_row(key, name, store)

    def _schedule_render(self) -> None:
        """Coalesce redraws: many events/keystrokes within 50 ms cost one render."""
        if not self._render_pending:
            self._render_pending = True
            self.after(50, self._render_view)

    def _render_view(self) -> None:
        """Redraw the transcript box from _view_store: merged final lines with the
        speakers' current names, then the in-progress (interim) lines in grey.
        Keeps the user's scroll position unless they were already at the bottom."""
        self._render_pending = False
        box = self.transcript_box
        at_bottom = box.yview()[1] >= 0.999
        first_visible = box.yview()[0]
        store = self._view_store
        chunks: list = []
        if store is not None:
            for line in merge_utterances(store.final_utterances(), max_gap=self._merge_gap()):
                chunks += [f"[{line.timestamp:%H:%M:%S}] ", "time",
                           store.speaker_name(line.speaker_key), "name", f": {line.text}\n", ()]
            for key, text in self._interims.items():
                chunks += [store.speaker_name(key), "name", f": {text} …\n", "interim"]
        box.config(state="normal")
        box.delete("1.0", "end")
        if chunks:
            box.insert("end", *chunks)
        if at_bottom:
            box.see("end")
        else:
            box.yview_moveto(first_visible)
        box.config(state="disabled")

    def _on_close(self) -> None:
        if self._session is not None:
            if not messagebox.askyesno(
                "Завершить сеанс?",
                "Идёт запись. Закрыть окно сейчас и сохранить то, что уже записано?",
            ):
                return
            self._on_stop()
        if self._busy:
            messagebox.showinfo(
                "Подождите",
                "Идёт загрузка или разбор по голосам. Окно можно закрыть, когда появится сообщение «Готово».",
            )
            return
        self._stop_monitor()
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
                self._handle_event(event, self._session.store)
                if self._session is None:
                    # _handle_event triggered an auto-stop (e.g. an error
                    # event), tearing down the session - don't keep pulling
                    # from what may now be a stale/torn-down session's queue.
                    break
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _handle_event(self, event, store: TranscriptStore) -> None:
        if event.kind == "new_speaker":
            self._add_speaker_row(event.speaker_key, event.speaker_name, store)
        elif event.kind == "utterance":
            self._heard_speech = True
            # The final text is already in the store; the view only tracks interims.
            if event.is_final:
                self._interims.pop(event.speaker_key, None)
            else:
                self._interims[event.speaker_key] = event.text
            self._schedule_render()
        elif event.kind == "error":
            messagebox.showerror("Ошибка соединения", f"Сессия остановлена из-за ошибки:\n{event.text}")
            self._on_stop()

    def _add_speaker_row(self, speaker_key, default_name: str, store: TranscriptStore) -> None:
        """One editable row per speaker. Renames go straight to the given store,
        which keeps working after the session has stopped (for re-export)."""
        if speaker_key in self._speaker_rows:
            return
        row = ttk.Frame(self.speakers_container)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=default_name + ":", foreground="#555").pack(side="left")
        var = tk.StringVar(value=default_name)

        def on_change(*_args, key=speaker_key, var=var, store=store):
            name = var.get().strip()
            if name:
                store.rename(key, name)
                self._schedule_render()  # relabel the speaker's earlier lines too

        entry = tk.Entry(row, textvariable=var, width=16)
        entry.pack(side="left", padx=4)
        var.trace_add("write", on_change)
        self._speaker_rows[speaker_key] = entry


def run(settings: Settings) -> None:
    app = App(settings)
    app.mainloop()
