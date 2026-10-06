import unittest
import unittest.mock
from datetime import datetime, timedelta

import numpy as np

from teams_transcribe.audio_utils import Resampler, Segmenter, TARGET_RATE
from teams_transcribe.echo_filter import find_mic_echo
from teams_transcribe.exporter import export_session
from teams_transcribe.postprocess import assign_speakers
from teams_transcribe.speaker_id import OnlineSpeakerClusterer, OnlineSpeakerIdentifier
from teams_transcribe.transcript_store import TranscriptStore, Utterance


def tone(seconds, amp=0.3, rate=TARGET_RATE):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds, rate=TARGET_RATE):
    return np.zeros(int(seconds * rate), dtype=np.float32)


class SegmenterTests(unittest.TestCase):
    def test_splits_on_pause(self):
        seg = Segmenter()
        audio = np.concatenate([silence(0.5), tone(1.0), silence(1.0), tone(1.5), silence(1.0)])
        out = seg.feed(audio) + seg.flush()
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0].start, 0.2, delta=0.1)  # includes pre-roll
        self.assertGreater(out[1].start, out[0].end)

    def test_flush_returns_pending_speech(self):
        seg = Segmenter()
        self.assertEqual(seg.feed(tone(1.0)), [])
        self.assertEqual(len(seg.flush()), 1)

    def test_ignores_short_blips_and_silence(self):
        seg = Segmenter()
        out = seg.feed(np.concatenate([silence(1), tone(0.1), silence(2)])) + seg.flush()
        self.assertEqual(out, [])

    def test_long_speech_is_cut(self):
        seg = Segmenter(max_segment=2.0)
        out = seg.feed(tone(5.0)) + seg.flush()
        self.assertGreaterEqual(len(out), 2)
        for a, b in zip(out, out[1:]):
            self.assertAlmostEqual(a.end, b.start, delta=0.001)  # no audio lost between parts

    def test_forced_cut_lands_on_quietest_point(self):
        # 6 s of continuous "speech" with a 0.2 s dip at 4.0 s: too short to count as a
        # pause (min_silence=0.7), but the forced cut at max_segment=5 s must use it
        # instead of chopping blindly at 5.0 s.
        seg = Segmenter(max_segment=5.0, cut_search=2.0)
        audio = np.concatenate([tone(4.0), tone(0.2, amp=0.02), tone(1.8)])
        out = seg.feed(audio) + seg.flush()
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0].end, 4.1, delta=0.15)
        self.assertAlmostEqual(out[1].start, out[0].end, delta=0.001)
        self.assertAlmostEqual(out[1].end, 6.0, delta=0.05)


class QuietAudioTests(unittest.TestCase):
    def test_quiet_speech_is_detected(self):
        seg = Segmenter()
        out = seg.feed(np.concatenate([silence(1.0), tone(2.0, amp=0.007), silence(1.5)])) + seg.flush()
        self.assertEqual(len(out), 1)  # rms ~0.005: below the old fixed threshold of 0.006

    def test_soft_continuous_speech_does_not_raise_the_threshold(self):
        seg = Segmenter()
        rng = np.random.default_rng(0)
        speech = (rng.standard_normal(TARGET_RATE * 60) * 0.01).astype(np.float32)  # rms 0.01 for a minute
        seg.feed(speech)
        stats = seg.pop_stats()
        self.assertGreater(stats["speech_pct"], 90)
        self.assertLessEqual(stats["threshold"], 0.006)

    def test_normalize_level_only_amplifies_quiet_audio(self):
        from teams_transcribe.audio_utils import normalize_level

        quiet = tone(1.0, amp=0.05)
        self.assertAlmostEqual(float(np.max(np.abs(normalize_level(quiet)))), 0.5, places=2)
        loud = tone(1.0, amp=0.8)
        self.assertIs(normalize_level(loud), loud)
        zeros = silence(1.0)
        self.assertIs(normalize_level(zeros), zeros)


class Mp3ExportTests(unittest.TestCase):
    def _write_wav(self, path, seconds=2.0, rate=48000):
        import wave

        samples = (tone(seconds, amp=0.3, rate=rate) * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(samples.tobytes())

    def test_compress_session_audio_replaces_wavs_with_mp3(self):
        import tempfile
        from pathlib import Path

        from teams_transcribe.audio_export import compress_session_audio

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write_wav(d / "mic.wav")
            self._write_wav(d / "system.wav", rate=44100)
            self.assertEqual(compress_session_audio(d), [])
            self.assertEqual(sorted(p.name for p in d.iterdir()), ["mic.mp3", "system.mp3"])
            self.assertGreater((d / "system.mp3").stat().st_size, 1000)

    def test_failure_keeps_the_wav(self):
        import tempfile
        from pathlib import Path

        from teams_transcribe.audio_export import compress_session_audio

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "mic.wav").write_bytes(b"not a wav file")
            self.assertEqual(compress_session_audio(d), ["mic.wav"])
            self.assertTrue((d / "mic.wav").exists())
            self.assertFalse((d / "mic.mp3").exists())


class PromptContextTests(unittest.TestCase):
    def test_context_is_used_then_goes_stale(self):
        from teams_transcribe.whisper_stream import PromptContext

        c = PromptContext(max_age=10.0)
        self.assertIsNone(c.get(0.0))
        c.update("Россельхознадзор обратился в прокуратуру", segment_end=8.0)
        self.assertEqual(c.get(8.2), "Россельхознадзор обратился в прокуратуру")
        self.assertIsNone(c.get(30.0))  # a long pause: do not carry the old topic over

    def test_long_context_is_trimmed_at_a_word_boundary(self):
        from teams_transcribe.whisper_stream import PromptContext

        c = PromptContext(max_chars=20)
        c.update("раз два три четыре пять шесть", segment_end=1.0)
        tail = c.get(1.0)
        self.assertLessEqual(len(tail), 20)
        self.assertIn(tail, "раз два три четыре пять шесть")
        self.assertTrue("раз два три четыре пять шесть".endswith(tail))
        self.assertFalse(tail.startswith(("ре", "ять")))  # never a cut-off word

    def test_prompt_echo_is_dropped_but_real_text_kept(self):
        from teams_transcribe.whisper_stream import strip_prompt_echo

        prompt = "Россельхознадзор обратился в прокуратуру"
        self.assertEqual(strip_prompt_echo("обратился в прокуратуру", prompt), "")
        self.assertEqual(strip_prompt_echo("Препарат отозвали до завершения расследования", prompt),
                         "Препарат отозвали до завершения расследования")
        self.assertEqual(strip_prompt_echo("да", prompt), "да")
        self.assertEqual(strip_prompt_echo("текст", None), "текст")


class WhisperStreamStopTests(unittest.TestCase):
    class FakeEngine:
        def __init__(self, delay, block=False):
            self.delay, self.block, self.prompts = delay, block, []

        def transcribe(self, audio, language, prompt=None):
            import time

            self.prompts.append(prompt)
            if self.block:
                time.sleep(30)
            time.sleep(self.delay)
            return "слово"

    def _run(self, engine, stall_timeout, seconds=20, use_context=True):
        from teams_transcribe.whisper_stream import WhisperStream

        results = []
        stream = WhisperStream(engine, language="ru", sample_rate=16000, use_context=use_context,
                               on_result=lambda s, d, f, spk, t: results.append(t))
        stream.start()
        speech = (tone(seconds, amp=0.3) * 32767).astype(np.int16).tobytes()
        for i in range(0, len(speech), 4096):
            stream.send(speech[i:i + 4096])
        import time

        began = time.monotonic()
        stream.stop(stall_timeout=stall_timeout)
        return results, time.monotonic() - began

    def test_stop_waits_for_the_whole_backlog(self):
        engine = self.FakeEngine(delay=0.4)
        results, waited = self._run(engine, stall_timeout=5.0)  # ~3 segments x 0.4 s, several seconds of lag overall
        self.assertGreaterEqual(len(results), 3)
        self.assertEqual(len(engine.prompts), len(results))
        self.assertIsNone(engine.prompts[0])
        self.assertTrue(all(p == "слово" for p in engine.prompts[1:]))  # context passed on

    def test_context_is_off_by_default(self):
        from teams_transcribe.whisper_stream import WhisperStream

        engine = self.FakeEngine(delay=0)
        stream = WhisperStream(engine, language="ru", sample_rate=16000, on_result=lambda *a: None)
        stream.start()
        speech = (tone(20, amp=0.3) * 32767).astype(np.int16).tobytes()
        for i in range(0, len(speech), 4096):
            stream.send(speech[i:i + 4096])
        stream.stop()
        self.assertGreaterEqual(len(engine.prompts), 2)
        self.assertTrue(all(p is None for p in engine.prompts))

    def test_stop_gives_up_on_a_stuck_recognizer(self):
        results, waited = self._run(self.FakeEngine(delay=0, block=True), stall_timeout=1.0, seconds=9)
        self.assertEqual(results, [])
        self.assertLess(waited, 10)


class FileModeTests(unittest.TestCase):
    class FakeEngine:
        def __init__(self):
            self.calls = 0

        def transcribe(self, audio, language, prompt=None):
            self.calls += 1
            return f"фраза{self.calls}"

    def _speech(self):
        # two bursts of "speech" separated by a long pause -> two segments
        return np.concatenate([silence(0.5), tone(2.0), silence(1.5), tone(2.0), silence(1.0)])

    def test_pipeline_runs_on_in_memory_audio(self):
        from teams_transcribe.file_mode import FileRunOptions, transcribe_audio

        opts = FileRunOptions(model="x", device="cpu", compute_type="int8", diarization="off")
        result = transcribe_audio(self._speech(), opts, engine=self.FakeEngine())
        self.assertEqual(result.segments, 2)
        self.assertEqual(result.words, 2)
        self.assertEqual(result.text(), "фраза1 фраза2")
        self.assertEqual(result.speakers, 1)
        self.assertAlmostEqual(result.audio_seconds, 7.0, places=1)

    def test_post_diarization_is_applied_with_injected_diarizer(self):
        from teams_transcribe.file_mode import FileRunOptions, transcribe_audio

        def diarizer(audio, token, device):
            return [(0.0, 3.0, "A"), (3.0, 7.0, "B")]

        opts = FileRunOptions(model="x", device="cpu", compute_type="int8", diarization="post")
        result = transcribe_audio(self._speech(), opts, engine=self.FakeEngine(), diarizer=diarizer)
        self.assertEqual(result.speakers, 2)

    def test_post_diarization_without_token_is_skipped_with_a_note(self):
        from teams_transcribe.file_mode import FileRunOptions, transcribe_audio

        opts = FileRunOptions(model="x", device="cpu", compute_type="int8", diarization="post", hf_token=None)
        result = transcribe_audio(self._speech(), opts, engine=self.FakeEngine())
        self.assertEqual(result.speakers, 1)
        self.assertTrue(any("HF_TOKEN" in n for n in result.notes))

    def test_compare_texts_finds_missing_and_extra_phrases(self):
        from teams_transcribe.file_mode import compare_texts

        ref = "раз два три четыре пять шесть семь восемь девять десять"
        hyp = "раз два три девять десять одиннадцать двенадцать тринадцать"
        cmp = compare_texts(ref, hyp)
        self.assertIn("четыре пять шесть семь восемь", cmp["missing"])
        self.assertIn("одиннадцать двенадцать тринадцать", cmp["extra"])
        self.assertAlmostEqual(compare_texts(ref, ref)["wer"], 0.0)
        self.assertGreater(cmp["wer"], 0.2)

    def test_reference_loader_strips_transcript_prefixes(self):
        import tempfile
        from pathlib import Path

        from teams_transcribe.file_mode import load_reference_text

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.txt"
            lines = ["[00:00:05] Собеседник 1: привет мир", "[00:00:09] Иван: как дела", "просто строка"]
            p.write_text(chr(10).join(lines), encoding="utf-8")
            self.assertEqual(load_reference_text(p), "привет мир как дела просто строка")


class DiarizationScoringTests(unittest.TestCase):
    TURNS = [(0.0, "Анна"), (10.0, "Борис"), (20.0, "Вера")]  # reference: who starts speaking when

    def _utts(self, labels):
        """Six 5-second utterances: 0-5 and 5-10 Anna, 10-15 and 15-20 Boris, 20-25 and 25-30 Vera."""
        from teams_transcribe.transcript_store import Utterance

        return [
            Utterance(("system", lab), datetime(2000, 1, 1) + timedelta(seconds=i * 5), True, "x", i * 5.0, 5.0)
            for i, lab in enumerate(labels)
        ]

    def test_perfect_diarization(self):
        from teams_transcribe.file_mode import score_diarization

        r = score_diarization(self._utts([0, 0, 1, 1, 2, 2]), self.TURNS)
        self.assertEqual((r["found"], r["real"]), (3, 3))
        self.assertAlmostEqual(r["accuracy"], 1.0)
        self.assertEqual(r["merged"] + r["split"], [])

    def test_labels_are_matched_regardless_of_numbering(self):
        from teams_transcribe.file_mode import score_diarization

        self.assertAlmostEqual(score_diarization(self._utts([7, 7, 3, 3, 9, 9]), self.TURNS)["accuracy"], 1.0)

    def test_two_people_merged(self):
        from teams_transcribe.file_mode import score_diarization

        r = score_diarization(self._utts([0, 0, 1, 1, 1, 1]), self.TURNS)  # Boris and Vera are one voice
        self.assertEqual(r["found"], 2)
        self.assertAlmostEqual(r["accuracy"], 4 / 6, places=2)  # Anna + Boris right, Vera lost
        self.assertLess(r["purity"], 1.0)
        self.assertAlmostEqual(r["completeness"], 1.0)
        self.assertEqual(r["merged"], [("Г2", ["Борис", "Вера"])])

    def test_one_person_split(self):
        from teams_transcribe.file_mode import score_diarization

        r = score_diarization(self._utts([0, 1, 2, 2, 3, 3]), self.TURNS)  # Anna got two voices
        self.assertEqual(r["found"], 4)
        self.assertLess(r["completeness"], 1.0)
        self.assertAlmostEqual(r["purity"], 1.0)
        self.assertEqual(r["split"], [("Анна", ["Г1", "Г2"])])

    def test_explicit_offset_shifts_the_reference(self):
        from teams_transcribe.file_mode import score_diarization

        # the audio starts 5 s before the first reference line: Anna's turn is 5..15
        utts = self._utts([0, 0, 1, 1, 2, 2])
        self.assertLess(score_diarization(utts, self.TURNS, offset=5.0)["accuracy"], 1.0)

    def test_reference_turns_are_parsed(self):
        import tempfile
        from pathlib import Path

        from teams_transcribe.file_mode import load_reference_turns

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.txt"
            p.write_text(chr(10).join(["[00:00:05] Анна: привет", "просто строка", "[01:02:03] Борис: пока", "[07:30] Вера: тест"]), encoding="utf-8")
            self.assertEqual(load_reference_turns(p), [(5.0, "Анна"), (3723.0, "Борис"), (450.0, "Вера")])
            p.write_text("без разметки", encoding="utf-8")
            self.assertEqual(load_reference_turns(p), [])


class ResamplerTests(unittest.TestCase):
    def test_48k_to_16k_length_and_chunk_continuity(self):
        x = tone(1.0, rate=48000)
        whole = Resampler(48000).process(x)
        r = Resampler(48000)
        parts = np.concatenate([r.process(x[i:i + 1024]) for i in range(0, len(x), 1024)])
        self.assertAlmostEqual(len(whole), 16000, delta=3)
        self.assertAlmostEqual(len(parts), 16000, delta=3)

    def test_44k1_length(self):
        out = Resampler(44100).process(tone(1.0, rate=44100))
        self.assertAlmostEqual(len(out), 16000, delta=5)


class ClustererTests(unittest.TestCase):
    def test_groups_similar_vectors(self):
        c = OnlineSpeakerClusterer(threshold=0.7)
        a, b = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        ids = [c.assign(a), c.assign(b), c.assign(a + 0.05), c.assign(b + 0.05)]
        self.assertEqual(ids, [0, 1, 0, 1])

    def test_short_segments_reuse_last_speaker(self):
        ident = OnlineSpeakerIdentifier(lambda audio: np.array([1.0, 0.0]), min_seconds=1.0)
        self.assertIsNone(ident(tone(0.5)))
        self.assertEqual(ident(tone(1.5)), 0)
        self.assertEqual(ident(tone(0.5)), 0)


class DiarizationAssignTests(unittest.TestCase):
    def _store(self):
        s = TranscriptStore()
        t0 = datetime(2026, 1, 1, 10, 0, 0)
        for i, (start, dur) in enumerate([(0.0, 2.0), (3.0, 2.0), (6.0, 2.0)]):
            s.get_or_create_system_speaker_name(0)
            s.add_utterance(("system", 0), t0.replace(second=int(start)), True, f"u{i}", start, dur)
        return s

    def test_assign_by_overlap_and_apply_keeps_custom_name(self):
        s = self._store()
        s.rename(("system", 0), "Анна")
        turns = [(0.0, 2.5, "SPEAKER_01"), (2.5, 5.5, "SPEAKER_00"), (5.5, 9.0, "SPEAKER_01")]
        us = s.system_utterances()
        a = assign_speakers(us, turns)
        self.assertEqual([a[id(u)] for u in us], [0, 1, 0])  # numbered by first appearance
        s.apply_diarization(a)
        keys = [u.speaker_key for u in s.final_utterances()]
        self.assertEqual(keys, [("system", 0), ("system", 1), ("system", 0)])
        self.assertEqual(s.speaker_name(("system", 0)), "Анна")  # custom name carried over
        self.assertEqual(s.speaker_name(("system", 1)), "Собеседник 2")

    def test_no_overlap_falls_back_to_nearest(self):
        s = self._store()
        us = s.system_utterances()
        a = assign_speakers(us, [(20.0, 30.0, "A"), (9.0, 10.0, "B")])
        self.assertEqual(a[id(us[2])], a[id(us[2])])
        self.assertEqual(len(a), 3)


class EchoFilterTests(unittest.TestCase):
    T0 = datetime(2026, 10, 5, 9, 54, 0)

    def _utt(self, source, sec, text, dur=2.0):
        key = ("mic", None) if source == "mic" else ("system", 0)
        return Utterance(key, self.T0.replace(second=sec), True, text, float(sec), dur)

    def test_real_session_echo_is_detected_despite_different_segmentation(self):
        # Verbatim from Output/DeepGramMeeting05-10-2026_09-53: the mic heard the speakers.
        us = [
            self._utt("mic", 1, "Уже глупо говорить о том, что данные на биржах о выставляемых"),
            self._utt("sys", 2, "Уже глупо говорить о том, что данные на биржах о"),
            self._utt("sys", 5, "выставляемых проектах вообще ничему не соответствуют. Это просто неизвестно,"),
            self._utt("mic", 6, "вообще ничему не соответствует. Это просто неизвестно что"),
            self._utt("mic", 9, "там. Да. Это чистые фантазии."),
            self._utt("sys", 10, "что. Так. Да. Это чистые фантазии."),
            self._utt("sys", 13, "У нас там фильм назывался?"),
        ]
        echo = find_mic_echo(us)
        self.assertEqual(echo, {id(us[0]), id(us[3]), id(us[4])})

    def test_garbled_echo_is_caught_by_character_match(self):
        # Also from the real session: the mic cut words short ("так" vs "такое", "не" vs "нельзя").
        us = [
            self._utt("sys", 18, "Какие-то фантазии. Что это такое? Чему это соответствует?"),
            self._utt("mic", 19, "Что это так?"),
            self._utt("mic", 44, "Конечно, это не"),
            self._utt("sys", 44, "Конечно, это нельзя. Это называется инсайдерской информацией."),
        ]
        self.assertEqual(find_mic_echo(us), {id(us[1]), id(us[2])})

    def test_genuine_mic_speech_is_kept(self):
        us = [
            self._utt("sys", 0, "Как прошли выходные?"),
            self._utt("mic", 3, "Отлично, ездили за город на дачу."),
            self._utt("sys", 6, "Здорово, а у нас всё дождь."),
        ]
        self.assertEqual(find_mic_echo(us), set())

    def test_short_mic_utterance_needs_exact_match(self):
        us = [
            self._utt("sys", 0, "Договорились, тогда до завтра."),
            self._utt("mic", 1, "Да."),  # a genuine "yes" - not in the system text
            self._utt("mic", 2, "До завтра"),  # exact substring -> echo
        ]
        self.assertEqual(find_mic_echo(us), {id(us[2])})

    def test_same_words_far_apart_in_time_are_not_echo(self):
        us = [
            self._utt("sys", 0, "Давайте вернёмся к этому вопросу позже."),
            self._utt("mic", 40, "Давайте вернёмся к этому вопросу позже."),
        ]
        self.assertEqual(find_mic_echo(us), set())

    def test_export_drops_echo_from_txt_but_keeps_it_flagged_in_json(self):
        import json
        import tempfile
        from pathlib import Path

        s = TranscriptStore()
        s.register_mic()
        s.get_or_create_system_speaker_name(0)
        s.add_utterance(("system", 0), self.T0, True, "Сегодня обсуждаем бюджет на квартал", 0.0, 2.0)
        s.add_utterance(("mic", None), self.T0.replace(second=1), True, "обсуждаем бюджет на квартал", 1.0, 2.0)
        s.add_utterance(("mic", None), self.T0.replace(second=5), True, "Я подготовил цифры", 5.0, 2.0)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertEqual(export_session(s, out), 1)
            txt = (out / "transcript.txt").read_text(encoding="utf-8")
            self.assertNotIn("Я: обсуждаем бюджет", txt)
            self.assertIn("Я: Я подготовил цифры", txt)
            entries = json.loads((out / "transcript.json").read_text(encoding="utf-8"))
            self.assertEqual([e["mic_echo"] for e in entries], [False, True, False])
            self.assertEqual((out / "speakers" / "Я.txt").read_text(encoding="utf-8"), "Я подготовил цифры")
            self.assertEqual(export_session(s, out, drop_mic_echo=False), 0)


class LevelMeterTests(unittest.TestCase):
    def test_pcm16_rms(self):
        from teams_transcribe.audio_capture import pcm16_rms

        self.assertEqual(pcm16_rms(b""), 0.0)
        self.assertEqual(pcm16_rms(b"\x00\x00" * 100), 0.0)
        full = np.full(1000, 16384, dtype=np.int16).tobytes()  # half scale, constant
        self.assertAlmostEqual(pcm16_rms(full), 0.5, places=3)
        self.assertAlmostEqual(pcm16_rms(full + b"\x01"), 0.5, places=3)  # odd trailing byte ignored

    def test_level_to_percent_db_scale(self):
        from teams_transcribe.gui import level_to_percent

        self.assertEqual(level_to_percent(0.0), 0)
        self.assertEqual(level_to_percent(1.0), 100)
        self.assertEqual(level_to_percent(0.001), 0)  # -60 dB floor
        self.assertEqual(level_to_percent(0.1), 66)  # -20 dB
        self.assertEqual(level_to_percent(5.0), 100)  # clamped

    def test_level_monitor_survives_device_failure(self):
        from teams_transcribe.audio_capture import DeviceInfo, LevelMonitor

        class FakePA:
            def open(self, **kw):
                if kw["input_device_index"] == 7:
                    raise OSError("device busy")
                return unittest.mock.Mock()

        mic = DeviceInfo(index=7, name="mic", sample_rate=48000, channels=1)
        system = DeviceInfo(index=9, name="loop", sample_rate=48000, channels=2)
        mon = LevelMonitor(FakePA(), mic, system)
        mon.start()
        self.assertEqual(list(mon.errors()), ["mic"])
        self.assertEqual(mon.mic_level, 0.0)
        self.assertEqual(mon.system_level, 0.0)
        mon.stop()  # must not raise


class ModelCacheTests(unittest.TestCase):
    def test_engine_reused_for_same_settings_and_replaced_on_change(self):
        from unittest import mock

        from teams_transcribe import whisper_stream

        with mock.patch.object(whisper_stream.WhisperEngine, "load") as load, \
                mock.patch.object(whisper_stream, "_cached_engine", None):
            a = whisper_stream.get_engine("small", "cpu", "int8")
            b = whisper_stream.get_engine("small", "cpu", "int8")
            c = whisper_stream.get_engine("medium", "cpu", "int8")
        self.assertIs(a, b)
        self.assertIsNot(a, c)
        self.assertEqual(load.call_count, 3)  # load() itself is idempotent

    def test_embedder_reused_for_same_settings(self):
        from unittest import mock

        from teams_transcribe import speaker_id

        with mock.patch.object(speaker_id.PyannoteEmbedder, "load"), \
                mock.patch.object(speaker_id, "_cached_embedder", None):
            a = speaker_id.get_embedder("tok", "cpu")
            b = speaker_id.get_embedder("tok", "cpu")
            c = speaker_id.get_embedder("tok", "cuda")
        self.assertIs(a, b)
        self.assertIsNot(a, c)


class HallucinationFilterTests(unittest.TestCase):
    def test_removes_credits_seen_in_real_session(self):
        from teams_transcribe.hallucinations import clean_hallucinations as c

        self.assertEqual(c("Субтитры подогнал «Симон»"), "")
        self.assertEqual(c("Субтитры создавал DimaTorzok"), "")
        self.assertEqual(c("Продолжение следует..."), "")
        self.assertEqual(c("Редактор субтитров А.Семкин Корректор А.Егорова"), "")

    def test_cuts_phrase_but_keeps_real_speech(self):
        from teams_transcribe.hallucinations import clean_hallucinations as c

        self.assertEqual(c("Дневной рубеж. Субтитры подогнал «Симон»"), "Дневной рубеж.")
        self.assertEqual(c("Субтитры создавал DimaTorzok Помним, что вчера голосовали"), "Помним, что вчера голосовали")

    def test_real_speech_is_untouched(self):
        from teams_transcribe.hallucinations import clean_hallucinations as c

        for text in ("Продолжение следует завтра, коллеги.", "Нужны ли нам субтитры в этом видео?", "Спасибо, всем пока"):
            self.assertEqual(c(text), text)

    def test_english_and_german_and_empty(self):
        from teams_transcribe.hallucinations import clean_hallucinations as c

        self.assertEqual(c("Thanks for watching!"), "")
        self.assertEqual(c("Untertitel der Amara.org-Community"), "")
        self.assertEqual(c("..."), "")


class MergeUtterancesTests(unittest.TestCase):
    T0 = datetime(2026, 1, 1, 10, 0, 0)

    def _u(self, key, offset, dur, text):
        from teams_transcribe.transcript_store import Utterance

        return Utterance(key, self.T0 + timedelta(seconds=offset), True, text, offset, dur)

    def test_comma_pauses_of_one_speaker_form_one_line(self):
        from teams_transcribe.transcript_store import merge_utterances

        k = ("system", 0)
        lines = merge_utterances([self._u(k, 0, 2, "Привет,"), self._u(k, 4.5, 2, "как дела"), self._u(k, 9, 1, "сегодня")])
        self.assertEqual([l.text for l in lines], ["Привет, как дела сегодня"])  # 2.5 s and 2 s pauses

    def test_other_speaker_in_between_splits_lines(self):
        from teams_transcribe.transcript_store import merge_utterances

        a, b = ("system", 0), ("system", 1)
        lines = merge_utterances([self._u(a, 0, 2, "раз"), self._u(b, 2.5, 1, "два"), self._u(a, 4, 1, "три")])
        self.assertEqual([l.text for l in lines], ["раз", "два", "три"])

    def test_long_pause_starts_new_line_and_zero_disables_merging(self):
        from teams_transcribe.transcript_store import DEFAULT_MERGE_GAP, merge_utterances

        k = ("system", 0)
        us = [self._u(k, 0, 1, "a"), self._u(k, 1 + DEFAULT_MERGE_GAP + 1, 1, "b")]
        self.assertEqual(len(merge_utterances(us)), 2)
        close = [self._u(k, 0, 1, "a"), self._u(k, 1.5, 1, "b")]
        self.assertEqual(len(merge_utterances(close, max_gap=0)), 2)

    def test_line_length_is_capped(self):
        from teams_transcribe.transcript_store import merge_utterances

        k = ("system", 0)
        us = [self._u(k, i * 10, 9, f"w{i}") for i in range(12)]  # non-stop speech for 2 minutes
        lines = merge_utterances(us, max_seconds=90)
        self.assertGreater(len(lines), 1)
        self.assertTrue(all((l.end - l.timestamp).total_seconds() <= 90 for l in lines))

    def test_rename_relabels_every_earlier_line(self):
        s = TranscriptStore()
        k = ("system", 0)
        s.get_or_create_system_speaker_name(0)
        s.add_utterance(k, self.T0, True, "раз", 0, 1)
        s.rename(k, "Иван")
        self.assertEqual(s.speaker_name(s.final_utterances()[0].speaker_key), "Иван")


if __name__ == "__main__":
    unittest.main()
