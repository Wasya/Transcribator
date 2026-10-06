import unittest
import unittest.mock
from datetime import datetime

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


if __name__ == "__main__":
    unittest.main()
