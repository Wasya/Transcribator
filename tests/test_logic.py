import unittest
from datetime import datetime

import numpy as np

from teams_transcribe.audio_utils import Resampler, Segmenter, TARGET_RATE
from teams_transcribe.postprocess import assign_speakers
from teams_transcribe.speaker_id import OnlineSpeakerClusterer, OnlineSpeakerIdentifier
from teams_transcribe.transcript_store import TranscriptStore


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


if __name__ == "__main__":
    unittest.main()
