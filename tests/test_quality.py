import unittest
from unittest.mock import patch

from studio.quality import finalize_cues, quality_summary
from studio.recall import vad_fallback_events_for_gaps
from studio.schemas import ProviderSettings
from studio.translation import audit_translation, safe_high_risk, translate_cues


class QualityOptimizationTests(unittest.TestCase):
    def test_dense_short_cues_are_merged_without_overlap(self):
        rows = finalize_cues(
            [
                {"start": 0.0, "end": 0.3, "source": "あ", "zh": "啊"},
                {"start": 0.1, "end": 0.95, "source": "や", "zh": "呀"},
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["zh"], "啊，呀")
        self.assertGreaterEqual(rows[0]["end"], 0.9)

    def test_adjacent_exact_duplicates_are_collapsed(self):
        rows = finalize_cues(
            [
                {"start": 0.0, "end": 0.5, "source": "はい", "zh": "好的"},
                {"start": 0.54, "end": 1.2, "source": "はい", "zh": "好的"},
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["zh"], "好的")

    def test_same_start_cues_cannot_leave_an_overlap(self):
        rows = finalize_cues(
            [
                {"start": 2.0, "end": 2.5, "source": "第一句", "zh": "这是一条很长很长的第一句字幕内容"},
                {"start": 2.0, "end": 3.0, "source": "第二句", "zh": "这是一条很长很长的第二句字幕内容"},
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(sum(rows[i]["end"] > rows[i + 1]["start"] for i in range(len(rows) - 1)), 0)

    def test_review_marks_translation_warnings_but_publish_does_not(self):
        cue = {
            "start": 0.0,
            "end": 1.0,
            "source": "かきたちゃん",
            "zh": "柿田酱",
            "translation_warnings": ["人名需复核"],
        }
        review = finalize_cues([cue], publish=False)
        publish = finalize_cues([cue], publish=True)
        self.assertTrue(review[0]["zh"].startswith("【需校对】"))
        self.assertNotIn("【需校对】", publish[0]["zh"])

    def test_quality_summary_measures_vad_activity_coverage(self):
        summary = quality_summary(
            [{"start": 1.0, "end": 3.0, "zh": "对白"}],
            40.0,
            activity_segments=[
                {"start": 0.0, "end": 2.0},
                {"start": 4.0, "end": 6.0},
            ],
        )
        self.assertEqual(summary["activity_seconds"], 4.0)
        self.assertEqual(summary["covered_activity_seconds"], 1.0)
        self.assertEqual(summary["activity_coverage_percent"], 25.0)

    def test_vad_fallback_only_returns_unreviewed_gap_pieces(self):
        rows = vad_fallback_events_for_gaps(
            [{"start": 10.0, "end": 14.0}, {"start": 30.0, "end": 32.0}],
            [{"start": 9.0, "end": 20.0, "duration": 11.0}],
            [{"start": 11.0, "end": 13.0}],
        )
        self.assertEqual([(x["start"], x["end"]) for x in rows], [(10.0, 11.0), (13.0, 14.0)])
        self.assertTrue(all(x["source"] == "vad_gap_fallback" for x in rows))

    def test_japanese_action_direction_is_audited_and_has_safe_fallback(self):
        self.assertTrue(audit_translation("あ、抜けちゃった", "啊，搞砸了", "ja"))
        self.assertEqual(safe_high_risk("あ、抜けちゃった", "啊，搞砸了"), "啊，掉出来了")
        self.assertTrue(audit_translation("入れます", "不要这样", "ja"))
        self.assertEqual(safe_high_risk("入れます", "不要这样"), "要放进去了")

    def test_source_script_gets_a_focused_third_repair(self):
        class FakeProvider:
            def __init__(self):
                self.calls = 0

            def chat_json(self, model, prompt, request):
                self.calls += 1
                if self.calls < 3:
                    return {"id": 1, "zh": "我在看かきた酱的照片"}
                return {"id": 1, "zh": "我在看柿田酱的照片"}

        provider = FakeProvider()
        settings = ProviderSettings(kind="local_ollama", model="test")
        with patch("studio.translation.provider_from_settings", return_value=provider):
            rows = translate_cues(
                [{"start": 0, "end": 1, "source": "かきたちゃんの写真を見てる"}],
                settings,
            )
        self.assertEqual(provider.calls, 3)
        self.assertEqual(rows[0]["zh"], "我在看柿田酱的照片")
        self.assertNotIn("translation_warnings", rows[0])


if __name__ == "__main__":
    unittest.main()
