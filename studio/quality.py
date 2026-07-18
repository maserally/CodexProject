from __future__ import annotations

import re
from typing import Any

from .languages import source_text


PROFILE_SETTINGS = {
    "precision": {
        "speech_threshold": 0.20,
        "nonlexical_factor": 1.35,
        "recovery_threshold": 0.12,
        "consensus_threshold": 0.65,
        "vad_gap_fallback": False,
    },
    "balanced": {
        "speech_threshold": 0.15,
        "nonlexical_factor": 1.20,
        "recovery_threshold": 0.08,
        "consensus_threshold": 0.52,
        "vad_gap_fallback": True,
    },
    "recall": {
        "speech_threshold": 0.10,
        "nonlexical_factor": 1.05,
        "recovery_threshold": 0.045,
        "consensus_threshold": 0.40,
        "vad_gap_fallback": True,
    },
}


def strip_chinese_periods(text: str) -> str:
    return re.sub(r"。|\.(?=\s*$)", "", text).strip()


def publish_text(text: str) -> str:
    text = text.replace("【听不清】", "").replace("[听不清]", "")
    text = text.replace("【地点名不明】", "").replace("【场所名不明】", "")
    text = text.replace("【人名不明】", "有人")
    text = re.sub(r"。|\.(?=\s*$)", "", text)
    return re.sub(r"\s+", " ", text).strip(" ，、")


def _combine_rows(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    row = dict(left)
    row["end"] = max(float(left["end"]), float(right["end"]))
    left_source = source_text(left).strip()
    right_source = source_text(right).strip()
    row["source"] = " ".join(x for x in (left_source, right_source) if x)
    row.pop("ja", None)
    left_zh = str(left.get("zh", "")).strip()
    right_zh = str(right.get("zh", "")).strip()
    row["zh"] = left_zh if left_zh == right_zh else "，".join(x for x in (left_zh, right_zh) if x)
    warnings = list(dict.fromkeys(
        list(left.get("translation_warnings", []))
        + list(right.get("translation_warnings", []))
    ))
    if warnings:
        row["translation_warnings"] = warnings
    return row


def finalize_cues(
    cues: list[dict[str, Any]],
    *,
    min_duration: float = 0.85,
    remove_periods: bool = True,
    publish: bool = False,
) -> list[dict[str, Any]]:
    rows = [dict(x) for x in sorted(cues, key=lambda x: (x["start"], x["end"]))]
    if publish:
        for row in rows:
            row["zh"] = publish_text(row.get("zh", ""))
        rows = [x for x in rows if x.get("zh", "").strip()]
    if remove_periods:
        for row in rows:
            row["zh"] = strip_chinese_periods(row.get("zh", ""))

    # Merge duplicate, overlapping, or sub-second adjacent cues when that is safer
    # than flashing text. This pass also resolves dense starts before timing is clamped.
    merged: list[dict[str, Any]] = []
    index = 0
    while index < len(rows):
        row = dict(rows[index])
        while index + 1 < len(rows):
            nxt = rows[index + 1]
            duration = float(row["end"]) - float(row["start"])
            gap = float(nxt["start"]) - float(row["end"])
            dense_starts = float(nxt["start"]) - float(row["start"]) < min_duration + 0.04
            impossible_spacing = float(nxt["start"]) - float(row["start"]) <= 0.08
            duplicate = row.get("zh", "").strip() == nxt.get("zh", "").strip()
            combined_len = len(row.get("zh", "")) + len(nxt.get("zh", ""))
            should_merge = duplicate and gap <= 0.15
            should_merge = should_merge or impossible_spacing or (
                gap <= 0.20
                and combined_len <= 30
                and (duration < min_duration or dense_starts)
            )
            if not should_merge:
                break
            row = _combine_rows(row, nxt)
            index += 1
        merged.append(row)
        index += 1

    for index, row in enumerate(merged):
        start = float(row["start"])
        end = float(row["end"])
        next_start = float(merged[index + 1]["start"]) if index + 1 < len(merged) else end + 2
        if end - start < min_duration:
            end = min(next_start - 0.04, start + min_duration)
        if index + 1 < len(merged):
            end = min(end, next_start - 0.04)
        row["start"] = round(start, 3)
        row["end"] = round(max(start + 0.04, end), 3)
        if not publish and row.get("translation_warnings"):
            row["zh"] = "【需校对】" + row.get("zh", "").removeprefix("【需校对】")
        row["id"] = index + 1
    return merged


def find_gaps(cues: list[dict[str, Any]], duration: float, threshold: float = 30.0):
    gaps = []
    cursor = 0.0
    for cue in sorted(cues, key=lambda x: x["start"]):
        if cue["start"] - cursor >= threshold:
            gaps.append({"start": cursor, "end": cue["start"], "duration": cue["start"] - cursor})
        cursor = max(cursor, cue["end"])
    if duration - cursor >= threshold:
        gaps.append({"start": cursor, "end": duration, "duration": duration - cursor})
    return gaps


def _merge_intervals(rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
    intervals = sorted(
        (max(0.0, float(x["start"])), max(0.0, float(x["end"])))
        for x in rows
        if float(x["end"]) > float(x["start"])
    )
    merged: list[list[float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(x[0], x[1]) for x in merged]


def _overlap_seconds(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    total = 0.0
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        total += max(0.0, end - start)
        if left[left_index][1] <= right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return total


def quality_summary(
    cues: list[dict[str, Any]],
    duration: float,
    activity_segments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    durations = [x["end"] - x["start"] for x in cues]
    gaps = find_gaps(cues, duration)
    cue_intervals = _merge_intervals(cues)
    activity_intervals = _merge_intervals(activity_segments or [])
    activity_seconds = sum(end - start for start, end in activity_intervals)
    covered_activity = _overlap_seconds(cue_intervals, activity_intervals)
    for gap in gaps:
        gap_interval = [(float(gap["start"]), float(gap["end"]))]
        gap["activity_seconds"] = round(_overlap_seconds(gap_interval, activity_intervals), 3)
    return {
        "cue_count": len(cues),
        "display_seconds": round(sum(durations), 3),
        "under_085_seconds": sum(x < 0.85 - 1e-6 for x in durations),
        "exact_two_seconds": sum(abs(x - 2.0) < 0.01 for x in durations),
        "overlaps": sum(cues[i]["end"] > cues[i + 1]["start"] for i in range(len(cues) - 1)),
        "placeholders": sum(
            any(token in x.get("zh", "") for token in ("【听不清】", "【地点名不明】", "【场所名不明】", "【人名不明】"))
            for x in cues
        ),
        "review_markers": sum("【需校对】" in x.get("zh", "") for x in cues),
        "chinese_periods": sum("。" in x.get("zh", "") for x in cues),
        "activity_seconds": round(activity_seconds, 3),
        "covered_activity_seconds": round(covered_activity, 3),
        "activity_coverage_percent": round(
            covered_activity * 100.0 / activity_seconds, 1
        ) if activity_seconds else 0.0,
        "long_gaps": gaps,
        "longest_gap": round(max((x["duration"] for x in gaps), default=0.0), 3),
    }
