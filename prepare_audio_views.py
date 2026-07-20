from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def channel_report(path: Path) -> dict[str, object]:
    with sf.SoundFile(path) as stream:
        channels = stream.channels
        sample_rate = stream.samplerate
        frames = len(stream)
        if channels < 2:
            return {
                "channels": channels,
                "sample_rate": sample_rate,
                "duration": round(frames / sample_rate, 3),
                "layout": "mono",
            }
        sum_left_sq = sum_right_sq = sum_cross = sum_mid_sq = sum_side_sq = 0.0
        for block in stream.blocks(blocksize=sample_rate * 30, dtype="float32", always_2d=True):
            left = block[:, 0].astype(np.float64)
            right = block[:, 1].astype(np.float64)
            mid = (left + right) * 0.5
            side = (left - right) * 0.5
            sum_left_sq += float(np.dot(left, left))
            sum_right_sq += float(np.dot(right, right))
            sum_cross += float(np.dot(left, right))
            sum_mid_sq += float(np.dot(mid, mid))
            sum_side_sq += float(np.dot(side, side))
        correlation = sum_cross / math.sqrt(max(sum_left_sq * sum_right_sq, 1e-24))
        side_to_mid = 10.0 * math.log10(max(sum_side_sq, 1e-24) / max(sum_mid_sq, 1e-24))
        dual_mono = correlation >= 0.999 and side_to_mid <= -50.0
        return {
            "channels": channels,
            "sample_rate": sample_rate,
            "duration": round(frames / sample_rate, 3),
            "correlation": round(correlation, 8),
            "side_to_mid_db": round(side_to_mid, 4),
            "layout": "dual_mono" if dual_mono else "true_stereo",
        }


def run_ffmpeg(source: Path, target: Path, audio_filter: str) -> None:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-vn", "-af", audio_filter, "-ar", "16000", "-ac", "1", "-c:a", "flac",
        "-compression_level", "5", str(target),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("media")
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()

    source = Path(args.media)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    report_path = workdir / "audio_view_report.json"
    expected_outputs = [
        workdir / "raw_view.flac",
        workdir / "conservative_enhanced_view.flac",
        workdir / "left_view.flac",
        workdir / "right_view.flac",
    ]
    if report_path.exists() and all(path.exists() and path.stat().st_size for path in expected_outputs):
        try:
            cached = json.loads(report_path.read_text(encoding="utf-8"))
            if int(cached.get("source_size", -1)) == source.stat().st_size:
                print("audio views resumed from verified local checkpoint", flush=True)
                return
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    report = channel_report(source)
    raw = workdir / "raw_view.flac"
    enhanced = workdir / "conservative_enhanced_view.flac"
    left = workdir / "left_view.flac"
    right = workdir / "right_view.flac"
    # The raw view only performs a standard downmix. The enhanced view removes
    # sub-speech rumble and gently reduces level disparity; it is independent
    # evidence and never replaces the raw view.
    run_ffmpeg(source, raw, "pan=mono|c0=0.5*c0+0.5*c1" if report["channels"] >= 2 else "anull")
    run_ffmpeg(
        source,
        enhanced,
        (
            ("pan=mono|c0=0.5*c0+0.5*c1," if report["channels"] >= 2 else "")
            + "highpass=f=70,lowpass=f=7600,"
            + "acompressor=threshold=0.06:ratio=1.6:attack=20:release=180:makeup=1.35,"
            + "alimiter=limit=0.97"
        ),
    )
    if report["channels"] >= 2:
        run_ffmpeg(source, left, "pan=mono|c0=c0")
        run_ffmpeg(source, right, "pan=mono|c0=c1")
    else:
        run_ffmpeg(source, left, "anull")
        run_ffmpeg(source, right, "anull")
    report.update(
        {
            "source_size": source.stat().st_size,
            "raw_view": str(raw),
            "enhanced_view": str(enhanced),
            "left_view": str(left),
            "right_view": str(right),
        }
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
