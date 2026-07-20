from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from ensemble_common import acoustic_risk_reasons, full_coverage_windows


def acoustic_features(clip: np.ndarray, sample_rate: int) -> dict[str, float]:
    if clip.ndim > 1:
        clip = clip.mean(axis=1)
    clip = np.asarray(clip, dtype=np.float32)
    if not len(clip):
        return {
            "rms_db": -120.0,
            "crest_db": 0.0,
            "high_frequency_ratio": 0.0,
            "spectral_flatness": 0.0,
        }
    rms = float(np.sqrt(np.mean(np.square(clip), dtype=np.float64) + 1e-12))
    peak = float(np.max(np.abs(clip)) + 1e-12)
    spectrum = np.abs(np.fft.rfft(clip * np.hanning(len(clip)))) + 1e-12
    frequencies = np.fft.rfftfreq(len(clip), 1.0 / sample_rate)
    power = np.square(spectrum)
    high_ratio = float(power[frequencies >= 3500].sum() / max(float(power.sum()), 1e-12))
    flatness = float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))
    return {
        "rms_db": round(20.0 * np.log10(rms + 1e-12), 4),
        "crest_db": round(20.0 * np.log10(peak / max(rms, 1e-12)), 4),
        "high_frequency_ratio": round(high_ratio, 6),
        "spectral_flatness": round(flatness, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("media")
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--core-seconds", type=float, default=16.0)
    parser.add_argument("--context-seconds", type=float, default=2.0)
    args = parser.parse_args()

    events = json.loads(Path(args.events).read_text(encoding="utf-8"))
    audio, sample_rate = sf.read(args.media, dtype="float32", always_2d=False)
    duration = len(audio) / sample_rate
    windows = full_coverage_windows(
        events,
        duration,
        core_seconds=args.core_seconds,
        context_seconds=args.context_seconds,
    )
    for row in windows:
        clip = audio[int(row["start"] * sample_rate) : int(row["end"] * sample_rate)]
        row.update(acoustic_features(clip, sample_rate))
        row["acoustic_risk_reasons"] = acoustic_risk_reasons(row)
        row["acoustic_risk"] = bool(row["acoustic_risk_reasons"])

    Path(args.output).write_text(
        json.dumps(windows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"ensemble windows={len(windows)} high_risk={sum(bool(x['acoustic_risk']) for x in windows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
