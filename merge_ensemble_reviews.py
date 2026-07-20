from __future__ import annotations

import argparse
import json
from pathlib import Path

from ensemble_common import needs_third_vote, transcript_similarity


def by_window(rows: list[dict]) -> dict[int, dict]:
    return {int(row["window_index"]): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", required=True)
    parser.add_argument("--cohere", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    qwen_rows = json.loads(Path(args.qwen).read_text(encoding="utf-8"))
    cohere_rows = by_window(json.loads(Path(args.cohere).read_text(encoding="utf-8")))
    merged: list[dict] = []
    for qwen in qwen_rows:
        row = dict(qwen)
        cohere = cohere_rows.get(int(row["window_index"]), {})
        row["cohere_source"] = str(cohere.get("cohere_source") or "").strip()
        similarity = transcript_similarity(row.get("qwen_source", ""), row["cohere_source"])
        row["qwen_cohere_similarity"] = round(similarity, 6)
        row["needs_third_vote"] = needs_third_vote(
            row.get("qwen_source", ""),
            row["cohere_source"],
            acoustic_risk=bool(row.get("acoustic_risk")),
            speech_expected=float(row.get("speech_score", 0.0)) >= 0.15,
        )
        merged.append(row)

    Path(args.output).write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"ensemble merged={len(merged)} conflicts={sum(bool(x['needs_third_vote']) for x in merged)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
