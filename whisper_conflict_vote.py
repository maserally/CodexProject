from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import whisper

from ensemble_common import choose_consensus, normalize_transcript, transcript_similarity


def best_whisper_view(qwen_text: str, cohere_text: str, candidates: dict[str, str]):
    scores = {
        name: transcript_similarity(text, qwen_text) + transcript_similarity(text, cohere_text)
        for name, text in candidates.items()
    }
    winner = max(scores, key=lambda name: (scores[name], len(normalize_transcript(candidates[name]))))
    return candidates[winner], winner, scores


def load_large_v3(model_path: str):
    path = Path(model_path)
    if path.is_dir():
        from faster_whisper import WhisperModel

        model = WhisperModel(
            str(path),
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8",
            local_files_only=True,
        )

        def transcribe(clip, language: str) -> str:
            segments, _ = model.transcribe(
                clip,
                language=language,
                task="transcribe",
                beam_size=5,
                temperature=0,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            return "".join(str(segment.text or "") for segment in segments).strip()

        return transcribe

    model = whisper.load_model(
        model_path, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    def transcribe(clip, language: str) -> str:
        result = model.transcribe(
            clip,
            language=language,
            task="transcribe",
            fp16=torch.cuda.is_available(),
            temperature=0,
            condition_on_previous_text=False,
            verbose=False,
        )
        return str(result.get("text") or "").strip()

    return transcribe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("media")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--language", choices=("ja", "ko"), required=True)
    parser.add_argument("--left")
    parser.add_argument("--right")
    parser.add_argument("--audio-report")
    args = parser.parse_args()

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    output_path = Path(args.output)
    audit_path = Path(args.audit)
    if output_path.exists():
        try:
            cached_rows = json.loads(output_path.read_text(encoding="utf-8"))
            cached_by_index = {
                int(row["window_index"]): row
                for row in cached_rows if isinstance(row, dict) and row.get("ensemble_resolved")
            }
            for index, row in enumerate(rows):
                cached = cached_by_index.get(int(row.get("window_index", -1)))
                if cached:
                    rows[index] = cached
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    conflict_indices = [
        index for index, row in enumerate(rows)
        if row.get("needs_third_vote") and not row.get("ensemble_resolved")
    ]
    transcribe_large_v3 = None
    audio_views = None
    if conflict_indices:
        audio_views = {"mid": whisper.load_audio(args.media)}
        layout = "mono"
        if args.audio_report and Path(args.audio_report).exists():
            report = json.loads(Path(args.audio_report).read_text(encoding="utf-8"))
            layout = str(report.get("layout", "mono"))
        if layout == "true_stereo" and args.left and args.right:
            audio_views["left"] = whisper.load_audio(args.left)
            audio_views["right"] = whisper.load_audio(args.right)
        transcribe_large_v3 = load_large_v3(args.model)

    audit = []
    if audit_path.exists():
        try:
            cached_audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if isinstance(cached_audit, list):
                audit = cached_audit
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    audited_indices = {int(row.get("window_index", -1)) for row in audit}
    for number, row in enumerate(rows, 1):
        if row.get("ensemble_resolved"):
            continue
        qwen_text = str(row.get("qwen_source") or "").strip()
        cohere_text = str(row.get("cohere_source") or "").strip()
        whisper_text = ""
        whisper_view = "mid"
        whisper_view_scores = {}
        if row.get("needs_third_vote"):
            view_candidates = {}
            for view_name, view_audio in audio_views.items():
                clip = view_audio[int(row["start"] * 16000) : int(row["end"] * 16000)]
                view_candidates[view_name] = transcribe_large_v3(clip, args.language)
            whisper_text, whisper_view, whisper_view_scores = best_whisper_view(
                qwen_text, cohere_text, view_candidates
            )
            print(f"large-v3 conflict {number}/{len(rows)}", flush=True)

        if whisper_text:
            final_text, winner, similarities = choose_consensus(
                qwen_text, cohere_text, whisper_text
            )
        elif normalize_transcript(qwen_text):
            final_text, winner = qwen_text, "qwen"
            similarities = {
                "qwen_cohere": float(row.get("qwen_cohere_similarity", 0)),
                "qwen_whisper": 0.0,
                "cohere_whisper": 0.0,
            }
        else:
            final_text, winner = cohere_text, "cohere"
            similarities = {
                "qwen_cohere": float(row.get("qwen_cohere_similarity", 0)),
                "qwen_whisper": 0.0,
                "cohere_whisper": 0.0,
            }
        row["whisper_source"] = whisper_text
        row["whisper_view"] = whisper_view
        row["whisper_view_scores"] = whisper_view_scores
        row["final_source"] = final_text
        row["ensemble_winner"] = winner
        row["similarities"] = similarities
        best_pair = max(similarities.values(), default=0.0)
        row["ensemble_confidence"] = (
            "two_model_consensus" if best_pair >= 0.55 else "single_model_low_confidence"
        )
        row["ensemble_resolved"] = True
        audit_row = {
                "window_index": row.get("window_index", number - 1),
                "start": row["start"],
                "end": row["end"],
                "review_reasons": row.get("review_reasons", []),
                "qwen": qwen_text,
                "cohere": cohere_text,
                "whisper": whisper_text,
                "whisper_view": whisper_view,
                "whisper_view_scores": whisper_view_scores,
                "winner": winner,
                "confidence": row["ensemble_confidence"],
                "similarities": similarities,
            }
        if int(audit_row["window_index"]) not in audited_indices:
            audit.append(audit_row)
            audited_indices.add(int(audit_row["window_index"]))
        output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"large-v3 conflicts={len(conflict_indices)}", flush=True)


if __name__ == "__main__":
    main()
