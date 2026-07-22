from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoProcessor

try:
    from transformers import CohereAsrForConditionalGeneration
except ImportError:  # transformers before native Cohere ASR support
    from transformers import AutoModelForSpeechSeq2Seq as CohereAsrForConditionalGeneration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("media")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", choices=("ja", "ko"), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--review-all", action="store_true")
    args = parser.parse_args()

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    output_path = Path(args.output)
    if output_path.exists():
        try:
            cached_rows = json.loads(output_path.read_text(encoding="utf-8"))
            cached_by_index = {
                int(row["window_index"]): row
                for row in cached_rows if isinstance(row, dict) and "window_index" in row
            }
            for row in rows:
                cached = cached_by_index.get(int(row.get("window_index", -1)))
                if cached and "cohere_source" in cached:
                    for key in (
                        "cohere_source", "qwen_cohere_similarity", "needs_third_vote"
                    ):
                        row[key] = cached.get(key)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    audio, sample_rate = sf.read(args.media, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    review_indices = [
        index for index, row in enumerate(rows)
        if "cohere_source" not in row and (args.review_all or row.get("needs_review"))
    ]
    if not review_indices:
        output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Cohere review skipped: no low-confidence windows", flush=True)
        return

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    batch_size = max(1, args.batch_size)
    reviewed = 0
    offset = 0
    while offset < len(review_indices):
        indices = review_indices[offset : offset + batch_size]
        clips = [
            audio[int(rows[index]["start"] * sample_rate) : int(rows[index]["end"] * sample_rate)]
            for index in indices
        ]
        inputs = processor(
            audio=clips,
            sampling_rate=sample_rate,
            return_tensors="pt",
            language=args.language,
            punctuation=False,
            padding=True,
        )
        audio_chunk_index = inputs.get("audio_chunk_index")
        inputs.to(model.device, dtype=model.dtype)
        try:
            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=384, do_sample=False)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower() or batch_size == 1:
                raise
            del inputs
            batch_size = max(1, batch_size // 2)
            torch.cuda.empty_cache()
            print(f"Cohere 显存不足，批量自动降为 {batch_size}", flush=True)
            continue
        decoded = processor.decode(
            output_ids,
            skip_special_tokens=True,
            audio_chunk_index=audio_chunk_index,
            language=args.language,
        )
        if isinstance(decoded, str):
            decoded = [decoded]
        for index, text in zip(indices, decoded):
            row = rows[index]
            row["cohere_source"] = str(text or "").strip()
        reviewed += len(indices)
        offset += len(indices)
        print(f"Cohere review {reviewed}/{len(review_indices)}", flush=True)
        output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Cohere reviewed={len(review_indices)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
