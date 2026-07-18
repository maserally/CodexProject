from __future__ import annotations

from typing import Any


LANGUAGES: dict[str, dict[str, str]] = {
    "ja": {
        "name": "日语",
        "source_subtitle": "日文字幕",
        "bilingual_subtitle": "中日双语",
        "whisper_code": "ja",
    },
    "ko": {
        "name": "韩语",
        "source_subtitle": "韩文字幕",
        "bilingual_subtitle": "中韩双语",
        "whisper_code": "ko",
    },
}


def language_info(code: str) -> dict[str, str]:
    try:
        return LANGUAGES[code]
    except KeyError as exc:
        raise ValueError(f"不支持的源语言：{code}") from exc


def source_text(row: dict[str, Any]) -> str:
    """Read generic source text while accepting version 1 Japanese rows."""
    return str(row.get("source", row.get("ja", "")))

