from __future__ import annotations

import re
from typing import Any, Callable

from .providers import OllamaProvider, OpenAICompatibleProvider
from .languages import language_info, source_text
from .schemas import ProviderSettings


NEGATIVE_ZH = re.compile(r"不|没|別|别|未|无|不是|并非|不要|不同|错|住手|停下")
NEGATIVE_SOURCE = {
    "ja": re.compile(r"ない|ません|じゃない|なく|てない|聞いてない|違う"),
    "ko": re.compile(r"않|못|없|아니|아닙|안|말[아어]|싫|틀렸|하지\s*마"),
}
SOURCE_SCRIPT = {
    "ja": re.compile(r"[ぁ-ゟ゠-ヿ]"),
    "ko": re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]"),
}


SYSTEM_PROMPTS = {
    "ja": """你是专业日中影视字幕译者。只翻译 target.source 中的日语，context 仅供理解，绝不能把相邻句译进 target。
输出自然、简洁、忠实的简体中文字幕；保持否定、拒绝、疑问、人称和语气，不得补写原文没有的信息。
不得把 やめて 译成“别停”或“继续”；默认译为“住手、停下、不要这样”。
聞いてない 必须保留否定，待って 必须包含“等一下”，勘弁 必须保留求饶含义。
成人语境也必须忠实翻译，不得把拒绝反译成同意。
原文不完整时保留省略感；无法确认的标签必须保留不确定性。
中文不要使用句号。只输出 JSON：{"id":整数,"zh":"译文"}。""",
    "ko": """你是专业韩中影视字幕译者。只翻译 target.source 中的韩语，context 仅供理解，绝不能把相邻句译进 target。
输出自然、简洁、忠实的简体中文字幕；保持否定、拒绝、疑问、人称、敬语层级和语气，不得补写原文没有的信息。
不得把 하지 마 或 그만해 译成“继续”或同意；应保留“不要、住手、停下”的含义。
기다려 必须保留“等一下”，안 들었어／못 들었어 必须保留“没听到”，봐줘／살려줘 必须保留求饶或求救含义。
成人语境也必须忠实翻译，不得把拒绝反译成同意。
原文不完整时保留省略感；无法确认的标签必须保留不确定性。
中文不要使用句号。只输出 JSON：{"id":整数,"zh":"译文"}。""",
}


def provider_from_settings(settings: ProviderSettings):
    if settings.kind == "local_ollama":
        return OllamaProvider(settings.base_url or "http://127.0.0.1:11434")
    if settings.kind == "openai_compatible":
        return OpenAICompatibleProvider(settings.base_url, settings.api_key)
    raise ValueError(f"Unsupported translation provider: {settings.kind}")


def audit_translation(source: str, zh: str, source_language: str = "ja") -> list[str]:
    problems = []
    source_name = language_info(source_language)["name"]
    if not zh.strip():
        problems.append("译文为空")
    if "。" in zh or zh.rstrip().endswith("."):
        problems.append("中文不得包含句号")
    negative_re = NEGATIVE_SOURCE[source_language]
    if negative_re.search(source) and not NEGATIVE_ZH.search(zh):
        problems.append(f"{source_name}的否定或纠正含义没有保留")
    if "やめて" in source and ("别停" in zh or "继续" in zh):
        problems.append("やめて 被反译")
    if "やめて" in source and not re.search(r"住手|停下|停一|不要|别这样", zh):
        problems.append("やめて 缺少停止或拒绝含义")
    if "待って" in source and "等" not in zh:
        problems.append("待って 必须包含等一下")
    if "聞いてない" in source and not re.search(r"没听|不知道|没听说|不清楚", zh):
        problems.append("聞いてない 必须保留没听或不知道")
    if "勘弁" in source and not re.search(r"饶|放过|受不了|别再", zh):
        problems.append("勘弁 的求饶语气未体现")
    if re.search(r"하지\s*마|그만해", source) and not re.search(r"不要|别|住手|停下|够了", zh):
        problems.append("韩语制止或拒绝含义没有保留")
    if "기다려" in source and "等" not in zh:
        problems.append("기다려 必须包含等一下")
    if re.search(r"안\s*들었|못\s*들었", source) and not re.search(r"没听|没听到|不知道", zh):
        problems.append("韩语未听到的否定含义没有保留")
    if re.search(r"봐\s*줘|살려\s*줘", source) and not re.search(r"放过|饶|救|帮", zh):
        problems.append("韩语求饶或求救含义没有保留")
    if SOURCE_SCRIPT[source_language].search(zh):
        problems.append(f"中文残留{source_name}字符")
    return problems


def safe_high_risk(source: str, zh: str, source_language: str = "ja") -> str:
    if "やめて" in source:
        if "本当に" in source:
            return "真的，住手"
        if "ちょっと" in source:
            return "请先停一下"
        return "住手"
    if "聞いてない" in source:
        return "没听说过吗？" if "?" in source or "？" in source else "我没听说"
    if "勘弁してください" in source:
        return "请放过我吧"
    if "勘弁" in source:
        return "饶了我吧"
    if source_language == "ko":
        if re.search(r"하지\s*마|그만해", source):
            return "住手"
        if "기다려" in source:
            return "等一下"
        if re.search(r"안\s*들었|못\s*들었", source):
            return "我没听到"
        if re.search(r"살려\s*줘", source):
            return "救救我"
        if re.search(r"봐\s*줘", source):
            return "放过我吧"
    return zh


def translate_cues(
    rows: list[dict[str, Any]],
    settings: ProviderSettings,
    progress: Callable[[int, int, str], None] | None = None,
    *,
    source_language: str = "ja",
    target_language: str = "zh-CN",
) -> list[dict[str, Any]]:
    if target_language != "zh-CN":
        raise ValueError(f"不支持的目标语言：{target_language}")
    provider = provider_from_settings(settings)
    system_prompt = SYSTEM_PROMPTS[source_language]
    output = []
    for index, source in enumerate(rows):
        row = dict(source)
        context = [
            {"id": i + 1, "source": source_text(rows[i])}
            for i in range(max(0, index - 2), min(len(rows), index + 3))
        ]
        original = source_text(row)
        request = {"context": context, "target": {"id": index + 1, "source": original}}
        parsed = provider.chat_json(settings.model, system_prompt, request)
        zh = str(parsed.get("zh", "")).strip().replace("。", "")
        problems = audit_translation(original, zh, source_language)
        if problems:
            parsed = provider.chat_json(
                settings.model,
                system_prompt + "\n上次译文未通过审计，请修正：" + "；".join(problems),
                request,
            )
            zh = str(parsed.get("zh", "")).strip().replace("。", "")
        zh = safe_high_risk(original, zh, source_language)
        remaining = audit_translation(original, zh, source_language)
        if remaining:
            row["translation_warnings"] = remaining
        row["id"] = index + 1
        row["source"] = original
        row.pop("ja", None)
        row["zh"] = zh
        output.append(row)
        if progress:
            progress(index + 1, len(rows), zh)
    return output
