"""Làm sạch mô tả sản phẩm trước khi gửi LLM."""

import re

_MAX_INPUT_CHARS = 6000
_JUNK_MARKERS = (".aplus-v2", "AplusModule", ".apm-hovermodule", "(function(")


def _strip_html(text: str) -> str:
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.I | re.S)
    return re.sub(r"<[^>]+>", " ", text)


def _strip_css_blocks(text: str) -> str:
    return re.sub(r"\{[^{}]*\}", " ", text)


def _extract_readable_sentences(text: str, *, max_sentences: int = 25) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text)
    good: list[str] = []
    for part in parts:
        part = part.strip()
        if len(part) < 35:
            continue
        lower = part.lower()
        if any(m in lower for m in _JUNK_MARKERS):
            continue
        if "{" in part or "}" in part or "function(" in lower:
            continue
        alpha = sum(c.isalpha() or c.isspace() for c in part)
        if alpha / max(len(part), 1) < 0.65:
            continue
        good.append(part)
        if len(good) >= max_sentences:
            break
    return " ".join(good)


def prepare_description_for_llm(description: str, *, max_chars: int = _MAX_INPUT_CHARS) -> str:
    """Làm sạch HTML/CSS Amazon A+ và cắt độ dài hợp lý cho API."""
    if not description:
        return ""

    text = description.strip()
    text = _strip_html(text)
    text = _strip_css_blocks(text)
    text = re.sub(r"\s+", " ", text).strip()

    if any(m in text for m in _JUNK_MARKERS) or len(text) > max_chars * 2:
        extracted = _extract_readable_sentences(text)
        if extracted:
            text = extracted

    if len(text) > max_chars:
        cut = text[:max_chars]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        text = cut + "…"

    return text
