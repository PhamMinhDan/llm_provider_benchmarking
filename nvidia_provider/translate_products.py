"""
Dịch merged_products.csv sang tiếng Việt qua NVIDIA API.

Ghi đè các cột văn bản: title, description, category, tags (tên cột giữ nguyên).
Giữ nguyên: product_id, source, brand, price, rating, reviews_count, image_url.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nvidia_provider.llm import (  # noqa: E402
    MODELS,
    ModelConfig,
    complete_with_fallback,
    create_client,
    probe_models,
)
from nvidia_provider.prompts import (  # noqa: E402
    TRANSLATE_FIELD_NAMES,
    PRODUCT_ROW_SYSTEM,
    PRODUCT_ROW_USER,
)
from nvidia_provider.text_utils import prepare_description_for_llm  # noqa: E402

PRESERVE_COLUMNS = frozenset(
    {
        "product_id",
        "source",
        "brand",
        "price",
        "rating",
        "reviews_count",
        "image_url",
    }
)


def _checkpoint_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".checkpoint")


def build_messages(row: dict) -> list[dict]:
    brand = (row.get("brand") or "").strip() or "(không rõ)"
    title = (row.get("title") or "").strip()
    category = (row.get("category") or "").strip()
    tags = (row.get("tags") or "").strip()
    raw_desc = row.get("description") or ""
    description = prepare_description_for_llm(raw_desc)

    if not description.strip() and title:
        description = (
            "(Mô tả gốc không đọc được hoặc rỗng. "
            "Dựa trên title, brand, category — viết description tiếng Việt ngắn, "
            "không bịa thông số kỹ thuật.)"
        )

    user_content = PRODUCT_ROW_USER.format(
        brand=brand,
        title=title,
        description=description,
        category=category or "(trống)",
        tags=tags or "(trống)",
    )
    system_content = PRODUCT_ROW_SYSTEM
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _normalize_text(s: str) -> str:
    # Normalize whitespace and casing for "unchanged" detection.
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _needs_retry_fields(src_row: dict, translated: dict[str, str]) -> list[str]:
    """Return fields that appear empty or unchanged (likely translation failure)."""
    retry_fields: list[str] = []
    for key in ("title", "category", "tags"):
        src = _normalize_text(src_row.get(key) or "")
        tr = _normalize_text(translated.get(key) or "")
        if not tr or tr == src:
            retry_fields.append(key)
    return retry_fields


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("JSON phải là object.")
    return data


def parse_translation(raw: str) -> dict[str, str]:
    data = _extract_json(raw)
    out: dict[str, str] = {}
    for key in TRANSLATE_FIELD_NAMES:
        val = data.get(key)
        if val is None:
            raise ValueError(f"Thiếu khóa JSON: {key}")
        out[key] = str(val).strip()
    return out


def apply_translation(row: dict, translated: dict[str, str]) -> None:
    for key in TRANSLATE_FIELD_NAMES:
        if key in translated and translated[key]:
            row[key] = translated[key]


def load_done_ids(checkpoint_path: Path) -> set[str]:
    if not checkpoint_path.is_file():
        return set()
    return {
        line.strip()
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def mark_done(checkpoint_path: Path, product_id: str) -> None:
    with checkpoint_path.open("a", encoding="utf-8") as f:
        f.write(product_id + "\n")


def row_needs_translation(row: dict) -> bool:
    return any((row.get(k) or "").strip() for k in TRANSLATE_FIELD_NAMES)


def resolve_candidates(
    client,
    *,
    model_name: str | None,
    skip_probe: bool,
    verbose: bool,
) -> list[ModelConfig]:
    if model_name:
        cfg = next((m for m in MODELS if m.name == model_name), None)
        if cfg is None:
            raise ValueError(f"Model không có trong MODELS: {model_name}")
        if verbose:
            print(f"Dùng model cố định: {model_name}\n")
        return [cfg]

    if skip_probe:
        if verbose:
            print("Bỏ probe — dùng toàn bộ MODELS theo thứ tự ưu tiên.\n")
        return list(MODELS)

    candidates, responded = probe_models(client, verbose=verbose)
    if not candidates:
        raise RuntimeError("Không có model nào phản hồi probe.")
    if verbose:
        winner = candidates[0]
        print(
            f"\n→ Model ưu tiên: {winner.name} "
            f"(TTFT: {responded[winner.name]:.3f}s, fallback: {len(candidates) - 1})\n"
        )
    return candidates


def translate_csv(
    input_path: Path,
    output_path: Path,
    *,
    limit: int | None = None,
    resume: bool = True,
    delay_s: float = 0.0,
    model_name: str | None = None,
    skip_probe: bool = False,
    verbose: bool = True,
) -> None:
    client = create_client()
    candidates = resolve_candidates(
        client,
        model_name=model_name,
        skip_probe=skip_probe,
        verbose=verbose,
    )

    checkpoint = _checkpoint_path(output_path)
    done_ids = load_done_ids(checkpoint) if resume else set()
    if verbose and done_ids:
        print(f"Resume: bỏ qua {len(done_ids)} sản phẩm (checkpoint).\n")

    with input_path.open(encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            raise ValueError("CSV không có header.")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    missing = [k for k in TRANSLATE_FIELD_NAMES if k not in fieldnames]
    if missing:
        raise ValueError(f"CSV thiếu cột: {missing}")

    write_header = not output_path.is_file() or not resume or not done_ids
    out_mode = "w" if write_header else "a"

    processed = 0
    skipped = 0
    errors = 0

    with output_path.open(out_mode, encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for row in rows:
            if limit is not None and processed >= limit:
                break

            pid = (row.get("product_id") or "").strip()
            if resume and pid and pid in done_ids:
                skipped += 1
                continue

            out_row = dict(row)
            for col in PRESERVE_COLUMNS:
                if col in row:
                    out_row[col] = row[col]

            if not row_needs_translation(row):
                writer.writerow(out_row)
                fout.flush()
                if pid:
                    mark_done(checkpoint, pid)
                processed += 1
                continue

            try:
                messages = build_messages(row)
                raw, used_model = complete_with_fallback(
                    client, candidates, messages, max_tokens=2048
                )
                translated = parse_translation(raw)

                retry_fields = _needs_retry_fields(row, translated)
                if retry_fields:
                    # Some providers/models sometimes "fail open" by copying EN text
                    # for title/category/tags while still translating description.
                    strict_note = (
                        "BẠN CHƯA DỊCH ĐÚNG. Hãy dịch SANG TIẾNG VIỆT cho các trường: "
                        f"{', '.join(retry_fields)}. Tuyệt đối KHÔNG giữ nguyên tiếng Anh đầu vào "
                        "(trừ thương hiệu/tên riêng/size/SKU). Không để chuỗi rỗng. "
                        "Trả lại JSON đúng 4 khóa: title, description, category, tags."
                    )
                    # Rebuild messages with a stricter system note.
                    messages = build_messages(row)
                    messages[0]["content"] = messages[0]["content"] + "\n\n" + strict_note
                    raw, used_model = complete_with_fallback(
                        client, candidates, messages, max_tokens=2048
                    )
                    translated = parse_translation(raw)
                apply_translation(out_row, translated)
                if verbose:
                    title_preview = (out_row.get("title") or "")[:55]
                    retry_note = f" retry_fields={retry_fields}" if retry_fields else ""
                    print(f"[{processed + 1}] {pid} ({used_model}) {title_preview}…{retry_note}")
                if pid:
                    mark_done(checkpoint, pid)
            except Exception as e:
                errors += 1
                if verbose:
                    print(f"  ✗ {pid}: {e}", file=sys.stderr)

            writer.writerow(out_row)
            fout.flush()
            processed += 1

            if delay_s > 0:
                time.sleep(delay_s)

    if verbose:
        print(
            f"\nXong. Dịch mới: {processed}, bỏ qua (resume): {skipped}, lỗi: {errors}."
        )
        print(f"File: {output_path}")
        print(
            f"Cột dịch: {', '.join(TRANSLATE_FIELD_NAMES)}. "
            f"Giữ nguyên: {', '.join(sorted(PRESERVE_COLUMNS))}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dịch title, description, category, tags sang tiếng Việt (cùng tên cột)."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=_ROOT / "merged_products.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_ROOT / "merged_products_vi.csv",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--model", default=None)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Không tìm thấy file input: {args.input}")

    if args.no_resume:
        if args.output.is_file():
            args.output.unlink()
        cp = _checkpoint_path(args.output)
        if cp.is_file():
            cp.unlink()

    translate_csv(
        args.input,
        args.output,
        limit=args.limit,
        resume=not args.no_resume,
        delay_s=args.delay,
        model_name=args.model,
        skip_probe=args.skip_probe,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
