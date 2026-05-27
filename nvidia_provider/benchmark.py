"""
1. Probe song song tất cả model trong LIMIT_RESPONSE_DURATION giây.
2. Chọn model ưu tiên cao nhất (theo thứ tự MODELS) có phản hồi probe.
3. Gửi context lên model đó. Nếu sau CONTEXT_FIRST_TOKEN_TIMEOUT giây
   không nhận được token đầu tiên → thử model tiếp theo trong MODELS.
"""

import sys
import threading
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nvidia_provider.context import context
from nvidia_provider.llm import (
    LIMIT_RESPONSE_DURATION,
    ModelConfig,
    create_client,
    probe_models,
)

CONTEXT_LIVENESS_TIMEOUT = 1.5
CONTEXT_CONTENT_TIMEOUT = 15.0


def run_with_context(
    cfg: ModelConfig,
    client: OpenAI,
    liveness_timeout: float,
    content_timeout: float,
) -> bool:
    """
    Stream context lên model với 2-phase timeout:
      Phase 1 — Liveness: chờ bất kỳ token nào (reasoning hoặc content) trong liveness_timeout.
      Phase 2 — Content:  chờ content token thực sự trong content_timeout kể từ khi model sống.
    Trả về True nếu nhận được ít nhất 1 content token.
    """
    liveness_event = threading.Event()
    content_event = threading.Event()
    abort_event = threading.Event()
    stream_holder: list[openai.Stream] = []

    def _stream():
        stream: openai.Stream | None = None
        try:
            stream = client.chat.completions.create(
                model=cfg.name,
                messages=context,
                max_tokens=4096,
                temperature=0.6,
                top_p=0.7,
                stream=True,
                extra_body=cfg.extra_body or None,
                timeout=120,
            )
            stream_holder.append(stream)

            for chunk in stream:
                if abort_event.is_set():
                    return
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None) or ""
                content = getattr(delta, "content", None) or ""

                if reasoning or content:
                    liveness_event.set()

                if content:
                    content_event.set()
                    print(content, end="", flush=True)

            if content_event.is_set():
                print()
        except Exception as e:
            if not abort_event.is_set():
                print(f"\n  ⚠ Stream error ({cfg.name}): {e}", file=sys.stderr)
        finally:
            if stream is not None:
                stream.close()

    def _abort():
        abort_event.set()
        if stream_holder:
            stream_holder[0].close()

    t = threading.Thread(target=_stream, daemon=True)
    t.start()

    if not liveness_event.wait(timeout=liveness_timeout):
        _abort()
        return False

    if not content_event.wait(timeout=content_timeout):
        _abort()
        return False

    t.join(timeout=120)
    if t.is_alive():
        _abort()
    return True


def main():
    client = create_client()
    candidates, responded = probe_models(client)

    if not candidates:
        print("\nKhông có model nào phản hồi trong thời gian probe.")
        sys.exit(1)

    selected = candidates[0]
    print(
        f"\n→ Probe winner: {selected.name} (TTFT: {responded[selected.name]:.3f}s)\n"
    )

    for cfg in candidates:
        print("=" * 60)
        print(
            f"Gửi context → {cfg.name}  "
            f"(liveness: {CONTEXT_LIVENESS_TIMEOUT}s / content: {CONTEXT_CONTENT_TIMEOUT}s)"
        )
        print("=" * 60 + "\n")

        if run_with_context(
            cfg, client, CONTEXT_LIVENESS_TIMEOUT, CONTEXT_CONTENT_TIMEOUT
        ):
            sys.exit(0)

        print(
            f"\n  ✗ {cfg.name}: không có content sau"
            f" {CONTEXT_LIVENESS_TIMEOUT + CONTEXT_CONTENT_TIMEOUT:.0f}s"
            " → thử model tiếp theo...\n"
        )

    print("Tất cả model đều không phản hồi context.")
    sys.exit(1)


if __name__ == "__main__":
    main()
