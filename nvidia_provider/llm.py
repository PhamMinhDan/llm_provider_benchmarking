"""Client NVIDIA Integrate API: probe model, chọn ưu tiên, gọi chat completion."""

import os
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field

import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

INVOKE_URL_BASE = "https://integrate.api.nvidia.com/v1"
LIMIT_RESPONSE_DURATION = 1.5
TEST_MESSAGES = [{"role": "user", "content": "hello"}]

_LOW_THINKING_TOKENS = 1024
_QWEN_EXTRA = {"chat_template_kwargs": {"enable_thinking": "low"}}
_NEMOTRON_EXTRA = {
    "chat_template_kwargs": {"enable_thinking": False},
    "reasoning_budget": _LOW_THINKING_TOKENS,
}
_NEMOTRON_NANO_9B_EXTRA = {
    "min_thinking_tokens": 0,
    "max_thinking_tokens": _LOW_THINKING_TOKENS,
}
_GLM_EXTRA = {
    "chat_template_kwargs": {"enable_thinking": False, "clear_thinking": False}
}
_SEED_OSS_EXTRA = {"thinking_budget": _LOW_THINKING_TOKENS}
_GPT_OSS_EXTRA = {"reasoning_effort": "low"}


@dataclass
class ModelConfig:
    name: str
    extra_body: dict = field(default_factory=dict)


MODELS: list[ModelConfig] = [
    ModelConfig("meta/llama-3.1-405b-instruct"),
    ModelConfig("qwen/qwen3-coder-480b-a35b-instruct", _QWEN_EXTRA),
    ModelConfig("qwen/qwen3.5-397b-a17b", _QWEN_EXTRA),
    ModelConfig("qwen/qwen3.5-122b-a10b", _QWEN_EXTRA),
    ModelConfig("qwen/qwen3-next-80b-a3b-instruct", _QWEN_EXTRA),
    ModelConfig("qwen/qwen3-next-80b-a3b-thinking", _QWEN_EXTRA),
    ModelConfig("mistralai/mistral-large-3-675b-instruct-2512"),
    ModelConfig("mistralai/ministral-14b-instruct-2512"),
    ModelConfig("bytedance/seed-oss-36b-instruct", _SEED_OSS_EXTRA),
    ModelConfig("qwen/qwq-32b", _QWEN_EXTRA),
    ModelConfig("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", _NEMOTRON_EXTRA),
    ModelConfig("nvidia/nemotron-3-nano-30b-a3b", _NEMOTRON_EXTRA),
    ModelConfig("nvidia/nvidia-nemotron-nano-9b-v2", _NEMOTRON_NANO_9B_EXTRA),
    ModelConfig("openai/gpt-oss-120b", _GPT_OSS_EXTRA),
    ModelConfig("openai/gpt-oss-20b", _GPT_OSS_EXTRA),
    ModelConfig("z-ai/glm-5.1", _GLM_EXTRA),
    ModelConfig("z-ai/glm5", _GLM_EXTRA),
    ModelConfig("z-ai/glm4.7", _GLM_EXTRA),
    ModelConfig("meta/llama-3.3-70b-instruct"),
]


def create_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY không được tìm thấy trong biến môi trường.")
    return OpenAI(base_url=INVOKE_URL_BASE, api_key=api_key)


def probe(cfg: ModelConfig, client: OpenAI) -> tuple[str, float | None, str]:
    """Trả về (model_name, ttft_seconds, status)."""
    start = time.perf_counter()
    try:
        stream = client.chat.completions.create(
            model=cfg.name,
            messages=TEST_MESSAGES,
            max_tokens=16,
            temperature=0.6,
            top_p=0.7,
            stream=True,
            extra_body=cfg.extra_body or None,
            timeout=LIMIT_RESPONSE_DURATION + 2,
        )
        with stream:
            for chunk in stream:
                if chunk.choices:
                    return cfg.name, time.perf_counter() - start, "ok"
        return cfg.name, None, "no content"
    except openai.APITimeoutError:
        return cfg.name, None, "timeout"
    except openai.APIConnectionError:
        return cfg.name, None, "connection error"
    except openai.APIStatusError as e:
        return cfg.name, None, f"HTTP {e.status_code}"
    except Exception as e:
        return cfg.name, None, f"error: {e}"


def probe_models(
    client: OpenAI,
    *,
    verbose: bool = True,
) -> tuple[list[ModelConfig], dict[str, float]]:
    """Probe song song; trả về (danh sách candidate theo ưu tiên, map ttft)."""
    if verbose:
        print(f"Probe {len(MODELS)} model trong {LIMIT_RESPONSE_DURATION}s...\n")

    executor = ThreadPoolExecutor(max_workers=len(MODELS))
    future_to_cfg = {executor.submit(probe, cfg, client): cfg for cfg in MODELS}
    done, not_done = wait(future_to_cfg.keys(), timeout=LIMIT_RESPONSE_DURATION)
    executor.shutdown(wait=False, cancel_futures=True)

    responded: dict[str, float] = {}
    for future in done:
        name, ttft, status = future.result()
        if ttft is not None:
            responded[name] = ttft
            if verbose:
                print(f"  ✓ {name:<52} {ttft:.3f}s")
        elif verbose:
            print(f"  ✗ {name:<52} [{status}]")

    if verbose:
        for future in not_done:
            name = future_to_cfg[future].name
            print(
                f"  ✗ {name:<52} [không phản hồi trong {LIMIT_RESPONSE_DURATION}s]"
            )

    candidates = [cfg for cfg in MODELS if cfg.name in responded]
    return candidates, responded


def chat_complete(
    client: OpenAI,
    cfg: ModelConfig,
    messages: list[dict],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.35,
    timeout: float = 120,
) -> str | None:
    """Gọi API không stream; trả về nội dung text hoặc None."""
    response = client.chat.completions.create(
        model=cfg.name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.7,
        stream=False,
        extra_body=cfg.extra_body or None,
        timeout=timeout,
    )
    if not response.choices:
        return None
    content = response.choices[0].message.content
    return content.strip() if content else None


def complete_with_fallback(
    client: OpenAI,
    candidates: list[ModelConfig],
    messages: list[dict],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.35,
) -> tuple[str, str]:
    """Thử lần lượt các model; trả về (text, model_name)."""
    last_error: Exception | None = None
    for cfg in candidates:
        try:
            text = chat_complete(
                client,
                cfg,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if text:
                return text, cfg.name
        except Exception as e:
            last_error = e
            continue
    msg = str(last_error) if last_error else "không có model trả về nội dung"
    raise RuntimeError(f"Dịch thất bại: {msg}")
