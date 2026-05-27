import os
import sys

from openai import OpenAI

_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
_REASONING_COLOR = "\033[90m" if _USE_COLOR else ""
_RESET_COLOR = "\033[0m" if _USE_COLOR else ""

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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",
)


completion = client.chat.completions.create(
    model="z-ai/glm5",
    messages=[{"role": "user", "content": ""}],
    temperature=1,
    top_p=1,
    max_tokens=16384,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}
    },
    stream=True,
)

for chunk in completion:
    if not getattr(chunk, "choices", None):
        continue
    if len(chunk.choices) == 0 or getattr(chunk.choices[0], "delta", None) is None:
        continue
    delta = chunk.choices[0].delta
    reasoning = getattr(delta, "reasoning_content", None)
    if reasoning:
        print(f"{_REASONING_COLOR}{reasoning}{_RESET_COLOR}", end="")
    if getattr(delta, "content", None) is not None:
        print(delta.content, end="")
