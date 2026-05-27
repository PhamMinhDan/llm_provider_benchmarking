import base64
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests

from nvidia_provider.context import context

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def read_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def chat(
    messages: list[dict],
    api_key: str | None = None,
    stream: bool = True,
    max_tokens: int = 4096,
    temperature: float = 0.60,
    top_p: float = 0.70,
    enable_thinking: bool = False,
) -> None:
    api_key = api_key or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY không được tìm thấy. Hãy set biến môi trường hoặc truyền vào tham số api_key."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream" if stream else "application/json",
    }

    payload = {
        "model": "qwen/qwq-32b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }

    response = requests.post(INVOKE_URL, headers=headers, json=payload, stream=stream)
    response.raise_for_status()

    if stream:
        for line in response.iter_lines():
            if line:
                print(line.decode("utf-8"))
    else:
        print(response.json())


if __name__ == "__main__":
    chat(messages=context)
