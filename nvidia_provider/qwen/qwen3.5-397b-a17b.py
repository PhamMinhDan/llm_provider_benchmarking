
import requests, base64

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = True

def read_b64(path):
  with open(path, "rb") as f:
    return base64.b64encode(f.read()).decode()

headers = {
  "Authorization": "Bearer $NVIDIA_API_KEY",
  "Accept": "text/event-stream" if stream else "application/json"
}

payload = {
  "model": "qwen/qwen3.5-397b-a17b",
  "messages": [{"role":"user","content":""}],
  "max_tokens": 16384,
  "temperature": 0.60,
  "top_p": 0.95,
  "top_k": 20,
  "presence_penalty": 0,
  "repetition_penalty": 1,
  "stream": stream,
  "chat_template_kwargs": {"enable_thinking":True},
}

response = requests.post(invoke_url, headers=headers, json=payload, stream=stream)
if stream:
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
else:
    print(response.json())
