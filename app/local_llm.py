"""Optional local LLM analysis through Ollama.

The application keeps working with the deterministic analyzer when Ollama is
not installed or the local model is unavailable.
"""

from __future__ import annotations

import json
from urllib.error import URLError
from urllib.request import Request, urlopen


def analyze_locally(transcript: str, model: str = "qwen2.5:7b") -> dict | None:
    prompt = f"""Ты анализируешь протокол рабочего совещания.
Верни только JSON следующего вида:
{{"summary":"краткое саммари", "tasks":[{{"task":"что сделать", "responsible":"кто", "deadline":"срок"}}]}}
Не придумывай данные. Если ответственный или срок неизвестен, напиши "Не указан".
Транскрипт:
{transcript}
"""
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False, "format": "json"}).encode("utf-8")
    request = Request("http://127.0.0.1:11434/api/generate", data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
        parsed = json.loads(result.get("response", "{}"))
        if not isinstance(parsed.get("tasks"), list) or not isinstance(parsed.get("summary"), str):
            return None
        return parsed
    except (OSError, URLError, TimeoutError, json.JSONDecodeError):
        return None
