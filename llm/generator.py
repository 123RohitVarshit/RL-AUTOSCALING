"""Groq code generation with strict validation and honest source tags."""
from __future__ import annotations

import importlib
import importlib.util
import os
import tempfile
import time
from typing import Tuple

from llm.templates import SYSTEM_PROMPT, USER_PROMPT

MOCK_CODE = """class AutoscalerAgent:
    def __init__(self):
        self.last_scale = -100
        self.history = []

    def decide(self, state: dict) -> int:
        queue = state.get("queue", 0)
        step = state.get("step", 0)
        rate = state.get("arrival_rate", 5)
        self.history.append(rate)

        if step - self.last_scale < 5:
            return 0

        trend = 0
        if len(self.history) > 5:
            trend = self.history[-1] - self.history[-5]

        if queue > 12 or (trend > 1.5 and queue > 8):
            self.last_scale = step
            return 1

        if queue < 3 and trend < 0:
            self.last_scale = step
            return 2

        return 0
"""


def _api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if key.startswith("gsk_"):
        return key
    try:
        from google.colab import userdata
        k = userdata.get("GROQ_API_KEY")
        if k:
            os.environ["GROQ_API_KEY"] = k
            return k
    except Exception:
        pass
    return ""


def _validate_code(code: str) -> bool:
    """Validate syntax, class existence, and runtime return values."""
    try:
        compile(code, "<string>", "exec")
    except SyntaxError:
        return False

    try:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".py", delete=False, mode="w", encoding="utf-8"
        )
        tmp.write(code)
        tmp.close()

        spec = importlib.util.spec_from_file_location("check_agent", tmp.name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        os.unlink(tmp.name)

        if not hasattr(module, "AutoscalerAgent"):
            return False

        agent = module.AutoscalerAgent()

        test_states = [
            {"active_servers": 2, "booting_servers": 0, "queue": 0.0,
             "arrival_rate": 5.0, "step": 0, "time_since_scale": 100},
            {"active_servers": 2, "booting_servers": 0, "queue": 25.0,
             "arrival_rate": 16.0, "step": 30, "time_since_scale": 20},
            {"active_servers": 8, "booting_servers": 0, "queue": 1.0,
             "arrival_rate": 2.0, "step": 200, "time_since_scale": 60},
            {"active_servers": 3, "booting_servers": 2, "queue": 48.0,
             "arrival_rate": 22.0, "step": 90, "time_since_scale": 2},
        ]

        for s in test_states:
            action = agent.decide(s)
            if action not in (0, 1, 2):
                return False

        return True

    except Exception:
        return False


def generate_agent_code(max_attempts: int = 5,
                        backoff: float = 3.0) -> Tuple[str, str]:
    """Returns (code, source).

    source values:
      groq              valid first-attempt LLM output
      groq_attempt_N    valid LLM output after retries
      mock_no_key       no API key available
      mock_no_openai    openai package missing
      mock_invalid      all LLM attempts failed validation
    """
    key = _api_key()
    if not key:
        print("[INFO] No Groq key available; using mock.")
        return MOCK_CODE, "mock_no_key"

    try:
        from openai import OpenAI
    except ImportError:
        print("[WARN] openai package missing; using mock.")
        return MOCK_CODE, "mock_no_openai"

    client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")

    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": USER_PROMPT},
                ],
                temperature=0.3,
                max_tokens=900,
            )

            code = (resp.choices[0].message.content or "").strip()
            code = code.replace("```python", "").replace("```", "").strip()

            if not code:
                print(f"[WARN] Attempt {attempt}: empty response.")
                continue

            if _validate_code(code):
                source = "groq" if attempt == 1 else f"groq_attempt_{attempt}"
                print(f"[OK] Valid Groq code on attempt {attempt} ({len(code)} chars).")
                return code, source

            print(f"[WARN] Attempt {attempt}: code failed validation.")

        except Exception as e:
            print(f"[WARN] Attempt {attempt} API error: {e}")

        if attempt < max_attempts:
            time.sleep(backoff * attempt)

    print("[ERROR] All Groq attempts failed validation or API calls.")
    print("        Using mock, but this is NOT a valid LLM experimental result.")
    return MOCK_CODE, "mock_invalid"


def load_agent_from_code(code: str):
    tmp = tempfile.NamedTemporaryFile(
        suffix=".py", delete=False, mode="w", encoding="utf-8"
    )
    tmp.write(code)
    tmp.close()

    spec = importlib.util.spec_from_file_location("llm_agent", tmp.name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.unlink(tmp.name)

    if not hasattr(module, "AutoscalerAgent"):
        raise ValueError("No AutoscalerAgent class found.")

    return module.AutoscalerAgent()
