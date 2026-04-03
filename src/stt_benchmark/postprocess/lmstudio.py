"""Post-processing via LM Studio OpenAI-compatible API."""

import time

from openai import OpenAI

from .prompt import HANDY_CLEANUP_PROMPT


class LMStudioPostProcessor:
    def __init__(
        self,
        model: str = "qwen3.5-4b",
        base_url: str = "http://localhost:1234/v1",
    ):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key="not-needed")

    def process(self, raw_text: str) -> dict:
        prompt = HANDY_CLEANUP_PROMPT.replace("${output}", raw_text)
        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
        )
        elapsed = time.perf_counter() - t0
        return {
            "text": response.choices[0].message.content.strip(),
            "time_s": elapsed,
            "model": self.model,
        }

    @property
    def name(self) -> str:
        return f"Post: {self.model}"
