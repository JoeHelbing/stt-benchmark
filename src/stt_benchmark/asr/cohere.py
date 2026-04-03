"""Cohere Transcribe ASR runner."""

import time

import numpy as np
import torch

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"


class CohereASR:
    def __init__(self):
        self.model = None
        self.processor = None
        self._load_time: float | None = None

    def load(self):
        if self.model is not None:
            return
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map="auto",
        )
        self._load_time = time.perf_counter() - t0

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = "en",
    ) -> dict:
        self.load()
        t0 = time.perf_counter()
        inputs = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            language=language,
        )
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)

        outputs = self.model.generate(**inputs, max_new_tokens=256)
        text = self.processor.decode(outputs, skip_special_tokens=True)
        elapsed = time.perf_counter() - t0

        # decode returns a list of strings
        if isinstance(text, list):
            text = text[0]

        return {
            "text": text.strip().lower(),
            "time_s": elapsed,
            "model": "cohere-transcribe",
            "load_time_s": self._load_time,
        }

    def unload(self):
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        torch.cuda.empty_cache()

    @property
    def name(self) -> str:
        return "Cohere Transcribe 2B"
