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
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        ).to("cuda")
        self.model.eval()
        self._load_time = time.perf_counter() - t0

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = "en",
    ) -> dict:
        self.load()
        t0 = time.perf_counter()
        texts = self.model.transcribe(
            processor=self.processor,
            audio_arrays=[audio],
            sample_rates=[sample_rate],
            language=language,
        )
        elapsed = time.perf_counter() - t0
        return {
            "text": texts[0].strip().lower(),
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
