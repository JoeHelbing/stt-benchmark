"""Gemma 4 (E2B/E4B) ASR runner via multimodal audio input."""

import os
import time
import wave
import tempfile

import torch
import numpy as np

VARIANTS = {
    "e2b": "google/gemma-4-E2B-it",
    "e4b": "google/gemma-4-E4B-it",
}

TRANSCRIBE_PROMPT = (
    "Transcribe the following speech segment in English into English text.\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits (e.g., 1.7 not one point seven)."
)


class Gemma4ASR:
    def __init__(self, variant: str = "e2b"):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Use: {list(VARIANTS)}")
        self.variant = variant
        self.model_id = VARIANTS[variant]
        self.model = None
        self.processor = None
        self._load_time: float | None = None

    def load(self):
        if self.model is not None:
            return
        from transformers import AutoProcessor, AutoModelForMultimodalLM

        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._load_time = time.perf_counter() - t0

    def _audio_to_wav_path(self, audio: np.ndarray, sample_rate: int) -> str:
        """Write audio array to a temp WAV file (Gemma expects file paths)."""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            pcm = (audio * 32767).astype(np.int16).tobytes()
            wf.writeframes(pcm)
        return tmp.name

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> dict:
        self.load()
        wav_path = self._audio_to_wav_path(audio, sample_rate)

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": wav_path},
                        {"type": "text", "text": TRANSCRIBE_PROMPT},
                    ],
                }
            ]

            t0 = time.perf_counter()
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(self.model.device)
            input_len = inputs["input_ids"].shape[-1]

            outputs = self.model.generate(**inputs, max_new_tokens=512)
            raw = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
            elapsed = time.perf_counter() - t0

            return {
                "text": raw.strip().lower(),
                "time_s": elapsed,
                "model": f"gemma-4-{self.variant}",
                "load_time_s": self._load_time,
            }
        finally:
            os.unlink(wav_path)

    def unload(self):
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        torch.cuda.empty_cache()

    @property
    def name(self) -> str:
        return f"Gemma 4 {self.variant.upper()}"
