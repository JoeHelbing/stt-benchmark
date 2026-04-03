"""Metrics: WER, VRAM tracking, timing."""

from contextlib import contextmanager
from dataclasses import dataclass

from jiwer import wer as compute_wer


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate between reference and hypothesis."""
    if not reference.strip():
        return 1.0 if hypothesis.strip() else 0.0
    return compute_wer(reference.lower(), hypothesis.lower())


@dataclass
class VRAMSnapshot:
    before_mb: float = 0.0
    peak_mb: float = 0.0
    after_mb: float = 0.0

    @property
    def delta_mb(self) -> float:
        return self.peak_mb - self.before_mb


@contextmanager
def track_vram():
    """Context manager that tracks GPU VRAM usage."""
    import torch

    snapshot = VRAMSnapshot()

    if not torch.cuda.is_available():
        yield snapshot
        return

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    snapshot.before_mb = torch.cuda.memory_allocated() / 1024**2

    yield snapshot

    torch.cuda.synchronize()
    snapshot.peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    snapshot.after_mb = torch.cuda.memory_allocated() / 1024**2
