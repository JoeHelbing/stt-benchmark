"""Load audio samples for benchmarking."""

import io

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset


def load_librispeech_samples(
    n: int = 50,
    config: str = "clean",
    split: str = "test",
    seed: int = 42,
) -> list[dict]:
    """Load n samples from LibriSpeech test-clean with ground truth."""
    ds = load_dataset(
        "openslr/librispeech_asr",
        config,
        split=split,
    )
    # Disable automatic audio decoding (avoids torchcodec dependency);
    # we decode manually with soundfile instead.
    ds = ds.cast_column("audio", Audio(decode=False))

    # Deterministic shuffle, pick n
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    samples = []
    for row in ds:
        audio_bytes = row["audio"]["bytes"]
        arr, sr = sf.read(io.BytesIO(audio_bytes))
        arr = arr.astype(np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        samples.append({
            "id": str(row["id"]),
            "audio": arr,
            "sample_rate": sr,
            "text": row["text"].lower(),
            "duration": len(arr) / sr,
        })
    return samples


def load_handy_recordings(
    recordings_dir: str = "~/.local/share/com.pais.handy/recordings",
) -> list[dict]:
    """Load WAV files from Handy's recordings directory (no ground truth)."""
    import os
    from pathlib import Path

    rec_dir = Path(os.path.expanduser(recordings_dir))
    if not rec_dir.exists():
        return []

    samples = []
    for wav in sorted(rec_dir.glob("*.wav")):
        data, sr = sf.read(wav)
        if data.ndim > 1:
            data = data.mean(axis=1)
        samples.append({
            "id": wav.stem,
            "audio": data.astype(np.float32),
            "sample_rate": sr,
            "text": None,  # no ground truth
            "duration": len(data) / sr,
        })
    return samples
