from stt_benchmark.samples import load_librispeech_samples


def test_load_librispeech_samples():
    samples = load_librispeech_samples(n=5)
    assert len(samples) == 5
    for s in samples:
        assert "id" in s
        assert "audio" in s  # numpy array
        assert "text" in s  # ground truth
        assert "duration" in s  # seconds
        assert s["audio"].ndim == 1
        assert s["duration"] > 0
