# STT Benchmark: Cohere Transcribe vs Gemma 4

Comparing speech-to-text accuracy and speed across dedicated ASR and multimodal
LLM approaches, plus evaluating LLM post-processing quality for transcript cleanup.

## Models Tested

### ASR (Speech-to-Text)

| Model | Type | Parameters | Quantization |
|-------|------|-----------|--------------|
| [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | Dedicated ASR (Conformer) | 2B | BF16 |
| [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) | Multimodal LLM | 2.3B effective | 4-bit (BnB) |
| [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it) | Multimodal LLM | 4B effective | 4-bit (BnB) |

### Post-Processing (Transcript Cleanup)

| Model | Parameters | Via |
|-------|-----------|-----|
| [Qwen 3.5 4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 4B | LM Studio |
| [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) | 2.3B | LM Studio |

Post-processing uses a real-world transcript cleanup prompt from
[Handy](https://github.com/cjpais/Handy) (open-source speech-to-text app),
covering filler removal, spoken corrections, punctuation, and number formatting.

## Dataset

[LibriSpeech test-clean](https://huggingface.co/datasets/openslr/librispeech_asr) --
50 randomly sampled utterances with ground-truth transcriptions for WER calculation.

## Results

_Results will be added after benchmark run._

## Metrics

- **WER** -- Word Error Rate vs ground truth (lower is better)
- **RTFx** -- Real-Time Factor, how many X faster than real-time (higher is better)
- **VRAM** -- Peak GPU memory usage during inference
- **Post-process WER** -- WER after LLM cleanup (does post-processing help or hurt?)

## Hardware

- **GPU:** NVIDIA RTX 3090 (24 GB VRAM)
- **OS:** Ubuntu Linux 24.04

## Running

```bash
# Clone and install
git clone https://github.com/JoeHelbing/stt-benchmark.git
cd stt-benchmark
uv sync

# Prerequisites:
# - NVIDIA GPU with CUDA
# - LM Studio running on localhost:1234 with qwen3.5-4b and gemma-4-e2b-it loaded

# Run benchmark (downloads models on first run)
uv run python -m stt_benchmark.runner

# Generate graphs from results
uv run python -m stt_benchmark.analyze data/results/benchmark_TIMESTAMP.json
```

## Project Structure

```
src/stt_benchmark/
  samples.py          # Audio sample loaders (LibriSpeech, Handy recordings)
  metrics.py          # WER calculation, VRAM tracking
  runner.py           # Main benchmark orchestrator
  analyze.py          # Graph generation (Plotly)
  asr/
    cohere.py         # Cohere Transcribe runner
    gemma4.py         # Gemma 4 E2B/E4B runner
  postprocess/
    prompt.py         # Handy cleanup prompt
    lmstudio.py       # LM Studio API client
data/results/         # Benchmark results (JSON)
figures/              # Generated graphs (HTML + PNG)
```

## License

MIT
