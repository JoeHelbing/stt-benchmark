"""Main benchmark orchestrator."""

import json
from datetime import datetime
from pathlib import Path

from .samples import load_librispeech_samples
from .asr.cohere import CohereASR
from .asr.gemma4 import Gemma4ASR
from .postprocess.lmstudio import LMStudioPostProcessor
from .metrics import wer, track_vram


def run_asr_benchmark(
    n_samples: int = 50,
    output_dir: str = "data/results",
) -> dict:
    """Run all ASR models on LibriSpeech samples."""
    print(f"Loading {n_samples} LibriSpeech samples...")
    samples = load_librispeech_samples(n=n_samples)

    asr_models = [
        CohereASR(),
        Gemma4ASR(variant="e2b"),
        Gemma4ASR(variant="e4b"),
    ]

    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_samples": n_samples,
            "dataset": "librispeech-test-clean",
        },
        "asr": {},
        "postprocess": {},
    }

    # Stage 1: ASR
    for model in asr_models:
        model_name = model.name
        print(f"\n{'='*60}")
        print(f"Running ASR: {model_name}")
        print(f"{'='*60}")

        results = []
        try:
            with track_vram() as vram:
                for i, sample in enumerate(samples):
                    print(f"  [{i+1}/{len(samples)}] {sample['id']} ({sample['duration']:.1f}s)")
                    result = model.transcribe(sample["audio"], sample["sample_rate"])
                    result["sample_id"] = sample["id"]
                    result["duration_s"] = sample["duration"]
                    result["rtfx"] = sample["duration"] / result["time_s"] if result["time_s"] > 0 else 0
                    if sample["text"]:
                        result["wer"] = wer(sample["text"], result["text"])
                        result["reference"] = sample["text"]
                    results.append(result)
        except Exception as e:
            print(f"  ERROR: {model_name} failed: {e}")
            model.unload()
            continue

        all_results["asr"][model_name] = {
            "results": results,
            "vram_peak_mb": vram.peak_mb,
            "vram_delta_mb": vram.delta_mb,
            "load_time_s": results[0]["load_time_s"] if results else None,
            "avg_wer": sum(r.get("wer", 0) for r in results) / len(results) if results else 0,
            "avg_rtfx": sum(r["rtfx"] for r in results) / len(results) if results else 0,
        }

        avg_wer = all_results["asr"][model_name]["avg_wer"]
        avg_rtfx = all_results["asr"][model_name]["avg_rtfx"]
        print(f"  -> Avg WER: {avg_wer:.2%}, Avg RTFx: {avg_rtfx:.1f}x, VRAM: {vram.peak_mb:.0f} MB")

        model.unload()

    # Stage 2: Post-processing
    postprocessors = [
        LMStudioPostProcessor(model="qwen3.5-4b"),
        LMStudioPostProcessor(model="gemma-4-e2b-it"),
    ]

    for pp in postprocessors:
        pp_name = pp.name
        print(f"\n{'='*60}")
        print(f"Running Post-Processing: {pp_name}")
        print(f"{'='*60}")

        pp_results = {}
        for asr_name, asr_data in all_results["asr"].items():
            print(f"  Processing {asr_name} outputs...")
            pp_for_asr = []
            for r in asr_data["results"]:
                pp_result = pp.process(r["text"])
                pp_result["sample_id"] = r["sample_id"]
                pp_result["raw_text"] = r["text"]
                if r.get("reference"):
                    pp_result["wer_before"] = r.get("wer", 0)
                    pp_result["wer_after"] = wer(r["reference"], pp_result["text"].lower())
                    pp_result["reference"] = r["reference"]
                pp_for_asr.append(pp_result)
            pp_results[asr_name] = pp_for_asr

        all_results["postprocess"][pp_name] = pp_results

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = out_path / f"benchmark_{ts}.json"
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {result_file}")

    return all_results


if __name__ == "__main__":
    run_asr_benchmark(n_samples=50)
