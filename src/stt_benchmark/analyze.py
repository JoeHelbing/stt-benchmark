"""Generate benchmark analysis graphs from results JSON."""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def fig_asr_wer(results: dict) -> go.Figure:
    """Bar chart: WER by ASR model."""
    models = []
    wers = []
    for name, data in results["asr"].items():
        models.append(name)
        wers.append(data["avg_wer"] * 100)

    fig = go.Figure(go.Bar(x=models, y=wers, text=[f"{w:.1f}%" for w in wers], textposition="auto"))
    fig.update_layout(title="ASR Word Error Rate (lower is better)", yaxis_title="WER %", template="plotly_white")
    return fig


def fig_asr_rtfx(results: dict) -> go.Figure:
    """Bar chart: RTFx by ASR model."""
    models = []
    rtfxs = []
    for name, data in results["asr"].items():
        models.append(name)
        rtfxs.append(data["avg_rtfx"])

    fig = go.Figure(go.Bar(x=models, y=rtfxs, text=[f"{r:.1f}x" for r in rtfxs], textposition="auto"))
    fig.update_layout(title="ASR Speed: Real-Time Factor (higher is better)", yaxis_title="RTFx", template="plotly_white")
    return fig


def fig_vram(results: dict) -> go.Figure:
    """Bar chart: peak VRAM by ASR model."""
    models = []
    vrams = []
    for name, data in results["asr"].items():
        models.append(name)
        vrams.append(data["vram_peak_mb"] / 1024)

    fig = go.Figure(go.Bar(x=models, y=vrams, text=[f"{v:.1f} GB" for v in vrams], textposition="auto"))
    fig.update_layout(title="Peak VRAM Usage", yaxis_title="VRAM (GB)", template="plotly_white")
    return fig


def fig_wer_by_duration(results: dict) -> go.Figure:
    """Line chart: WER by audio duration bucket."""
    fig = go.Figure()
    for name, data in results["asr"].items():
        df = pd.DataFrame(data["results"])
        if "wer" not in df.columns:
            continue
        df["duration_bucket"] = pd.cut(df["duration_s"], bins=[0, 5, 10, 20, 60], labels=["0-5s", "5-10s", "10-20s", "20s+"])
        grouped = df.groupby("duration_bucket", observed=True)["wer"].mean() * 100
        fig.add_trace(go.Scatter(x=grouped.index.astype(str), y=grouped.values, mode="lines+markers", name=name))

    fig.update_layout(title="WER by Audio Duration", yaxis_title="WER %", xaxis_title="Duration", template="plotly_white")
    return fig


def fig_postprocess_impact(results: dict) -> go.Figure:
    """Grouped bar: WER before vs after post-processing."""
    categories = []
    for pp_name, pp_data in results["postprocess"].items():
        for asr_name, entries in pp_data.items():
            if not entries or "wer_before" not in entries[0]:
                continue
            label = f"{asr_name}\n+ {pp_name}"
            avg_before = sum(e["wer_before"] for e in entries) / len(entries) * 100
            avg_after = sum(e["wer_after"] for e in entries) / len(entries) * 100
            categories.append({"pipeline": label, "stage": "Raw ASR", "wer": avg_before})
            categories.append({"pipeline": label, "stage": "Post-Processed", "wer": avg_after})

    if categories:
        df = pd.DataFrame(categories)
        fig = px.bar(df, x="pipeline", y="wer", color="stage", barmode="group", text_auto=".1f")
        fig.update_layout(title="Post-Processing Impact on WER", yaxis_title="WER %", template="plotly_white")
        return fig
    return go.Figure()


def fig_pipeline_heatmap(results: dict) -> go.Figure:
    """Heatmap: ASR model x Post-processor -> WER."""
    asr_names = list(results["asr"].keys())
    pp_names = list(results["postprocess"].keys())

    all_cols = ["No Post-Processing"] + pp_names
    z = []
    for asr_name in asr_names:
        row = [results["asr"][asr_name]["avg_wer"] * 100]
        for pp_name in pp_names:
            entries = results["postprocess"].get(pp_name, {}).get(asr_name, [])
            if entries and "wer_after" in entries[0]:
                avg = sum(e["wer_after"] for e in entries) / len(entries) * 100
            else:
                avg = None
            row.append(avg)
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z, x=all_cols, y=asr_names,
        text=[[f"{v:.1f}%" if v else "N/A" for v in row] for row in z],
        texttemplate="%{text}", colorscale="RdYlGn_r",
    ))
    fig.update_layout(title="Full Pipeline WER: ASR x Post-Processor", template="plotly_white")
    return fig


def generate_all(results_path: str, output_dir: str = "figures"):
    results = load_results(results_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    figures = {
        "01_asr_wer": fig_asr_wer,
        "02_asr_rtfx": fig_asr_rtfx,
        "03_vram": fig_vram,
        "04_wer_by_duration": fig_wer_by_duration,
        "05_postprocess_impact": fig_postprocess_impact,
        "06_pipeline_heatmap": fig_pipeline_heatmap,
    }

    for name, fn in figures.items():
        print(f"Generating {name}...")
        fig = fn(results)
        fig.write_html(str(out / f"{name}.html"))
        fig.write_image(str(out / f"{name}.png"), width=1000, height=600, scale=2)

    print(f"\n{len(figures)} figures saved to {output_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m stt_benchmark.analyze data/results/benchmark_TIMESTAMP.json")
        sys.exit(1)
    generate_all(sys.argv[1])
