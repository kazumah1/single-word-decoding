"""Helpers for running multiple training configurations and aggregating results.

The wrapper notebook is responsible for:
  1. Building the per-config Experiment dict (set `config_name` to a stable slug).
  2. Running the Experiment.
  3. Calling `Experiment.collect_run_artifacts(MULTI_CFG_OUT)` to copy figures
     + eval_results.json + training_history.csv into MULTI_CFG_OUT/<config_name>/.
  4. Calling `Experiment.cleanup_run_artifacts()` to free disk before the next run.
  5. Calling `free_memory()` between runs.

After every config has been trained, call:
  - `aggregate_eval_results(MULTI_CFG_OUT)` → pandas DataFrame
  - `plot_combined_loss_curves(MULTI_CFG_OUT, MULTI_CFG_OUT/'combined_curves.png')`
"""
from __future__ import annotations

import gc
import json
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch


def free_memory() -> None:
    """Drop CUDA / Python references between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def cleanup_run_artifacts(
    run_dir: str | Path,
    keep: Iterable[str] = (
        "figures", "eval_results.json", "training_history.csv", "config.yaml",
    ),
) -> None:
    """Drop checkpoints, caches, retrieval_outputs, lightning_logs, job files
    etc. under run_dir; keep only the named files / directories.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return
    keep_set = set(keep)
    for entry in run_dir.iterdir():
        if entry.name in keep_set:
            continue
        try:
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink()
        except OSError:
            pass


def aggregate_eval_results(
    runs_root: str | Path,
    split: str = "test",
    dataloader_idx: int = 0,
) -> pd.DataFrame:
    """Walk runs_root/<config_name>/eval_results.json, return a wide DataFrame
    with one row per configuration and one column per metric.

    The shape mirrors `pipeline_results.png`: rows are configs, columns are
    metrics (test_*). Pass split='val' for validation-time numbers.
    """
    runs_root = Path(runs_root)
    rows: list[dict] = []
    if not runs_root.exists():
        return pd.DataFrame()
    for cfg_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        f = cfg_dir / "eval_results.json"
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        key = f"{split}_{dataloader_idx}"
        block = data.get(key, {})
        if not block:
            continue
        row = {"config_name": cfg_dir.name}
        for k, v in block.items():
            if k == "config_name":
                continue
            row[k] = v
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("config_name").sort_index()
    return df


def plot_combined_loss_curves(
    runs_root: str | Path,
    out_path: str | Path,
    metrics: Iterable[str] = ("train_cnn_loss", "val_cnn_loss"),
    title: str = "Combined Training/Validation Loss Curves",
) -> Path | None:
    """Overlay loss curves for every configuration in runs_root.

    Saves a single PNG to out_path with one panel per metric, distinct colour
    per configuration. Returns the output path, or None if no data was found.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs_root = Path(runs_root)
    if not runs_root.exists():
        return None
    cfg_dirs = sorted(p for p in runs_root.iterdir() if p.is_dir())
    metrics = list(metrics)
    if not cfg_dirs or not metrics:
        return None

    fig, axes = plt.subplots(
        1, len(metrics), figsize=(6 * len(metrics), 4), squeeze=False,
    )
    cmap = plt.get_cmap("tab10")
    plotted_anything = False
    for ax_idx, metric in enumerate(metrics):
        ax = axes[0][ax_idx]
        for c_i, cfg_dir in enumerate(cfg_dirs):
            csv_path = cfg_dir / "training_history.csv"
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            sub = df[df["metric"] == metric]
            if sub.empty:
                continue
            sub = (
                sub.groupby("epoch", as_index=False)["value"].mean()
                   .sort_values("epoch")
            )
            ax.plot(
                sub["epoch"], sub["value"], marker="o", markersize=3,
                color=cmap(c_i % 10), label=cfg_dir.name, linewidth=1.5,
            )
            plotted_anything = True
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    if not plotted_anything:
        plt.close(fig)
        return None
    fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Suggested per-configuration deltas
# ---------------------------------------------------------------------------
# These are starting points for the wrapper notebook — apply via
# `update_config(default_config, CONFIG_DELTAS[name])`.
#
# IMPORTANT: the t5-large variants currently require a TransformerEncoder
# (x-transformers) brain transformer because transformer.py only ships
# TransformerEncoder + LlamaTransformer. If a dedicated T5-encoder
# transformer is added, swap `transformer_config.name` accordingly.
# ---------------------------------------------------------------------------
CONFIG_DELTAS: dict[str, dict] = {
    "simpleconv_t5large": {
        "config_name": "simpleconv_t5large",
        "brain_model_config.name": "SimpleConvTimeAgg",
        "data.feature.model_name": "t5-large",
        "transformer_config": {
            "name": "TransformerEncoder",
            "depth": 6, "heads": 8,
        },
    },
    "simpleconv_llama8b": {
        "config_name": "simpleconv_llama8b",
        "brain_model_config.name": "SimpleConvTimeAgg",
        "data.feature.model_name": "meta-llama/Meta-Llama-3.1-8B",
    },
    "multiscaleconv_t5large": {
        "config_name": "multiscaleconv_t5large",
        "brain_model_config.name": "MultiScaleSimpleConvTimeAgg",
        "data.feature.model_name": "t5-large",
        "transformer_config": {
            "name": "TransformerEncoder",
            "depth": 6, "heads": 8,
        },
    },
    "multiscaleconv_llama8b": {
        "config_name": "multiscaleconv_llama8b",
        "brain_model_config.name": "MultiScaleSimpleConvTimeAgg",
        "data.feature.model_name": "meta-llama/Meta-Llama-3.1-8B",
    },
    # The pretrain config is a TWO-STAGE recipe. Run the pretrain stage with
    # `pretrain_mode="maeeg"` (or "simclr") and brain_model_config.name set to
    # "MultiScaleSimpleConvPretrain", save the resulting checkpoint, then run
    # the finetune stage with this delta and `pretrain_checkpoint=<path>`.
    "multiscaleconv_pretrain_llama8b": {
        "config_name": "multiscaleconv_pretrain_llama8b",
        "brain_model_config.name": "MultiScaleSimpleConvTimeAgg",
        "data.feature.model_name": "meta-llama/Meta-Llama-3.1-8B",
    },
}
