"""Evaluate an SAE checkpoint on the held-out test store.

Reports explained variance (global 1-FVU and per-sample), normalized MSE, cosine, L0,
dead / dense latents and the number of concepts with activation frequency > 0.1 %
(paper: ~10,000 of 40,960). Saves ``summary.json`` and ``latent_freq.npy``.
Optionally PCA and k-means baselines at matched component counts (Fig. 2a).

    python -m picasso.evaluate --config configs/sae/evaluate.yaml
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib

import numpy as np
import torch

from picasso.config import cli_config, dump_json
from picasso.loader import EmbeddingStore
from picasso.metrics import Accumulator
from picasso.model import load

logger = logging.getLogger("picasso.evaluate")


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    checkpoint: pathlib.Path
    store_dir: pathlib.Path
    output_dir: pathlib.Path
    field: str = "cls"
    split: str = "test"
    batch_size: int = 8192
    max_rows: int | None = None
    baselines: bool = False
    baseline_components: tuple[int, ...] = (10, 30, 50, 80, 100)
    baseline_fit_rows: int = 200_000
    device: str = "cuda"


@torch.inference_mode()
def sae_metrics(cfg: EvalConfig, header: dict) -> tuple[dict, np.ndarray]:
    sae = load(cfg.checkpoint, cfg.device)
    store = EmbeddingStore(cfg.store_dir, cfg.field, cfg.split, cfg.max_rows)
    mean, std = store.mean.to(cfg.device), store.std.to(cfg.device)
    norm = header.get("normalize_input", False)
    acc = Accumulator(sae.cfg.d_sae, store.d, cfg.device)
    for x in store.sequential(cfg.batch_size):
        x = x.to(cfg.device)
        if norm:
            x = (x - mean) / std
        f, x_hat = sae(x)
        acc.update(x, x_hat, f)
    return acc.summary(), acc.frequencies().numpy()


def baselines(cfg: EvalConfig) -> dict:
    """Explained variance of PCA / k-means reconstructions with n components (train-fit, test-eval)."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA

    train = EmbeddingStore(cfg.store_dir, cfg.field, "train")
    test = EmbeddingStore(cfg.store_dir, cfg.field, cfg.split)
    xtr = train.random_rows(cfg.baseline_fit_rows, seed=0).numpy()               # random rows, all slides
    xte = test.random_rows(cfg.max_rows or cfg.baseline_fit_rows, seed=1).numpy()
    total_var = ((xte - xte.mean(0)) ** 2).sum()
    out = {}
    for n in cfg.baseline_components:
        pca = PCA(n_components=n, random_state=0).fit(xtr)
        rec = pca.inverse_transform(pca.transform(xte))
        out[f"pca_{n}"] = float(1 - ((rec - xte) ** 2).sum() / total_var)
        km = MiniBatchKMeans(n_clusters=n, random_state=0, batch_size=8192, n_init=3).fit(xtr)
        rec = km.cluster_centers_[km.predict(xte)]
        out[f"kmeans_{n}"] = float(1 - ((rec - xte) ** 2).sum() / total_var)
        logger.info("n=%d pca ev=%.4f kmeans ev=%.4f", n, out[f"pca_{n}"], out[f"kmeans_{n}"])
    return out


def run(cfg: EvalConfig) -> dict:
    with open(cfg.checkpoint, "rb") as fd:
        header = json.loads(fd.readline())
    summary, freq = sae_metrics(cfg, header)
    summary = {"checkpoint": str(cfg.checkpoint), "split": cfg.split, **summary}
    if cfg.baselines:
        summary["baselines"] = baselines(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.output_dir / "latent_freq.npy", freq)
    dump_json(cfg.output_dir / "summary.json", summary)
    logger.info("%s", json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    run(cli_config(EvalConfig, __doc__))
