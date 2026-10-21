"""Concept activations for a store or per-slide embedding files.

Writes, per slide, ``<slide_id>.concepts.npz`` with the TopK indices and values
([n, k] each), which is what downstream profiling / exemplar lookup consume.

    python -m picasso.infer --config configs/sae/infer.yaml
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib

import numpy as np
import torch

from picasso.config import cli_config
from picasso.model import load

logger = logging.getLogger("picasso.infer")


@dataclasses.dataclass(frozen=True)
class InferConfig:
    checkpoint: pathlib.Path
    embeddings_dir: pathlib.Path
    output_dir: pathlib.Path
    store_dir: pathlib.Path | None = None   # needed when the checkpoint used normalize_input
    field: str = "cls"
    batch_size: int = 8192
    device: str = "cuda"


@torch.inference_mode()
def run(cfg: InferConfig) -> pathlib.Path:
    with open(cfg.checkpoint, "rb") as fd:
        header = json.loads(fd.readline())
    sae = load(cfg.checkpoint, cfg.device)
    mean = std = None
    if header.get("normalize_input"):
        meta = json.loads((cfg.store_dir / "metadata.json").read_text())["fields"][cfg.field]
        mean = torch.tensor(meta["train_mean"], device=cfg.device)
        std = torch.tensor(meta["train_std"], device=cfg.device)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(cfg.embeddings_dir.glob(f"*.{cfg.field}.npy"))
    for p in files:
        sid = p.name.removesuffix(f".{cfg.field}.npy")
        out = cfg.output_dir / f"{sid}.concepts.npz"
        if out.exists():
            continue
        x_all = np.load(p)
        idx_all = np.empty((len(x_all), sae.cfg.top_k), dtype=np.int32)
        val_all = np.empty((len(x_all), sae.cfg.top_k), dtype=np.float16)
        for s in range(0, len(x_all), cfg.batch_size):
            x = torch.from_numpy(x_all[s : s + cfg.batch_size]).to(cfg.device).float()
            if mean is not None:
                x = (x - mean) / std
            h = sae.pre_activation(x)
            v, i = h.topk(sae.cfg.top_k, dim=-1)
            idx_all[s : s + len(x)] = i.cpu().numpy()
            val_all[s : s + len(x)] = v.clamp_min(0).cpu().numpy().astype(np.float16)
        np.savez(out, idx=idx_all, val=val_all)
    logger.info("wrote concepts for %d slides to %s", len(files), cfg.output_dir)
    return cfg.output_dir


if __name__ == "__main__":
    run(cli_config(InferConfig, __doc__))
