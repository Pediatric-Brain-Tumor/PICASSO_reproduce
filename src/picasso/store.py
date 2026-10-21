"""Consolidate per-slide embeddings into flat fp16 stores with a slide-level split.

Reads ``embeddings_dir/<slide_id>.<field>.npy`` for every slide with a ``.done`` marker,
assigns each slide to train/val/test (80/10/10, seeded, stratified by project) and writes

  ``output_dir/<field>_{train,val,test}.f16``   row-major [n, d] float16
  ``output_dir/<field>_{train,val,test}.index.parquet``   slide_id, project, start, n
  ``output_dir/splits.json``, ``output_dir/metadata.json`` (d, counts, train mean/std)

    python -m picasso.store --config configs/data/store.yaml
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib

import numpy as np
import polars as pl

from picasso.config import cli_config, dump_json

logger = logging.getLogger("picasso.store")


@dataclasses.dataclass(frozen=True)
class StoreConfig:
    embeddings_dir: pathlib.Path
    tiles_dir: pathlib.Path
    output_dir: pathlib.Path
    fields: tuple[str, ...] = ("cls",)
    projects: tuple[str, ...] | None = None
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 0


def assign_splits(summary: pl.DataFrame, cfg: StoreConfig) -> dict[str, list[str]]:
    rng = np.random.default_rng(cfg.seed)
    splits = {"train": [], "val": [], "test": []}
    for proj, group in summary.group_by("project", maintain_order=True):
        ids = sorted(group["slide_id"].to_list())
        rng.shuffle(ids)
        n = len(ids)
        n_test = int(round(n * cfg.test_frac))
        n_val = int(round(n * cfg.val_frac))
        splits["test"] += ids[:n_test]
        splits["val"] += ids[n_test : n_test + n_val]
        splits["train"] += ids[n_test + n_val :]
    return splits


def write_store(field: str, split: str, ids: list[str], summary: dict[str, str], cfg: StoreConfig, d: int) -> tuple[int, np.ndarray, np.ndarray]:
    rows = []
    n_total = 0
    for sid in ids:
        n = np.load(cfg.embeddings_dir / f"{sid}.{field}.npy", mmap_mode="r").shape[0]
        rows.append({"slide_id": sid, "project": summary[sid], "start": n_total, "n": n})
        n_total += n
    path = cfg.output_dir / f"{field}_{split}.f16"
    out = np.memmap(path, mode="w+", dtype=np.float16, shape=(n_total, d))
    s = np.zeros(d, dtype=np.float64)
    s2 = np.zeros(d, dtype=np.float64)
    for r in rows:
        arr = np.load(cfg.embeddings_dir / f"{r['slide_id']}.{field}.npy")
        out[r["start"] : r["start"] + r["n"]] = arr
        a = arr.astype(np.float64)
        s += a.sum(0)
        s2 += (a * a).sum(0)
    out.flush()
    del out
    pl.DataFrame(rows).write_parquet(cfg.output_dir / f"{field}_{split}.index.parquet")
    return n_total, s, s2


def run(cfg: StoreConfig) -> pathlib.Path:
    summary = pl.read_parquet(cfg.tiles_dir / "slides_summary.parquet").filter(pl.col("n_tiles").is_not_null())
    if cfg.projects:
        summary = summary.filter(pl.col("project").is_in(list(cfg.projects)))
    done = {p.name.removesuffix(".done") for p in cfg.embeddings_dir.glob("*.done")}
    summary = summary.filter(pl.col("slide_id").is_in(sorted(done)))
    logger.info("%d slides with embeddings", summary.height)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    splits = assign_splits(summary, cfg)
    dump_json(cfg.output_dir / "splits.json", splits)
    proj_of = dict(zip(summary["slide_id"], summary["project"]))
    meta = {"fields": {}, "splits": {k: len(v) for k, v in splits.items()}, "seed": cfg.seed}
    for field in cfg.fields:
        d = np.load(cfg.embeddings_dir / f"{splits['train'][0]}.{field}.npy", mmap_mode="r").shape[1]
        info = {"d": d}
        for split, ids in splits.items():
            n, s, s2 = write_store(field, split, ids, proj_of, cfg, d)
            info[f"n_{split}"] = n
            if split == "train":
                mean = s / n
                info["train_mean"] = mean.tolist()
                info["train_std"] = np.sqrt(np.maximum(s2 / n - mean**2, 0)).tolist()
            logger.info("%s %s: %d rows x %d", field, split, n, d)
        meta["fields"][field] = info
    (cfg.output_dir / "metadata.json").write_text(json.dumps(meta) + "\n")
    return cfg.output_dir


if __name__ == "__main__":
    run(cli_config(StoreConfig, __doc__))
