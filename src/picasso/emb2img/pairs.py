"""Build (VAE latent, Virchow2 CLS) pairs for Emb2Img from tiled slides.

``n_train`` random tiles of the train split (``-1`` = all) and ``n_test`` of the test split
(slide-level split from the SAE store). Tiles are read at ``resolution`` px from their slide
and VAE-encoded (latents scaled by ``vae.config.scaling_factor``). Shards are split by
slide so every process reads whole slides sequentially. Outputs in ``output_dir``:
  ``{split}_latents.npy`` [n, 4, res/8, res/8] fp16    ``{split}_emb.npy`` [n, 1280] fp16
  ``{split}_index.parquet`` slide_id, x, y             ``test_tiles.npy`` [n_test, res, res, 3] uint8

    python -m picasso.emb2img.pairs --config configs/emb2img/pairs.yaml --set shard=0 num_shards=8
    python -m picasso.emb2img.pairs --config configs/emb2img/pairs.yaml --set merge=true
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import pathlib

import numpy as np
import polars as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from picasso.config import cli_config, dump_json

logger = logging.getLogger("picasso.emb2img.pairs")


@dataclasses.dataclass(frozen=True)
class PairsConfig:
    store_dir: pathlib.Path
    tiles_dir: pathlib.Path
    embeddings_dir: pathlib.Path
    output_dir: pathlib.Path
    vae: str = "stabilityai/sd-vae-ft-mse"
    resolution: int = 256
    tile_px: int = 256
    n_train: int = 500_000
    n_test: int = 512
    batch_size: int = 128
    num_workers: int = 8
    shard: int = 0
    num_shards: int = 1
    merge: bool = False
    device: str = "cuda"
    seed: int = 0


class TileItems(Dataset):
    """Tiles from many slides; a small LRU of OpenSlide handles per worker."""

    def __init__(self, items: list[tuple[str, int, int, int]], tile_px: int, resolution: int, cache: int = 4):
        self.items = items
        self.tile_px, self.res, self.cache = tile_px, resolution, cache
        self._slides: collections.OrderedDict = collections.OrderedDict()

    def __len__(self):
        return len(self.items)

    def _get(self, path: str):
        import openslide

        if path in self._slides:
            self._slides.move_to_end(path)
            return self._slides[path]
        s = openslide.OpenSlide(path)
        self._slides[path] = s
        if len(self._slides) > self.cache:
            _, old = self._slides.popitem(last=False)
            old.close()
        return s

    def __getitem__(self, i):
        path, x, y, tile_l0 = self.items[i]
        s = self._get(path)
        level = s.get_best_level_for_downsample(tile_l0 / self.tile_px)
        ds = s.level_downsamples[level]
        size = int(round(tile_l0 / ds))
        im = s.read_region((x, y), level, (size, size)).convert("RGB")
        if size != self.res:
            im = im.resize((self.res, self.res), Image.LANCZOS)
        return torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()), i


def select_rows(cfg: PairsConfig, split: str, n: int, rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    """Per slide (this shard only): sorted local tile indices chosen for the pair set."""
    idx = pl.read_parquet(cfg.store_dir / f"cls_{split}.index.parquet")
    total = int(idx["n"].sum())
    rows = np.arange(total) if n < 0 or n >= total else np.sort(rng.choice(total, size=n, replace=False))
    starts = idx["start"].to_numpy()
    slide_pos = np.searchsorted(starts, rows, side="right") - 1      # sorted, because rows are sorted
    local = rows - starts[slide_pos]
    uniq, first = np.unique(slide_pos, return_index=True)
    bounds = np.append(first, len(rows))
    ids = idx["slide_id"].to_list()
    return [(ids[int(sp)], local[bounds[k] : bounds[k + 1]])
            for k, sp in enumerate(uniq) if int(sp) % cfg.num_shards == cfg.shard]


@torch.inference_mode()
def encode(cfg: PairsConfig, split: str, chosen: list[tuple[str, np.ndarray]], vae, keep_tiles: bool) -> None:
    summary = pl.read_parquet(cfg.tiles_dir / "slides_summary.parquet")
    path_of = dict(zip(summary["slide_id"], summary["path"]))
    items, embs, index = [], [], []
    for sid, loc in chosen:
        tiles = pl.read_parquet(cfg.tiles_dir / sid / "tiles.parquet")
        sub = tiles[loc.tolist()]
        p = path_of[sid]
        items += [(p, int(x), int(y), int(t)) for x, y, t in zip(sub["x"], sub["y"], sub["tile_l0"])]
        embs.append(np.asarray(np.load(cfg.embeddings_dir / f"{sid}.cls.npy", mmap_mode="r")[loc], dtype=np.float16))
        index.append(sub.select("slide_id", "x", "y"))
    n = len(items)
    lat_hw = cfg.resolution // 8
    latents = np.empty((n, 4, lat_hw, lat_hw), dtype=np.float16)
    tiles_u8 = np.empty((n, cfg.resolution, cfg.resolution, 3), dtype=np.uint8) if keep_tiles else None
    loader = DataLoader(TileItems(items, cfg.tile_px, cfg.resolution), batch_size=cfg.batch_size,
                        num_workers=cfg.num_workers, pin_memory=True)
    done = 0
    for px, i in loader:
        x = px.to(cfg.device).permute(0, 3, 1, 2).float().div(127.5).sub(1)
        with torch.autocast("cuda", dtype=torch.float16):
            z = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
        latents[i.numpy()] = z.float().cpu().numpy().astype(np.float16)
        if tiles_u8 is not None:
            tiles_u8[i.numpy()] = px.numpy()
        done += len(i)
        if done % (cfg.batch_size * 200) < cfg.batch_size:
            logger.info("%s shard %d: %d / %d", split, cfg.shard, done, n)
    tag = f"{split}_shard{cfg.shard}"
    np.save(cfg.output_dir / f"{tag}_latents.npy", latents)
    np.save(cfg.output_dir / f"{tag}_emb.npy", np.concatenate(embs) if embs else np.empty((0, 1280), np.float16))
    (pl.concat(index) if index else pl.DataFrame({"slide_id": [], "x": [], "y": []})).write_parquet(cfg.output_dir / f"{tag}_index.parquet")
    if tiles_u8 is not None:
        np.save(cfg.output_dir / f"{tag}_tiles.npy", tiles_u8)
    logger.info("wrote %s: %d pairs from %d slides", tag, n, len(chosen))


def merge_shards(cfg: PairsConfig) -> None:
    for split in ("train", "test"):
        parts = sorted(cfg.output_dir.glob(f"{split}_shard*_index.parquet"))
        if not parts:
            continue
        for kind in ("latents", "emb", "tiles"):
            files = sorted(cfg.output_dir.glob(f"{split}_shard*_{kind}.npy"))
            if not files:
                continue
            arrays = [np.load(f, mmap_mode="r") for f in files]
            out = np.lib.format.open_memmap(cfg.output_dir / f"{split}_{kind}.npy", mode="w+", dtype=arrays[0].dtype,
                                            shape=(sum(len(a) for a in arrays), *arrays[0].shape[1:]))
            pos = 0
            for a in arrays:
                out[pos : pos + len(a)] = a
                pos += len(a)
            out.flush(); del out
        pl.concat([pl.read_parquet(p) for p in parts]).write_parquet(cfg.output_dir / f"{split}_index.parquet")
        logger.info("merged %s: %d shards, %d rows", split, len(parts), pl.read_parquet(cfg.output_dir / f"{split}_index.parquet").height)
        for f in list(cfg.output_dir.glob(f"{split}_shard*")):
            f.unlink()


def run(cfg: PairsConfig) -> pathlib.Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.merge:
        merge_shards(cfg)
        return cfg.output_dir
    dump_json(cfg.output_dir / f"run_config_shard{cfg.shard}.json", cfg)
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(cfg.vae).to(cfg.device).eval().requires_grad_(False)
    rng = np.random.default_rng(cfg.seed)      # identical draw in every shard; shards keep disjoint slides
    for split, n, keep in (("test", cfg.n_test, True), ("train", cfg.n_train, False)):
        encode(cfg, split, select_rows(cfg, split, n, rng), vae, keep)
    return cfg.output_dir


if __name__ == "__main__":
    run(cli_config(PairsConfig, __doc__))
