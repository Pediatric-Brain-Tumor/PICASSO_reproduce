"""Virchow2 tile embeddings for tiled slides.

For every slide with ``tiles.parquet`` the tiles are read at ``tile_px`` @ ``target_mpp``,
resized to the model input (224), and encoded with Virchow2 (ViT-H/14, 4 registers).
Saved per slide, fp16, row-aligned with ``tiles.parquet``:
  ``<slide_id>.cls.npy``        [n, 1280]  class token            (PICASSO input, d=1280)
  ``<slide_id>.meanpatch.npy``  [n, 1280]  mean of patch tokens   (optional, for comparison)
plus ``<slide_id>.done``. Optionally a random ``keep_jpeg_frac`` of tiles is saved as JPEG
(``jpeg_dir/<slide_id>/<x>_<y>.jpg``) for later exemplar inspection.

Several GPUs: run one process per GPU with ``--set shard=i num_shards=n device=cuda:i``.

    python -m picasso.embed --config configs/data/embed.yaml
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import time

import numpy as np
import polars as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from picasso.config import cli_config, dump_json

logger = logging.getLogger("picasso.embed")

VIRCHOW2 = "hf-hub:paige-ai/Virchow2"


@dataclasses.dataclass(frozen=True)
class EmbedConfig:
    tiles_dir: pathlib.Path
    output_dir: pathlib.Path
    projects: tuple[str, ...] | None = None
    model: str = VIRCHOW2
    input_px: int = 224
    tile_px: int = 256
    save_mean_patch: bool = True
    keep_jpeg_frac: float = 0.0
    jpeg_dir: pathlib.Path | None = None
    jpeg_quality: int = 90
    batch_size: int = 256
    num_workers: int = 12
    shard: int = 0
    num_shards: int = 1
    device: str = "cuda"
    dtype: str = "float16"
    seed: int = 0


def build_model(name: str, device: str):
    import timm
    from timm.layers import SwiGLUPacked

    model = timm.create_model(name, pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model = model.eval().requires_grad_(False).to(device)
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(device)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(device)
    n_reg = int(getattr(model, "num_prefix_tokens", 5)) - 1
    return model, mean, std, n_reg


class SlideTiles(Dataset):
    """Reads the tiles of one slide; the OpenSlide handle is opened lazily per worker."""

    def __init__(self, slide_path: str, tiles: pl.DataFrame, tile_px: int, input_px: int):
        self.path = slide_path
        self.x = tiles["x"].to_numpy()
        self.y = tiles["y"].to_numpy()
        self.tile_l0 = int(tiles["tile_l0"][0])
        self.tile_px = tile_px
        self.input_px = input_px
        self._slide = None
        self._level = None

    def __len__(self) -> int:
        return len(self.x)

    def _open(self):
        import openslide

        if self._slide is None:
            self._slide = openslide.OpenSlide(self.path)
            self._level = self._slide.get_best_level_for_downsample(self.tile_l0 / self.tile_px)
            self._ds = self._slide.level_downsamples[self._level]
        return self._slide

    def __getitem__(self, i: int):
        slide = self._open()
        size = int(round(self.tile_l0 / self._ds))
        region = slide.read_region((int(self.x[i]), int(self.y[i])), self._level, (size, size)).convert("RGB")
        tile = region if size == self.tile_px else region.resize((self.tile_px, self.tile_px), Image.LANCZOS)
        inp = tile if self.tile_px == self.input_px else tile.resize((self.input_px, self.input_px), Image.BICUBIC)
        arr = torch.from_numpy(np.asarray(inp, dtype=np.uint8).copy()).permute(2, 0, 1)  # uint8 CHW
        raw = torch.from_numpy(np.asarray(tile, dtype=np.uint8).copy()) if self.tile_px != self.input_px else arr.permute(1, 2, 0)
        return arr, raw, i


def pending_slides(cfg: EmbedConfig) -> list[tuple[str, pathlib.Path]]:
    summary = pl.read_parquet(cfg.tiles_dir / "slides_summary.parquet")
    summary = summary.filter(pl.col("n_tiles").is_not_null()) if "n_tiles" in summary.columns else summary
    if cfg.projects:
        summary = summary.filter(pl.col("project").is_in(list(cfg.projects)))
    rows = sorted(zip(summary["slide_id"], summary["path"]))
    rows = rows[cfg.shard :: cfg.num_shards]
    return [(sid, pathlib.Path(p)) for sid, p in rows if not (cfg.output_dir / f"{sid}.done").exists()]


@torch.inference_mode()
def embed_slide(sid: str, path: pathlib.Path, cfg: EmbedConfig, model, mean, std, n_reg, dtype) -> dict:
    tiles = pl.read_parquet(cfg.tiles_dir / sid / "tiles.parquet")
    n = tiles.height
    cls_out = np.empty((n, model.embed_dim), dtype=np.float16)
    mp_out = np.empty((n, model.embed_dim), dtype=np.float16) if cfg.save_mean_patch else None
    keep = None
    if cfg.keep_jpeg_frac > 0:
        rng = np.random.default_rng(abs(hash(sid)) % (2**32) ^ cfg.seed)
        keep = rng.random(n) < cfg.keep_jpeg_frac
        (cfg.jpeg_dir / sid).mkdir(parents=True, exist_ok=True)
    ds = SlideTiles(str(path), tiles, cfg.tile_px, cfg.input_px)
    loader = DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    t0 = time.time()
    for pixels, raw, idx in loader:
        x = pixels.to(cfg.device, non_blocking=True).to(dtype).div_(255).sub_(mean.to(dtype)).div_(std.to(dtype))
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=cfg.device.startswith("cuda")):
            out = model(x)                     # [b, 1 + n_reg + 256, 1280]
        cls = out[:, 0].float()
        idx_np = idx.numpy()
        cls_out[idx_np] = cls.cpu().numpy().astype(np.float16)
        if mp_out is not None:
            mp_out[idx_np] = out[:, 1 + n_reg :].float().mean(1).cpu().numpy().astype(np.float16)
        if keep is not None:
            for j, i in enumerate(idx_np):
                if keep[i]:
                    Image.fromarray(raw[j].numpy()).save(
                        cfg.jpeg_dir / sid / f"{int(ds.x[i])}_{int(ds.y[i])}.jpg", quality=cfg.jpeg_quality
                    )
    np.save(cfg.output_dir / f"{sid}.cls.npy", cls_out)
    if mp_out is not None:
        np.save(cfg.output_dir / f"{sid}.meanpatch.npy", mp_out)
    (cfg.output_dir / f"{sid}.done").write_text("ok\n")
    dt = time.time() - t0
    return {"slide_id": sid, "n_tiles": n, "seconds": dt, "tiles_per_s": n / max(dt, 1e-6)}


def run(cfg: EmbedConfig) -> pathlib.Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(cfg.output_dir / f"run_config_shard{cfg.shard}.json", cfg)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dtype = getattr(torch, cfg.dtype)
    model, mean, std, n_reg = build_model(cfg.model, cfg.device)
    logger.info("model %s embed_dim=%d registers=%d", cfg.model, model.embed_dim, n_reg)
    todo = pending_slides(cfg)
    logger.info("shard %d/%d: %d slides pending", cfg.shard, cfg.num_shards, len(todo))
    for k, (sid, path) in enumerate(todo):
        try:
            info = embed_slide(sid, path, cfg, model, mean, std, n_reg, dtype)
            logger.info("[%d/%d] %s tiles=%d %.0f tiles/s", k + 1, len(todo), sid, info["n_tiles"], info["tiles_per_s"])
        except Exception as exc:
            logger.error("FAILED %s: %r", sid, exc)
    return cfg.output_dir


if __name__ == "__main__":
    run(cli_config(EmbedConfig, __doc__))
