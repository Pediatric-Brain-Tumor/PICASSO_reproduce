"""Tissue segmentation and tile grid for one or more whole-slide images (CLAM-like).

For every slide: a low-resolution tissue mask (HSV saturation, median filter,
threshold, morphological closing, small-component removal) and a non-overlapping
grid of tiles that are ``tile_px`` pixels at ``target_mpp`` microns/pixel, kept when
the tissue fraction under the tile is at least ``min_tissue_frac``.

Outputs per slide in ``output_dir/<slide_id>/``: ``tiles.parquet`` (x, y in level-0
pixels, tile_l0 = tile side in level-0 pixels, tissue_frac), ``slide.json``,
``mask.png`` (QA overlay).

    python -m picasso.tiling --config configs/data/tiling.yaml
"""

from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import multiprocessing as mp
import json
import logging
import pathlib

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from scipy import ndimage

from picasso.config import cli_config, dump_json

logger = logging.getLogger("picasso.tiling")


@dataclasses.dataclass(frozen=True)
class TilingConfig:
    slides_dir: pathlib.Path
    manifest: pathlib.Path
    output_dir: pathlib.Path
    projects: tuple[str, ...] | None = None   # restrict to these TCGA projects
    target_mpp: float = 0.5
    tile_px: int = 256
    mask_downsample: int = 64
    sat_thresh: int = 8        # CLAM default sthresh; also the floor when use_otsu
    use_otsu: bool = True      # Otsu on the filtered saturation (handles tinted backgrounds)
    median_size: int = 7       # CLAM default mthresh
    close_px: int = 4          # CLAM default close
    min_component_tiles: float = 4.0   # drop tissue blobs smaller than this many tiles
    min_tissue_frac: float = 0.5
    workers: int = 16


def slide_mpp(slide) -> float:
    import openslide

    mpp = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    if mpp:
        return float(mpp)
    mag = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER) or slide.properties.get("aperio.AppMag")
    if mag:
        return 10.0 / float(mag)   # 40x -> 0.25, 20x -> 0.5
    raise ValueError("no mpp metadata")


def tissue_mask(slide, cfg: TilingConfig, tile_l0: int) -> tuple[np.ndarray, float, int]:
    level = slide.get_best_level_for_downsample(cfg.mask_downsample)
    ds = slide.level_downsamples[level]
    thumb = slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB")
    sat = np.asarray(thumb.convert("HSV"))[..., 1]
    sat = ndimage.median_filter(sat, size=cfg.median_size)
    thresh = cfg.sat_thresh
    if cfg.use_otsu:
        import cv2

        otsu, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = max(cfg.sat_thresh, int(otsu))
    mask = sat > thresh
    if cfg.close_px > 1:
        mask = ndimage.binary_closing(mask, structure=np.ones((cfg.close_px, cfg.close_px)))
    labels, n = ndimage.label(mask)
    if n:
        tile_mask_px = tile_l0 / ds
        min_area = cfg.min_component_tiles * tile_mask_px**2
        sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, dtype=bool)
        keep[1:] = sizes >= min_area
        mask = keep[labels]
    return mask, ds, thresh


def tile_grid(mask: np.ndarray, ds: float, dims: tuple[int, int], tile_l0: int, min_frac: float) -> dict[str, np.ndarray]:
    """Tiles on a level-0 grid whose tissue fraction (from the mask) is >= ``min_frac``."""
    W, H = dims
    xs = np.arange(0, W - tile_l0 + 1, tile_l0)
    ys = np.arange(0, H - tile_l0 + 1, tile_l0)
    integral = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    mh, mw = mask.shape

    def box_sum(x0, y0, x1, y1):
        return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]

    gx0 = np.clip(np.round(xs / ds).astype(int), 0, mw)
    gx1 = np.clip(np.round((xs + tile_l0) / ds).astype(int), 0, mw)
    gy0 = np.clip(np.round(ys / ds).astype(int), 0, mh)
    gy1 = np.clip(np.round((ys + tile_l0) / ds).astype(int), 0, mh)
    X0, Y0 = np.meshgrid(gx0, gy0)
    X1, Y1 = np.meshgrid(gx1, gy1)
    area = np.maximum((X1 - X0) * (Y1 - Y0), 1)
    frac = box_sum(X0, Y0, X1, Y1) / area
    yy, xx = np.nonzero(frac >= min_frac)
    return {
        "x": xs[xx].astype(np.int32),
        "y": ys[yy].astype(np.int32),
        "tile_l0": np.full(len(xx), tile_l0, dtype=np.int32),
        "tissue_frac": frac[yy, xx].astype(np.float32),
    }


def process_slide(path: pathlib.Path, project: str, cfg: TilingConfig) -> dict:
    import openslide

    slide_id = path.name.removesuffix(".svs")
    out = cfg.output_dir / slide_id
    if (out / "tiles.parquet").exists():
        return json.loads((out / "slide.json").read_text())
    slide = openslide.OpenSlide(str(path))
    mpp = slide_mpp(slide)
    tile_l0 = int(round(cfg.tile_px * cfg.target_mpp / mpp))
    mask, ds, sat_used = tissue_mask(slide, cfg, tile_l0)
    tiles = tile_grid(mask, ds, slide.dimensions, tile_l0, cfg.min_tissue_frac)
    out.mkdir(parents=True, exist_ok=True)
    n_tiles = len(tiles["x"])
    table = pa.table({"slide_id": pa.array([slide_id] * n_tiles, pa.string()), **{k: pa.array(v) for k, v in tiles.items()}})
    pq.write_table(table, out / "tiles.parquet")
    # QA overlay: tissue mask in green, kept tiles outlined in red.
    level = slide.get_best_level_for_downsample(cfg.mask_downsample)
    thumb = np.asarray(slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB")).copy()
    thumb[mask] = (0.6 * thumb[mask] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    t = tile_l0 / ds
    for x, y in zip(tiles["x"], tiles["y"]):
        x0, y0 = int(x / ds), int(y / ds)
        x1, y1 = min(int(x0 + t), thumb.shape[1] - 1), min(int(y0 + t), thumb.shape[0] - 1)
        thumb[y0, x0:x1] = thumb[y1, x0:x1] = thumb[y0:y1, x0] = thumb[y0:y1, x1] = (255, 0, 0)
    Image.fromarray(thumb).save(out / "mask.png")
    info = {
        "slide_id": slide_id,
        "project": project,
        "path": str(path),
        "mpp": mpp,
        "dims": list(slide.dimensions),
        "tile_l0": tile_l0,
        "mask_downsample": ds,
        "sat_thresh_used": sat_used,
        "n_tiles": n_tiles,
        "tissue_area_mm2": float(mask.sum() * (mpp * ds / 1000) ** 2),
    }
    slide.close()
    dump_json(out / "slide.json", info)
    return info


def ready_slides(cfg: TilingConfig) -> list[tuple[pathlib.Path, str]]:
    """Slides from the manifest that are fully downloaded (``.ok`` marker present)."""
    man = pl.read_csv(cfg.manifest)
    if cfg.projects:
        man = man.filter(pl.col("project").is_in(list(cfg.projects)))
    found = []
    for proj, name in zip(man["project"], man["file_name"]):
        p = cfg.slides_dir / proj / name
        if pathlib.Path(str(p) + ".ok").exists():
            found.append((p, proj))
    return found


def run(cfg: TilingConfig) -> pathlib.Path:
    slides = ready_slides(cfg)
    logger.info("%d downloaded slides to tile", len(slides))
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    infos = []
    with cf.ProcessPoolExecutor(cfg.workers, mp_context=mp.get_context("spawn")) as pool:
        futs = {pool.submit(process_slide, p, proj, cfg): p for p, proj in slides}
        for fut in cf.as_completed(futs):
            try:
                info = fut.result()
                infos.append(info)
                logger.info("%s %s mpp=%.3f tiles=%d", info["project"], info["slide_id"], info["mpp"], info["n_tiles"])
            except Exception as exc:  # keep going; record failures
                logger.error("FAILED %s: %r", futs[fut].name, exc)
                infos.append({"slide_id": futs[fut].name.removesuffix(".svs"), "error": repr(exc)})
    pl.DataFrame(infos).write_parquet(cfg.output_dir / "slides_summary.parquet")
    ok = [i for i in infos if "n_tiles" in i]
    if ok:
        n = np.array([i["n_tiles"] for i in ok])
        logger.info("tiles/slide mean=%.0f median=%.0f total=%d over %d slides", n.mean(), np.median(n), n.sum(), len(ok))
    return cfg.output_dir


if __name__ == "__main__":
    run(cli_config(TilingConfig, __doc__))
