"""Generate tiles from held-out embeddings and measure the embedding closed loop.

For ``n`` test pairs: sample with PNDM (``num_steps``, CFG ``guidance``), decode with the VAE,
re-embed the generated tile with Virchow2 and report
  cos(gen, cond)       cosine between the generated tile's CLS and the conditioning CLS
  cos(real, cond)      the real tile re-embedded vs. its stored CLS (pipeline sanity, ~1)
  cos(shuffled)        generated vs. a random other conditioning CLS (chance level)
  cos(real nn)         real CLS vs. its nearest other real CLS in the test set (what "same tissue" scores)
plus a side-by-side grid ``grid.jpg`` (top: real tiles, bottom: generated). Writes ``summary.json``.

    python -m picasso.emb2img.sample_eval --config configs/emb2img/sample.yaml
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from picasso.config import cli_config, dump_json
from picasso.emb2img.model import Emb2Img

logger = logging.getLogger("picasso.emb2img.sample_eval")


@dataclasses.dataclass(frozen=True)
class SampleConfig:
    checkpoint: pathlib.Path
    pairs_dir: pathlib.Path
    output_dir: pathlib.Path
    sd_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    vae: str = "stabilityai/sd-vae-ft-mse"
    virchow: str = "hf-hub:paige-ai/Virchow2"
    n: int = 256
    batch_size: int = 32
    num_steps: int = 50
    guidance: float = 4.5
    resolution: int = 256
    grid_n: int = 12
    seed: int = 0
    device: str = "cuda"


@torch.inference_mode()
def generate(model: Emb2Img, vae, scheduler, emb: torch.Tensor, cfg: SampleConfig, gen) -> torch.Tensor:
    b = len(emb)
    lat = torch.randn((b, 4, cfg.resolution // 8, cfg.resolution // 8), generator=gen, device=cfg.device, dtype=torch.float16)
    lat = lat * scheduler.init_noise_sigma
    emb = emb.to(next(model.proj.parameters()).dtype)
    ctx = torch.cat([model.context(emb, torch.ones(b, dtype=torch.bool, device=cfg.device)), model.context(emb)])
    scheduler.set_timesteps(cfg.num_steps, device=cfg.device)
    for t in scheduler.timesteps:
        inp = scheduler.scale_model_input(torch.cat([lat, lat]), t)
        eps = model.unet(inp, t, encoder_hidden_states=ctx).sample
        eps_u, eps_c = eps.chunk(2)
        eps = eps_u + cfg.guidance * (eps_c - eps_u)
        lat = scheduler.step(eps, t, lat).prev_sample
    img = vae.decode(lat / vae.config.scaling_factor).sample
    return (img.float().clamp(-1, 1) + 1) / 2          # [b,3,H,W] in [0,1]


@torch.inference_mode()
def virchow_cls(model, mean, std, img01: torch.Tensor) -> torch.Tensor:
    x = F.interpolate(img01, size=(224, 224), mode="bicubic", align_corners=False)
    x = (x - mean) / std
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(x)
    return out[:, 0].float()


def run(cfg: SampleConfig) -> dict:
    from diffusers import AutoencoderKL, PNDMScheduler

    from picasso.embed import build_model

    torch.manual_seed(cfg.seed)
    model = Emb2Img.load(cfg.checkpoint, cfg.device)
    vae = AutoencoderKL.from_pretrained(cfg.vae, torch_dtype=torch.float16).to(cfg.device).eval()
    scheduler = PNDMScheduler.from_pretrained(cfg.sd_model, subfolder="scheduler")
    vit, mean, std, _ = build_model(cfg.virchow, cfg.device)
    emb_all = torch.from_numpy(np.load(cfg.pairs_dir / "test_emb.npy")[: cfg.n]).to(cfg.device).float()
    tiles = np.load(cfg.pairs_dir / "test_tiles.npy", mmap_mode="r")[: cfg.n]
    gen = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
    gens, cos_gen, cos_real = [], [], []
    for s in range(0, cfg.n, cfg.batch_size):
        e = emb_all[s : s + cfg.batch_size]
        img = generate(model, vae, scheduler, e, cfg, gen)
        gens.append((img.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8))
        cos_gen.append(F.cosine_similarity(virchow_cls(vit, mean, std, img), e))
        real = torch.from_numpy(np.asarray(tiles[s : s + cfg.batch_size])).to(cfg.device).permute(0, 3, 1, 2).float() / 255
        cos_real.append(F.cosine_similarity(virchow_cls(vit, mean, std, real), e))
        logger.info("%d / %d  cos(gen,cond)=%.3f  cos(real,cond)=%.3f", s + len(e), cfg.n, cos_gen[-1].mean(), cos_real[-1].mean())
    gens = np.concatenate(gens)
    cos_gen = torch.cat(cos_gen); cos_real = torch.cat(cos_real)
    en = F.normalize(emb_all, dim=1)
    sim = en @ en.T; sim.fill_diagonal_(-1)
    cos_nn = sim.max(1).values
    perm = torch.randperm(cfg.n, generator=torch.Generator().manual_seed(1)).to(cfg.device)
    gen_emb = torch.cat([virchow_cls(vit, mean, std, torch.from_numpy(gens[s:s+cfg.batch_size]).to(cfg.device).permute(0,3,1,2).float()/255)
                         for s in range(0, cfg.n, cfg.batch_size)])
    cos_shuf = F.cosine_similarity(gen_emb, emb_all[perm])
    summary = {
        "checkpoint": str(cfg.checkpoint), "n": cfg.n, "num_steps": cfg.num_steps, "guidance": cfg.guidance,
        "cos_gen_cond_mean": cos_gen.mean().item(), "cos_gen_cond_median": cos_gen.median().item(),
        "cos_real_cond_mean": cos_real.mean().item(),
        "cos_gen_shuffled_mean": cos_shuf.mean().item(),
        "cos_real_nearest_neighbour_mean": cos_nn.mean().item(),
        "frac_gen_above_nn": (cos_gen > cos_nn).float().mean().item(),
    }
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.output_dir / "generated.npy", gens)
    np.save(cfg.output_dir / "cos_gen_cond.npy", cos_gen.cpu().numpy())
    k = min(cfg.grid_n, cfg.n); r = cfg.resolution
    grid = np.concatenate([np.concatenate(list(np.asarray(tiles[:k])), axis=1), np.concatenate(list(gens[:k]), axis=1)], axis=0)
    Image.fromarray(grid).save(cfg.output_dir / "grid.jpg", quality=90)
    dump_json(cfg.output_dir / "summary.json", summary)
    logger.info("%s", json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    run(cli_config(SampleConfig, __doc__))
