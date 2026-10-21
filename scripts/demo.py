"""Concept intervention on one tile with a trained SAE and Emb2Img.

tile (256 px at 0.5 um/px) -> Virchow2 CLS -> add s * W_dec[latent] -> Emb2Img -> image strip.
The first panel is the input tile, then one generated tile per strength (same noise seed).

    python scripts/demo.py --image tile.png --sae <sae.pt or dir> --emb2img <ema dir> --latent <concept id>
"""
import argparse, pathlib

import numpy as np
import torch
from PIL import Image

from picasso.embed import build_model
from picasso.emb2img.model import Emb2Img
from picasso.emb2img.sample_eval import SampleConfig, generate, virchow_cls
from picasso.model import load as load_sae

p = argparse.ArgumentParser()
p.add_argument("--image", required=True)
p.add_argument("--sae", required=True)
p.add_argument("--emb2img", required=True)
p.add_argument("--latent", type=int, required=True)
p.add_argument("--levels", default="0,10,20,30")
p.add_argument("--guidance", type=float, default=4.5)
p.add_argument("--steps", type=int, default=50)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="demo.jpg")
a = p.parse_args()
dev = "cuda"
from diffusers import AutoencoderKL, PNDMScheduler

cfg = SampleConfig(checkpoint=pathlib.Path(a.emb2img), pairs_dir=pathlib.Path("."), output_dir=pathlib.Path("."),
                   num_steps=a.steps, guidance=a.guidance)
vit, mean, std, _ = build_model(cfg.virchow, dev)
sae = load_sae(pathlib.Path(a.sae), dev)
model = Emb2Img.load(cfg.checkpoint, dev)
vae = AutoencoderKL.from_pretrained(cfg.vae, torch_dtype=torch.float16).to(dev).eval()
sched = PNDMScheduler.from_pretrained(cfg.sd_model, subfolder="scheduler")

tile = Image.open(a.image).convert("RGB").resize((256, 256), Image.LANCZOS)
x = torch.from_numpy(np.asarray(tile)).permute(2, 0, 1)[None].float().div(255).to(dev)
cls = virchow_cls(vit, mean, std, x)
with torch.inference_mode():
    f = sae.encode(cls)
top = f[0].topk(5)
print("top concepts of this tile:", [(int(i), round(float(v), 2)) for v, i in zip(top.values, top.indices)])
print(f"latent {a.latent} activation on this tile: {float(f[0, a.latent]):.2f}")
panels = [np.asarray(tile)]
for s in [float(v) for v in a.levels.split(",")]:
    cond = cls + s * sae.W_dec.data[a.latent]
    g = torch.Generator(device=dev).manual_seed(a.seed)
    img = generate(model, vae, sched, cond, cfg, g)
    panels.append((img[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
Image.fromarray(np.concatenate(panels, 1)).save(a.out, quality=90)
print("wrote", a.out)
