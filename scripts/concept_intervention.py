"""Check 2: does dialing up one SAE concept change morphology accordingly?

Takes test tiles on which the concept is inactive, adds s * W_dec[j] to their CLS for a
ladder of strengths s (same initial noise for every level), generates with Emb2Img, and
measures (a) a pixel proxy, (b) the concept's own activation when the generated tile is
re-embedded with Virchow2 and re-encoded by the SAE, (c) cosine to the original CLS.
Writes grid.jpg (rows = tiles, columns = strengths) and summary.json.
"""
import argparse, json, os, pathlib
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, PNDMScheduler
from picasso.embed import build_model
from picasso.emb2img.model import Emb2Img
from picasso.emb2img.sample_eval import SampleConfig, generate, virchow_cls
from picasso.model import load as load_sae

p = argparse.ArgumentParser()
p.add_argument("--emb2img", required=True); p.add_argument("--sae", required=True); p.add_argument("--pairs", required=True)
p.add_argument("--out", required=True); p.add_argument("--latent", type=int, required=True)
p.add_argument("--levels", default="0,5,10,20,30"); p.add_argument("--n_tiles", type=int, default=8)
p.add_argument("--guidance", type=float, default=4.5); p.add_argument("--steps", type=int, default=100); p.add_argument("--seed", type=int, default=0)
a = p.parse_args(); dev = "cuda"; j = a.latent; levels = [float(x) for x in a.levels.split(",")]
cfg = SampleConfig(checkpoint=pathlib.Path(a.emb2img), pairs_dir=pathlib.Path(a.pairs), output_dir=pathlib.Path(a.out),
                   n=a.n_tiles, batch_size=a.n_tiles, num_steps=a.steps, guidance=a.guidance)
model = Emb2Img.load(cfg.checkpoint, dev)
vae = AutoencoderKL.from_pretrained(cfg.vae, torch_dtype=torch.float16).to(dev).eval()
sched = PNDMScheduler.from_pretrained(cfg.sd_model, subfolder="scheduler")
vit, mean, std, _ = build_model(cfg.virchow, dev)
sae = load_sae(pathlib.Path(a.sae), dev)
emb_all = torch.from_numpy(np.load(cfg.pairs_dir / "test_emb.npy")).to(dev).float()
tiles_all = np.load(cfg.pairs_dir / "test_tiles.npy", mmap_mode="r")
with torch.inference_mode():
    acts = sae.encode(emb_all)
inactive = torch.nonzero(acts[:, j] == 0).squeeze(1).cpu().numpy()
rng = np.random.default_rng(a.seed); pick = np.sort(rng.choice(inactive, a.n_tiles, replace=False))
emb = emb_all[pick]; tiles = np.asarray(tiles_all[pick])
direction = sae.W_dec.data[j]                       # unit-norm decoder row
print(f"latent {j}: active on {(acts[:, j] > 0).float().mean():.3%} of test tiles; picked tiles {pick.tolist()}")
def proxy(im):
    x = im.astype(np.float32); gray = x.mean(-1); return ((gray < 110) & (x[..., 2] > x[..., 0])).mean(axis=(1, 2))
rows = {int(t): [] for t in pick}; summary = []
for s in levels:
    cond = emb + s * direction
    g = torch.Generator(device=dev).manual_seed(a.seed)
    img = generate(model, vae, sched, cond, cfg, g)
    u8 = (img.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
    with torch.inference_mode():
        c = virchow_cls(vit, mean, std, img); re_act = sae.encode(c)[:, j]
    rec = {"strength": s, "proxy_dark_nuclei": float(proxy(u8).mean()), "reencoded_activation_mean": float(re_act.mean()),
           "reencoded_active_frac": float((re_act > 0).float().mean()), "cos_to_original_cls": float(F.cosine_similarity(c, emb).mean())}
    summary.append(rec); print(rec)
    for k, t in enumerate(pick): rows[int(t)].append(u8[k])
os.makedirs(a.out, exist_ok=True)
grid = np.concatenate([np.concatenate([tiles[k]] + rows[int(t)], 1) for k, t in enumerate(pick)], 0)
Image.fromarray(grid).save(pathlib.Path(a.out) / "grid.jpg", quality=88)
json.dump({"latent": j, "levels": levels, "tiles": pick.tolist(), "per_level": summary}, open(pathlib.Path(a.out) / "summary.json", "w"), indent=1)
