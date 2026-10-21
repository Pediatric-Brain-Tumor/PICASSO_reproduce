"""Find candidate SAE concepts for a pixel-level proxy (e.g. lymphocyte density) and dump exemplar grids.

Decodes a sample of train latents to pixels (VAE), computes SAE activations of their CLS,
and ranks latents by correlation between activation and the proxy. For the top candidates
writes a grid of the 12 strongest-activating tiles.
"""
import argparse, json, os, pathlib
import numpy as np, torch
from PIL import Image
from diffusers import AutoencoderKL
from picasso.model import load as load_sae

p = argparse.ArgumentParser()
p.add_argument("--sae", required=True); p.add_argument("--pairs", required=True); p.add_argument("--out", required=True)
p.add_argument("--n", type=int, default=8000); p.add_argument("--proxy", default="dark_nuclei"); p.add_argument("--top", type=int, default=6)
a = p.parse_args(); dev = "cuda"
sae = load_sae(pathlib.Path(a.sae), dev)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to(dev).eval()
lat = np.load(pathlib.Path(a.pairs) / "train_latents.npy", mmap_mode="r"); emb = np.load(pathlib.Path(a.pairs) / "train_emb.npy", mmap_mode="r")
rng = np.random.default_rng(1); idx = np.sort(rng.choice(len(lat), a.n, replace=False))
E = torch.from_numpy(np.ascontiguousarray(emb[idx])).to(dev).float()
with torch.inference_mode():
    acts = torch.cat([sae.encode(E[s:s+4096]) for s in range(0, a.n, 4096)]).cpu().numpy()   # [n, 40960]
imgs = np.empty((a.n, 256, 256, 3), np.uint8)
with torch.inference_mode():
    for s in range(0, a.n, 32):
        L = torch.from_numpy(np.ascontiguousarray(lat[idx[s:s+32]])).to(dev).half() / vae.config.scaling_factor
        im = (vae.decode(L).sample.float().clamp(-1, 1) + 1) / 2
        imgs[s:s+32] = (im.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
def proxies(im):
    x = im.astype(np.float32); r, g, b = x[..., 0], x[..., 1], x[..., 2]
    gray = x.mean(-1)
    return {
        "dark_nuclei": ((gray < 110) & (b > r)).mean(axis=(1, 2)),        # dark blue/purple pixels: nuclei-dense
        "white_space": (gray > 220).mean(axis=(1, 2)),                      # lumen / fat / background
        "pink_stroma": ((r > 150) & (r - b > 30) & (gray < 215)).mean(axis=(1, 2)),
    }
P = proxies(imgs); y = P[a.proxy]
freq = (acts > 0).mean(0); ok = freq > 0.002
A = acts[:, ok]; cols = np.nonzero(ok)[0]
Ac = (A - A.mean(0)) / (A.std(0) + 1e-6); yc = (y - y.mean()) / (y.std() + 1e-6)
corr = (Ac * yc[:, None]).mean(0)
order = np.argsort(-corr)[: a.top]
os.makedirs(a.out, exist_ok=True); report = []
for r, j in enumerate(order):
    lat_id = int(cols[j]); act = acts[:, lat_id]; top = np.argsort(-act)[:12]
    Image.fromarray(np.concatenate(list(imgs[top]), 1)).save(pathlib.Path(a.out) / f"cand{r}_latent{lat_id}.jpg", quality=85)
    nz = act[act > 0]
    report.append({"rank": r, "latent": lat_id, "corr": float(corr[j]), "freq": float(freq[lat_id]), "act_p50": float(np.median(nz)), "act_p90": float(np.percentile(nz, 90)), "act_max": float(nz.max()), "proxy_top12_mean": float(y[top].mean()), "proxy_overall_mean": float(y.mean())})
    print(report[-1])
json.dump(report, open(pathlib.Path(a.out) / f"candidates_{a.proxy}.json", "w"), indent=1)
np.save(pathlib.Path(a.out) / "sample_idx.npy", idx)
