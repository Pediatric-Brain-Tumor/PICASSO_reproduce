"""Check 1: does Emb2Img generate faithful tiles from SAE-reconstructed embeddings?

For n test tiles: generate from the original CLS and from SAE(decode(encode(CLS))).
Reports cos(gen, original CLS) for both, cos(recon, original), and writes a 3-row grid
(real / gen from original / gen from SAE recon).
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
p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=128); p.add_argument("--guidance", type=float, default=4.5)
p.add_argument("--steps", type=int, default=50); p.add_argument("--bs", type=int, default=32)
a = p.parse_args()
dev = "cuda"
cfg = SampleConfig(checkpoint=pathlib.Path(a.emb2img), pairs_dir=pathlib.Path(a.pairs), output_dir=pathlib.Path(a.out),
                   n=a.n, batch_size=a.bs, num_steps=a.steps, guidance=a.guidance)
model = Emb2Img.load(cfg.checkpoint, dev)
vae = AutoencoderKL.from_pretrained(cfg.vae, torch_dtype=torch.float16).to(dev).eval()
sched = PNDMScheduler.from_pretrained(cfg.sd_model, subfolder="scheduler")
vit, mean, std, _ = build_model(cfg.virchow, dev)
sae = load_sae(pathlib.Path(a.sae), dev)
emb = torch.from_numpy(np.load(cfg.pairs_dir / "test_emb.npy")[: a.n]).to(dev).float()
tiles = np.load(cfg.pairs_dir / "test_tiles.npy", mmap_mode="r")[: a.n]
with torch.inference_mode():
    f, recon = sae(emb)
print(f"SAE on test: cos(recon, orig) = {F.cosine_similarity(recon, emb).mean():.4f}, EV = {1 - ((recon-emb).var(0).sum()/emb.var(0).sum()):.4f}, L0 = {(f>0).sum(1).float().mean():.1f}")
res = {}
gens = {}
for name, cond in [("orig", emb), ("recon", recon)]:
    g = torch.Generator(device=dev).manual_seed(0)
    imgs, cos_o, cos_c = [], [], []
    for s in range(0, a.n, a.bs):
        img = generate(model, vae, sched, cond[s:s+a.bs], cfg, g)
        c = virchow_cls(vit, mean, std, img)
        cos_o.append(F.cosine_similarity(c, emb[s:s+a.bs])); cos_c.append(F.cosine_similarity(c, cond[s:s+a.bs]))
        imgs.append((img.permute(0,2,3,1).cpu().numpy()*255).round().astype(np.uint8))
    gens[name] = np.concatenate(imgs)
    res[name] = {"cos_gen_vs_original": torch.cat(cos_o).mean().item(), "cos_gen_vs_condition": torch.cat(cos_c).mean().item()}
    print(f"gen from {name:5s}: cos(gen, original CLS) = {res[name]['cos_gen_vs_original']:.4f}   cos(gen, its condition) = {res[name]['cos_gen_vs_condition']:.4f}")
# agreement between the two generations in embedding space
with torch.inference_mode():
    co = torch.cat([virchow_cls(vit, mean, std, torch.from_numpy(gens["orig"][s:s+a.bs]).to(dev).permute(0,3,1,2).float()/255) for s in range(0,a.n,a.bs)])
    cr = torch.cat([virchow_cls(vit, mean, std, torch.from_numpy(gens["recon"][s:s+a.bs]).to(dev).permute(0,3,1,2).float()/255) for s in range(0,a.n,a.bs)])
res["cos_gen_orig_vs_gen_recon"] = F.cosine_similarity(co, cr).mean().item()
print(f"cos(gen from orig, gen from recon) = {res['cos_gen_orig_vs_gen_recon']:.4f}")
os.makedirs(a.out, exist_ok=True)
k = 12
grid = np.concatenate([np.concatenate(list(np.asarray(tiles[:k])), 1), np.concatenate(list(gens["orig"][:k]), 1), np.concatenate(list(gens["recon"][:k]), 1)], 0)
Image.fromarray(grid).save(pathlib.Path(a.out) / "grid_real_orig_recon.jpg", quality=90)
json.dump(res, open(pathlib.Path(a.out) / "summary.json", "w"), indent=1)
