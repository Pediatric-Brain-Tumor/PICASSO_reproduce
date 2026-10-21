"""Train a TopK SAE on tile embeddings (PICASSO recipe), optionally with Matryoshka prefixes.

PICASSO (Kim et al. 2026): Virchow2 embedding (d=1280), expansion 32 (h=40960), k=30, MSE,
Adam lr 1e-4, batch 2048, 5 epochs, SAELens. SAELens' TopK trainer also adds the AuxK
dead-latent loss, kept here (``aux_alpha=0`` disables it). ``n_prefixes=1`` is plain TopK;
``n_prefixes>1`` enables a Matryoshka nested-prefix objective (not used by PICASSO).

    python -m picasso.train --config configs/sae/train.yaml
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import pathlib
import time

import torch

from picasso.config import cli_config, dump_json
from picasso.loader import EmbeddingStore
from picasso.metrics import Accumulator
from picasso.model import SAEConfig, TopKSAE, dump

logger = logging.getLogger("picasso.train")


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    store_dir: pathlib.Path
    output_dir: pathlib.Path
    field: str = "cls"
    expansion: int = 32
    top_k: int = 30
    aux_k: int = 640
    aux_alpha: float = 1 / 32
    n_prefixes: int = 1                  # 1 = plain TopK (PICASSO); >1 = Matryoshka
    dead_threshold_samples: int = 2_000_000
    datapoint_init_blend: float = 0.0    # 0 = Kaiming init (SAELens-like); 0.8 = reference code
    normalize_input: bool = False        # z-score with train mean/std (SAELens normalize_activations)
    epochs: int = 5
    in_memory: bool = False              # True: load the train split into RAM for a full shuffle
    max_train_rows: int | None = None    # smoke tests
    batch_size: int = 2048
    lr: float = 1e-4
    lr_schedule: str = "constant"        # "constant" (PICASSO) | "warmup_cosine" (reference code)
    n_lr_warmup: int = 1000
    grad_clip: float = 0.0               # 0 = off
    val_batches: int = 200
    log_every: int = 200
    save_every_epoch: bool = True
    seed: int = 42
    device: str = "cuda"


def sample_prefixes(d_sae: int, n_prefixes: int) -> torch.Tensor | None:
    if n_prefixes <= 1:
        return None
    lengths = torch.arange(1, d_sae)
    cdf = 1 - (1 / lengths.float()) ** 0.5
    pdf = torch.cat([cdf[:1], cdf[1:] - cdf[:-1]])
    picks = lengths[torch.multinomial(pdf / pdf.sum(), n_prefixes - 1, replacement=False)]
    return torch.sort(torch.cat([picks, torch.tensor([d_sae])])).values


def aux_k_loss(sae: TopKSAE, x: torch.Tensor, x_hat: torch.Tensor, h: torch.Tensor, dead: torch.Tensor) -> torch.Tensor:
    """Reconstruct the residual from the top ``aux_k`` dead latents (revives dead features)."""
    k = min(sae.cfg.aux_k, int(dead.sum()))
    if k == 0 or sae.cfg.aux_alpha == 0:
        return x.new_zeros(())
    _, idx = h.masked_fill(~dead, float("-inf")).topk(k, dim=-1)
    aux = torch.zeros_like(h).scatter(-1, idx, h.gather(-1, idx))
    residual = (x - x_hat).detach()
    recon = aux @ sae.W_dec   # no bias: the residual is bias-free
    return sae.cfg.aux_alpha * (recon - residual).pow(2).sum(-1).mean()


@torch.no_grad()
def datapoint_init(sae: TopKSAE, store: EmbeddingStore, blend: float, seed: int, normalize) -> None:
    acts = normalize(store.random_rows(sae.cfg.d_sae, seed=seed))
    centred = acts - acts.mean(dim=0, keepdim=True)
    noise = torch.nn.init.kaiming_uniform_(torch.empty_like(centred))
    rows = blend * centred + (1 - blend) * noise
    sae.W_dec.data.copy_(rows.to(sae.W_dec.device))
    sae.normalize_decoder()
    sae.W_enc.data.copy_(sae.W_dec.data.T)


def lr_at(step: int, cfg: TrainConfig, total: int) -> float:
    if cfg.lr_schedule == "constant":
        return cfg.lr
    if step < cfg.n_lr_warmup:
        return cfg.lr * step / cfg.n_lr_warmup
    progress = (step - cfg.n_lr_warmup) / max(1, total - cfg.n_lr_warmup)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def validate(sae: TopKSAE, val: EmbeddingStore, cfg: TrainConfig, normalize) -> dict[str, float]:
    sae.eval()
    acc = Accumulator(sae.cfg.d_sae, val.d, device=cfg.device)
    rows = val.random_rows(cfg.val_batches * cfg.batch_size, seed=0)   # fixed random subset across slides
    for s in range(0, len(rows), cfg.batch_size):
        x = normalize(rows[s : s + cfg.batch_size].to(cfg.device))
        f, x_hat = sae(x)
        acc.update(x, x_hat, f)
    sae.train()
    return acc.summary()


def run(cfg: TrainConfig) -> pathlib.Path:
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    train = EmbeddingStore(cfg.store_dir, cfg.field, "train", cfg.max_train_rows, in_memory=cfg.in_memory)
    val = EmbeddingStore(cfg.store_dir, cfg.field, "val")
    logger.info("train rows %d, val rows %d, d=%d", train.n, val.n, train.d)
    mean, std = train.mean.to(cfg.device), train.std.to(cfg.device)
    normalize = (lambda x: (x - mean) / std) if cfg.normalize_input else (lambda x: x)

    d_sae = cfg.expansion * train.d
    sae = TopKSAE(SAEConfig(train.d, d_sae, cfg.top_k, cfg.aux_k, cfg.aux_alpha))
    if cfg.datapoint_init_blend > 0:
        datapoint_init(sae, train, cfg.datapoint_init_blend, cfg.seed, lambda x: normalize(x.to(cfg.device)).cpu())
    sae.to(cfg.device).train()
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr, fused=cfg.device.startswith("cuda"))
    since_active = torch.zeros(d_sae, dtype=torch.long, device=cfg.device)
    steps_per_epoch = train.steps_per_epoch(cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(cfg.output_dir / "config.json", cfg)
    log_fd = open(cfg.output_dir / "log.jsonl", "a")
    logger.info("d_sae=%d steps/epoch=%d total=%d", d_sae, steps_per_epoch, total_steps)

    step, t0 = 0, time.time()
    for epoch in range(cfg.epochs):
        for x in train.batches(cfg.batch_size, seed=cfg.seed + epoch):
            x = normalize(x.to(cfg.device, non_blocking=True))
            sae.normalize_decoder()
            h = sae.pre_activation(x)
            values, idx = h.topk(cfg.top_k, dim=-1, sorted=False)
            f = torch.zeros_like(h).scatter(-1, idx, values)
            with torch.no_grad():
                since_active += len(x)
                since_active[(f != 0).any(dim=0)] = 0
                dead = since_active >= cfg.dead_threshold_samples
            x_hats = sae.decode(f, sample_prefixes(d_sae, cfg.n_prefixes))
            mse = (x_hats - x[:, None, :]).pow(2).sum(-1).mean()      # paper: mean_i ||x_hat - x||^2
            aux = aux_k_loss(sae, x, x_hats[:, -1], h, dead)
            (mse + aux).backward()
            sae.remove_parallel_grads()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, cfg, total_steps)
            if step % cfg.log_every == 0 or step == 1:
                with torch.no_grad():
                    ev = 1 - (x_hats[:, -1] - x).var(0).sum() / x.var(0).sum()
                rec = {"step": step, "epoch": epoch, "mse": mse.item(), "aux": aux.item(), "ev_batch": ev.item(),
                       "dead": int(dead.sum()), "lr": optimizer.param_groups[0]["lr"], "elapsed_s": time.time() - t0}
                log_fd.write(json.dumps(rec) + "\n"); log_fd.flush()
                logger.info("ep %d step %d/%d mse=%.3f aux=%.3f ev=%.4f dead=%d %.1f steps/s", epoch, step, total_steps,
                            rec["mse"], rec["aux"], rec["ev_batch"], rec["dead"], step / max(rec["elapsed_s"], 1e-6))
        v = validate(sae, val, cfg, normalize)
        v.update({"step": step, "epoch": epoch, "split": "val"})
        log_fd.write(json.dumps(v) + "\n"); log_fd.flush()
        logger.info("epoch %d VAL ev=%.4f ev_persample=%.4f nmse=%.4f cos=%.4f l0=%.1f dead=%d dense=%d active>0.1%%=%d",
                    epoch, v["explained_variance"], v["explained_variance_per_sample"], v["normalized_mse"], v["cosine"],
                    v["l0"], v["n_dead"], v["n_dense"], v["n_active_1e-3"])
        if cfg.save_every_epoch:
            dump(cfg.output_dir / f"sae_epoch{epoch:02d}.pt", sae, {"epoch": epoch, "step": step, "val": v,
                 "normalize_input": cfg.normalize_input, "field": cfg.field})
    dump(cfg.output_dir / "sae.pt", sae.eval().cpu(), {"epoch": cfg.epochs - 1, "step": step,
         "normalize_input": cfg.normalize_input, "field": cfg.field})
    logger.info("saved %s", cfg.output_dir / "sae.pt")
    return cfg.output_dir / "sae.pt"


if __name__ == "__main__":
    run(cli_config(TrainConfig, __doc__))
