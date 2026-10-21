"""Emb2Img training (DDP, any number of GPUs) with checkpoint/resume.

    python -m torch.distributed.run --nproc_per_node 8 -m picasso.emb2img.train --config configs/emb2img/train.yaml

Effective batch = batch_size x world_size x grad_accum. Latents/embeddings are memory-mapped
(the OS page cache is shared between ranks). Rank 0 keeps the EMA and writes
``ckpt/`` (model + EMA + optimizer + step; resumable) and ``ema/`` (inference checkpoint).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pathlib
import time

import numpy as np
import torch
import torch.distributed as dist

from picasso.config import cli_config, dump_json
from picasso.emb2img.model import Emb2Img

logger = logging.getLogger("picasso.emb2img.train")


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    pairs_dir: pathlib.Path
    output_dir: pathlib.Path
    sd_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    init_from_sd: bool = True
    steps: int = 16000
    batch_size: int = 128
    grad_accum: int = 1
    lr: float = 1e-4
    lr_warmup: int = 500
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    cond_dropout: float = 0.1
    ema_decay: float = 0.9999
    gradient_checkpointing: bool = True
    dtype: str = "bfloat16"
    log_every: int = 50
    save_every: int = 1000
    resume: bool = True
    seed: int = 0


def run(cfg: TrainConfig) -> None:
    from diffusers import DDPMScheduler, UNet2DConditionModel
    from diffusers.training_utils import EMAModel

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    device = f"cuda:{local}"
    torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = getattr(torch, cfg.dtype)
    lat = np.load(cfg.pairs_dir / "train_latents.npy", mmap_mode="r")
    emb = np.load(cfg.pairs_dir / "train_emb.npy", mmap_mode="r")
    n = len(lat)
    if rank == 0:
        logger.info("pairs %d, latents %s, world %d, effective batch %d", n, tuple(lat.shape[1:]), world,
                    cfg.batch_size * world * cfg.grad_accum)

    scheduler = DDPMScheduler.from_pretrained(cfg.sd_model, subfolder="scheduler")
    if cfg.init_from_sd:
        unet = UNet2DConditionModel.from_pretrained(cfg.sd_model, subfolder="unet")
    else:
        unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(cfg.sd_model, subfolder="unet"))
    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    model = Emb2Img(unet).to(device).train()
    ema = EMAModel(model.parameters(), decay=cfg.ema_decay) if rank == 0 else None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999), fused=True)
    step = 0
    ckpt_dir = cfg.output_dir / "ckpt"
    if cfg.resume and (ckpt_dir / "state.pt").exists():
        t_load = time.time()
        state = torch.load(ckpt_dir / "state.pt", map_location="cpu", weights_only=False)
        logger.info("rank %d: loaded state.pt in %.0fs", rank, time.time() - t_load)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        logger.info("rank %d: model+opt restored", rank)
        if ema is not None:
            ema.load_state_dict(state["ema"])
            ema.to(device)
            logger.info("rank %d: ema restored", rank)
        step = int(state["step"])
        if rank == 0:
            logger.info("resumed from step %d", step)
        del state
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg.lr_warmup), last_epoch=step - 1 if step else -1)

    if rank == 0:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        dump_json(cfg.output_dir / "config.json", cfg)
        log_fd = open(cfg.output_dir / "log.jsonl", "a")
    g = torch.Generator().manual_seed(cfg.seed * 1000 + rank + step)
    T = scheduler.config.num_train_timesteps
    t0, step0 = time.time(), step
    run_loss, micro = 0.0, 0
    while step < cfg.steps:
        idx = np.sort(torch.randint(0, n, (cfg.batch_size,), generator=g).numpy())
        x0 = torch.from_numpy(np.ascontiguousarray(lat[idx])).to(device, non_blocking=True).float()
        e = torch.from_numpy(np.ascontiguousarray(emb[idx])).to(device, non_blocking=True).float()
        noise = torch.randn_like(x0)
        t = torch.randint(0, T, (len(x0),), device=device)
        xt = scheduler.add_noise(x0, noise, t)
        drop = torch.rand(len(x0), device=device) < cfg.cond_dropout
        micro += 1
        sync = micro % cfg.grad_accum == 0
        ctx = ddp.no_sync() if not sync else torch.enable_grad()
        with ctx:
            with torch.autocast("cuda", dtype=dtype):
                pred = ddp(xt, t, e, drop)
            loss = torch.nn.functional.mse_loss(pred.float(), noise) / cfg.grad_accum
            loss.backward()
        run_loss += loss.item()
        if not sync:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if ema is not None:
            ema.step(model.parameters())
        step += 1
        if rank == 0 and (step % cfg.log_every == 0 or step == step0 + 1):
            rec = {"step": step, "loss": run_loss / (cfg.log_every if step % cfg.log_every == 0 else 1), "lr": sched.get_last_lr()[0],
                   "s_per_step": (time.time() - t0) / (step - step0), "mem_gb": torch.cuda.max_memory_allocated() / 1e9}
            log_fd.write(json.dumps(rec) + "\n"); log_fd.flush()
            logger.info("step %d/%d loss=%.4f lr=%.1e %.2fs/step eta=%.1fh mem=%.1fGB", step, cfg.steps, rec["loss"], rec["lr"],
                        rec["s_per_step"], rec["s_per_step"] * (cfg.steps - step) / 3600, rec["mem_gb"])
            run_loss = 0.0
        elif step % cfg.log_every == 0:
            run_loss = 0.0
        if step % cfg.save_every == 0 or step == cfg.steps:
            if rank == 0:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "ema": ema.state_dict(), "step": step},
                           ckpt_dir / "state.pt.tmp")
                os.replace(ckpt_dir / "state.pt.tmp", ckpt_dir / "state.pt")
                ema.store(model.parameters()); ema.copy_to(model.parameters())
                model.save(cfg.output_dir / "ema")
                if step % (cfg.save_every * 4) == 0:
                    model.save(cfg.output_dir / f"ema_step{step:06d}")
                ema.restore(model.parameters())
                logger.info("saved checkpoint at step %d", step)
            dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    run(cli_config(TrainConfig, __doc__))
