"""Emb2Img = SD v2 U-Net conditioned on a Virchow2 embedding through a linear projection.

The 1280-d embedding is projected to the U-Net's cross-attention width (1024 for SD v2)
and fed as a single context token. A learned null token replaces it for classifier-free
guidance dropout.
"""

from __future__ import annotations

import pathlib

import torch


class Emb2Img(torch.nn.Module):
    def __init__(self, unet, d_emb: int = 1280):
        super().__init__()
        self.unet = unet
        d_ctx = unet.config.cross_attention_dim
        self.proj = torch.nn.Sequential(torch.nn.LayerNorm(d_emb), torch.nn.Linear(d_emb, d_ctx))
        self.null = torch.nn.Parameter(torch.zeros(1, 1, d_ctx))

    def context(self, emb: torch.Tensor, drop: torch.Tensor | None = None) -> torch.Tensor:
        ctx = self.proj(emb)[:, None, :]                       # [b, 1, d_ctx]
        if drop is not None:
            ctx = torch.where(drop[:, None, None], self.null.expand(len(emb), -1, -1).to(ctx.dtype), ctx)
        return ctx

    def forward(self, noisy_latents, timesteps, emb, drop=None):
        return self.unet(noisy_latents, timesteps, encoder_hidden_states=self.context(emb, drop)).sample

    def save(self, path: pathlib.Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        from safetensors.torch import save_file

        self.unet.save_pretrained(path / "unet")
        state = {f"proj.{k}": v.detach().cpu().contiguous() for k, v in self.proj.state_dict().items()}
        state["null"] = self.null.detach().cpu().contiguous()
        save_file(state, str(path / "cond.safetensors"))

    @classmethod
    def load(cls, path: pathlib.Path, device: str = "cuda", dtype=torch.float16) -> "Emb2Img":
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained(path / "unet", torch_dtype=dtype)
        m = cls(unet)
        if (path / "cond.safetensors").exists():
            from safetensors.torch import load_file

            state = load_file(str(path / "cond.safetensors"))
            m.proj.load_state_dict({k[5:]: v for k, v in state.items() if k.startswith("proj.")})
            m.null.data.copy_(state["null"])
        else:  # older checkpoints
            cond = torch.load(path / "cond.pt", map_location="cpu")
            m.proj.load_state_dict(cond["proj"])
            m.null.data.copy_(cond["null"])
        return m.to(device=device, dtype=dtype).eval()
