# sae_picasso

Training code for the two models of PICASSO
([Kim et al., bioRxiv 2026](https://doi.org/10.64898/2026.06.12.731496)) on all diagnostic
H&E slides of TCGA:

- **SAE**: a TopK sparse autoencoder that splits a Virchow2 tile embedding into sparse concepts.
- **Emb2Img**: a latent diffusion model that turns a Virchow2 embedding back into an H&E tile.

Together they let you raise or lower one concept in an embedding and see how the tissue changes.

## Install

```bash
conda create -n picasso python=3.12 -y && conda activate picasso
conda install -c conda-forge openslide -y
pip install -e .
huggingface-cli login     # Virchow2 is gated: accept its terms on huggingface.co/paige-ai/Virchow2 first
source env.sh             # sets DATA_ROOT (default ./data) and OUTPUT_ROOT (default ./outputs)
```

## Pipeline

Every stage is resumable. Settings live in `configs/` and can be overridden with `--set key=value`.

### Data (stages 1-4)

```bash
PAR=8 bash pipeline/01_download.sh                 # 11,901 TCGA slides, 13.2 TB, from the GDC (open access)
bash pipeline/02_tile.sh                           # tissue mask, 256 px tiles at 0.5 µm/px
GPUS="0 1 2 3 4 5 6 7" bash pipeline/03_embed.sh   # Virchow2 CLS per tile
bash pipeline/04_build_store.sh                    # train / val / test split by slide (80/10/10)
```

The slide list is `manifests/tcga_dx_all.csv`. To refresh it from the GDC, run
`scripts/gdc_list.sh` and `scripts/make_manifest.py`.

To train on some cancer types only, for example brain tumours, see
[docs/train_on_subset.md](docs/train_on_subset.md).

### SAE (stage 5)

```bash
SEED=42 bash pipeline/05_train_sae.sh              # one GPU; then test-set metrics
```

The train split is streamed from disk. With about 320 GB of free RAM, add
`--set in_memory=true` for a full shuffle.

### Emb2Img (stages 6-7)

```bash
GPUS="0 1 2 3 4 5 6 7" bash pipeline/06_build_pairs.sh                 # VAE latents of 16.4M random train tiles
GPUS=0,1,2,3,4,5,6,7 GRAD_ACCUM=1 bash pipeline/07_train_emb2img.sh    # 16,000 steps, then sampling + eval
```

Keep the effective batch at 1024: with 4 GPUs set `GRAD_ACCUM=2`, with 1 GPU set `GRAD_ACCUM=8`.

The closed-loop score in the eval works like this: generate a tile from an embedding, embed the
generated tile with Virchow2 again, and take the cosine to the input embedding.

### Concept intervention

```bash
python scripts/demo.py --image tile.png --sae outputs/sae/virchow2_cls_x32_k30_seed42/sae.pt \
    --emb2img outputs/emb2img_sd15_256/ema --latent <concept id> --levels 0,10,20,30
```

The input is a 256 px tile at 0.5 µm/px. The script prints the tile's strongest concepts and
writes the tile next to one generated tile per strength.

| Other scripts | What they do |
|---|---|
| `scripts/concept_intervention.py` | Intervention on test tiles, with numbers |
| `scripts/find_concepts.py` | Concepts that track a pixel measure, with example tiles |
| `scripts/sae_recon_gen.py` | Checks that SAE-reconstructed embeddings still generate faithful tiles |
| `scripts/cfg_sweep.sh` | Closed-loop score across guidance scales |

## Resources

Times are estimates from throughput measured on RTX A6000 GPUs.

| Stage | Disk | Time |
|---|---|---|
| Download | 13.2 TB | about 1 day |
| Tiling | 160 GB | about 1 h on 24 CPU cores |
| Virchow2 embedding, about 154M tiles | 400 GB | about 27 h on 8 GPUs |
| Store | 400 GB | about 1 h |
| SAE | 0.4 GB | about 8 h on 1 GPU |
| Emb2Img pairs | 170 GB | about 5 h on 8 GPUs |
| Emb2Img training | 13 GB | about 8 h on 8 GPUs |

## Differences from PICASSO

| | PICASSO | Here | Why |
|---|---|---|---|
| Tissue segmentation | CLAM | Saturation threshold with Otsu | Same tile geometry without the CLAM dependency |
| SAE input | Virchow2, d = 1280 | Virchow2 CLS token | The paper gives d = 1280 but not which token |
| SAE loss | MSE | MSE + AuxK | SAELens default for TopK; it revives dead concepts |
| Emb2Img base model | Stable Diffusion 2 | Stable Diffusion 1.5 | SD 2 weights are no longer on Hugging Face |
| Emb2Img resolution | 512 px (tiles upscaled) | 256 px | Tiles are 256 px; 4× less compute |
| Emb2Img initialisation | not stated | SD 1.5 weights | Converges in hours instead of days |
| Guidance scale | 6.5 | 4.5 | Best closed-loop score in our runs |

Everything else follows the paper. The SAE uses k = 30, 32× expansion, Adam 1e-4, batch 2048
and 5 epochs. Emb2Img uses 16,000 steps, batch 1024, AdamW 1e-4, condition dropout 0.1, EMA
and PNDM sampling.

## Licenses

Virchow2 weights are CC-BY-NC-ND 4.0, Stable Diffusion 1.5 is CreativeML OpenRAIL-M, and TCGA
data are open access through the GDC. Models trained with this code inherit these terms and are
for non-commercial research use.
