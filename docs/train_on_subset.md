# Training on part of TCGA

The pipeline is the same for any subset of TCGA projects. You choose the slides in stage 1, or
the projects in stage 4 if all slides are already embedded. Everything else runs unchanged.

The example below trains on brain tumours: TCGA-GBM and TCGA-LGG, 1,704 slides, 1.4 TB.

List the projects and their slide counts with:

```bash
cut -d, -f4 manifests/tcga_dx_all.csv | tail -n +2 | sort | uniq -c
```

## Option A: start from raw slides

Make a manifest with only the projects you want:

```bash
python scripts/make_manifest.py --input manifests/tcga_dx_all.csv \
    --projects TCGA-GBM,TCGA-LGG --out manifests/brain.csv
```

Point the run at its own folders, so it does not mix with other runs:

```bash
export DATA_ROOT=$PWD/data_brain OUTPUT_ROOT=$PWD/outputs_brain
source env.sh
```

Then run all stages as in the main README, passing the new manifest to stage 1:

```bash
MANIFEST=manifests/brain.csv PAR=8 bash pipeline/01_download.sh
bash pipeline/02_tile.sh
GPUS="0 1 2 3 4 5 6 7" bash pipeline/03_embed.sh
bash pipeline/04_build_store.sh
SEED=42 bash pipeline/05_train_sae.sh --set epochs=20
GPUS="0 1 2 3 4 5 6 7" bash pipeline/06_build_pairs.sh
GPUS=0,1,2,3,4,5,6,7 bash pipeline/07_train_emb2img.sh
```

## Option B: full TCGA is already embedded

Stages 1-3 are already done, so build a store for the subset only. `STORE_DIR` and `PAIRS_DIR`
tell every later stage where the subset's training data lives:

```bash
export OUTPUT_ROOT=$PWD/outputs_brain
export STORE_DIR=$DATA_ROOT/store/brain PAIRS_DIR=$DATA_ROOT/emb2img_pairs_brain
source env.sh
PROJECTS="[TCGA-GBM,TCGA-LGG]" bash pipeline/04_build_store.sh
SEED=42 bash pipeline/05_train_sae.sh --set epochs=20
GPUS="0 1 2 3 4 5 6 7" bash pipeline/06_build_pairs.sh
GPUS=0,1,2,3,4,5,6,7 bash pipeline/07_train_emb2img.sh
```

## What to adjust for a smaller set

- **SAE epochs.** The default of 5 epochs is meant for the 123M tiles of full TCGA. A brain run
  has about 12M training tiles, so raise the epochs to 15-20 to see enough samples.
- **Loading into RAM.** A smaller train split often fits in memory: about 2.5 GB per million
  tiles. Add `--set in_memory=true` to stage 5 for a full shuffle.
- **Emb2Img pairs.** Nothing to change. If the train split has fewer than 16.4M tiles, all of
  them are used.
- **Test set.** The store keeps 10 % of the slides of each project for testing, so every
  project in the subset is represented.
