"""Slide manifest from a GDC listing (``scripts/gdc_list.sh``) or from an existing manifest CSV.

By default every listed slide is kept. ``--projects`` keeps only some TCGA projects, and
``--budget-bytes`` draws a random subset of a given size (at least ``--floor`` slides per project).

    python scripts/make_manifest.py --input gdc_tcga_dx_slides.tsv --out manifests/tcga_dx_all.csv
    python scripts/make_manifest.py --input manifests/tcga_dx_all.csv --projects TCGA-GBM,TCGA-LGG --out manifests/brain.csv
"""
import argparse, collections, csv, random

p = argparse.ArgumentParser()
p.add_argument("--input", required=True, help="GDC listing (.tsv) or manifest (.csv)")
p.add_argument("--out", required=True)
p.add_argument("--projects", default=None, help="comma-separated TCGA project ids, e.g. TCGA-GBM,TCGA-LGG")
p.add_argument("--budget-bytes", type=float, default=None)
p.add_argument("--floor", type=int, default=15)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

if a.input.endswith(".tsv"):
    rows = list(csv.DictReader(open(a.input), delimiter="\t"))
    pcol = [c for c in rows[0] if c.endswith("project_id")][0]
    ccol = [c for c in rows[0] if c.endswith("submitter_id")][0]
    for r in rows:
        r["project"], r["case"] = r[pcol], r[ccol]
else:
    rows = list(csv.DictReader(open(a.input)))
for r in rows:
    r["size"] = int(r["file_size"])
if a.projects:
    keep = set(a.projects.split(","))
    unknown = keep - {r["project"] for r in rows}
    if unknown:
        raise SystemExit(f"unknown projects: {sorted(unknown)}")
    rows = [r for r in rows if r["project"] in keep]
rng = random.Random(a.seed)
if a.budget_bytes is None:
    chosen = rows
else:
    byp = collections.defaultdict(list)
    for r in rows:
        byp[r["project"]].append(r)
    chosen = []
    for proj in sorted(byp):
        rng.shuffle(byp[proj]); chosen += byp[proj][: a.floor]
    acc = sum(r["size"] for r in chosen)
    rest = [r for proj in sorted(byp) for r in byp[proj][a.floor :]]
    rng.shuffle(rest)
    for r in rest:
        if acc + r["size"] <= a.budget_bytes:
            chosen.append(r); acc += r["size"]
chosen = sorted(chosen, key=lambda r: (r["project"], r["file_name"]))
with open(a.out, "w", newline="") as fd:
    w = csv.writer(fd, lineterminator="\n")
    w.writerow(["order", "file_id", "file_name", "project", "case", "file_size", "md5sum"])
    for i, r in enumerate(chosen):
        w.writerow([i, r["file_id"], r["file_name"], r["project"], r["case"], r["size"], r["md5sum"]])
cnt = collections.Counter(r["project"] for r in chosen)
print(f"{len(chosen)} slides, {sum(r['size'] for r in chosen) / 1e12:.2f} TB:", dict(sorted(cnt.items())))
