#!/usr/bin/env python3

import pandas as pd
from itertools import combinations
from pathlib import Path

ROOT = Path("/home/igor/arabidopsis_siRNA_methylation")

INFILE = ROOT / "results/nonTE_gene_analysis/all_TE_distant_1kb_concordant_gene_windows.tsv"
OUT = ROOT / "results/nonTE_gene_analysis"

genes = [
    "AT1G29930",
    "AT1G61520",
    "AT1G67090",
    "AT1G74470",
    "AT2G36530",
    "AT3G47470",
    "AT3G52930",
    "AT4G10340",
    "AT3G60750"
]

d = pd.read_csv(INFILE, sep="\t")

# String normalization
d["stress"] = d["stress"].astype(str).str.lower()
d["direction"] = d["direction"].astype(str).str.lower()
d["feature"] = d["feature"].astype(str).str.lower()

# Strongest candidate definition only:
# heat, loss, gene body, TE-distant >=1 kb, joint 1% tail
q = d[
    (d["stress"] == "heat") &
    (d["direction"] == "loss") &
    (d["feature"] == "gene_body") &
    (d["TE_distant_1kb"] == True) &
    (d["tail_pct"] == 1) &
    (d["gene_id"].isin(genes))
].copy()

# Remove duplicates caused by any upstream duplication.
key = [
    "gene_id","context","resolution","time",
    "chr","start","end"
]

q = q.drop_duplicates(key)

print("\n1% records:", len(q))

print("\n========================================")
print("1% WINDOWS BY GENE / CONTEXT / RESOLUTION")
print("========================================")

counts = (
    q.groupby(["gene_id","context","resolution"])
     .size()
     .reset_index(name="n_windows")
)

print(counts.to_string(index=False))

# ----------------------------------------------------------
# Exact coordinate sharing across methylation contexts
# Ignore time initially:
# same genomic window must be candidate in >=2 contexts.
# ----------------------------------------------------------

coord_cols = [
    "gene_id","resolution","chr","start","end"
]

records = []

for keyvals, z in q.groupby(coord_cols):

    contexts = sorted(set(z["context"]))

    if len(contexts) < 2:
        continue

    gene, resolution, chrom, start, end = keyvals

    times_by_context = []

    for ctx in contexts:
        times = sorted(
            set(z.loc[z["context"] == ctx, "time"].astype(str))
        )
        times_by_context.append(
            ctx + ":" + ",".join(times)
        )

    records.append({
        "gene_id": gene,
        "resolution": resolution,
        "chr": chrom,
        "start": start,
        "end": end,
        "n_contexts": len(contexts),
        "contexts": "+".join(contexts),
        "times_by_context": ";".join(times_by_context)
    })

shared = pd.DataFrame(records)

if len(shared):
    shared = shared.sort_values(
        ["n_contexts","gene_id","resolution","start"],
        ascending=[False,True,True,True]
    )

shared_file = OUT / "heat_validated_exact_crosscontext_windows_1pct.tsv"
shared.to_csv(shared_file, sep="\t", index=False)

print("\n========================================")
print("EXACT WINDOWS SHARED BY >=2 CONTEXTS")
print("========================================")

if len(shared):
    print(shared.to_string(index=False))
else:
    print("None")

# ----------------------------------------------------------
# Gene summary
# ----------------------------------------------------------

summary = []

for gene in genes:

    z = q[q["gene_id"] == gene]

    all_coords = set(
        zip(
            z["resolution"],
            z["chr"],
            z["start"],
            z["end"]
        )
    )

    shared_gene = (
        shared[shared["gene_id"] == gene]
        if len(shared)
        else pd.DataFrame()
    )

    n_shared = len(shared_gene)

    n_three = (
        int((shared_gene["n_contexts"] == 3).sum())
        if n_shared
        else 0
    )

    summary.append({
        "gene_id": gene,
        "n_unique_1pct_windows": len(all_coords),
        "n_windows_shared_2plus_contexts": n_shared,
        "n_windows_shared_all3_contexts": n_three,
        "fraction_windows_shared_2plus":
            n_shared / len(all_coords)
            if len(all_coords) else 0
    })

summary = pd.DataFrame(summary)

summary_file = OUT / "heat_validated_exact_window_sharing_summary_1pct.tsv"
summary.to_csv(summary_file, sep="\t", index=False)

print("\n========================================")
print("GENE SUMMARY")
print("========================================")
print(summary.to_string(index=False))

# ----------------------------------------------------------
# Pairwise context overlap
# ----------------------------------------------------------

print("\n========================================")
print("PAIRWISE EXACT-WINDOW OVERLAP")
print("========================================")

pair_rows = []

for gene in genes:

    z = q[q["gene_id"] == gene]

    for resolution in [100,500]:

        zr = z[z["resolution"] == resolution]

        sets = {}

        for ctx in ["CG","CHG","CHH"]:
            x = zr[zr["context"] == ctx]
            sets[ctx] = set(
                zip(x["chr"],x["start"],x["end"])
            )

        for a,b in combinations(["CG","CHG","CHH"],2):

            inter = sets[a] & sets[b]
            union = sets[a] | sets[b]

            pair_rows.append({
                "gene_id": gene,
                "resolution": resolution,
                "comparison": f"{a}-{b}",
                "n_A": len(sets[a]),
                "n_B": len(sets[b]),
                "n_shared": len(inter),
                "jaccard":
                    len(inter)/len(union)
                    if union else 0
            })

pairs = pd.DataFrame(pair_rows)

pair_file = OUT / "heat_validated_pairwise_window_overlap_1pct.tsv"
pairs.to_csv(pair_file, sep="\t", index=False)

print(
    pairs[pairs["n_shared"] > 0]
    .sort_values(
        ["gene_id","resolution","comparison"]
    )
    .to_string(index=False)
)

print("\nSaved:")
print(" ", shared_file)
print(" ", summary_file)
print(" ", pair_file)
print("\nDONE")
