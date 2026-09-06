#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
from collections import Counter

ROOT = Path("/home/igor/arabidopsis_siRNA_methylation")

INFILE = ROOT / "results/nonTE_gene_analysis/all_TE_distant_1kb_concordant_gene_windows.tsv"
OUT = ROOT / "results/nonTE_gene_analysis"

d = pd.read_csv(INFILE, sep="\t")

# Normalize
for c in ["stress","direction","feature","context","time"]:
    d[c] = d[c].astype(str)

# ----------------------------------------------------------
# Strict candidate definition
# ----------------------------------------------------------

q = d[
    (d["stress"].str.lower() == "heat") &
    (d["direction"].str.lower() == "loss") &
    (d["feature"].str.lower() == "gene_body") &
    (d["TE_distant_1kb"] == True) &
    (d["tail_pct"] == 1)
].copy()

# One record per exact window/context/time
key = [
    "gene_id","resolution","time","context",
    "chr","start","end"
]

q = q.drop_duplicates(key)

print("\nStrict heat-loss joint-1% records:", len(q))
print("Unique genes:", q["gene_id"].nunique())

# ----------------------------------------------------------
# Exact same window + same time
# ----------------------------------------------------------

group_cols = [
    "gene_id","resolution","time",
    "chr","start","end"
]

rows = []

for vals, z in q.groupby(group_cols):

    contexts = sorted(
        set(z["context"]),
        key=lambda x: ["CG","CHG","CHH"].index(x)
    )

    if len(contexts) < 2:
        continue

    gene, resolution, time, chrom, start, end = vals

    rows.append({
        "gene_id": gene,
        "resolution": resolution,
        "time": time,
        "chr": chrom,
        "start": start,
        "end": end,
        "n_contexts": len(contexts),
        "contexts": "+".join(contexts)
    })

shared = pd.DataFrame(rows)

if len(shared):
    shared = shared.sort_values(
        ["n_contexts","time","resolution","chr","start"],
        ascending=[False,True,True,True,True]
    )

outfile = OUT / "heat_same_time_exact_crosscontext_windows_1pct.tsv"
shared.to_csv(outfile, sep="\t", index=False)

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------

print("\n========================================")
print("SAME-TIME EXACT WINDOWS")
print("========================================")

if len(shared):
    print(shared.to_string(index=False))
else:
    print("None")

print("\n========================================")
print("COUNTS BY CONTEXT COMBINATION")
print("========================================")

if len(shared):
    s = (
        shared.groupby(["contexts"])
        .size()
        .reset_index(name="n_window_time_events")
        .sort_values("n_window_time_events", ascending=False)
    )
    print(s.to_string(index=False))

print("\n========================================")
print("COUNTS BY TIME")
print("========================================")

if len(shared):
    s = (
        shared.groupby(["time","contexts"])
        .size()
        .reset_index(name="n_window_time_events")
        .sort_values(["time","contexts"])
    )
    print(s.to_string(index=False))

print("\n========================================")
print("COUNTS BY RESOLUTION")
print("========================================")

if len(shared):
    s = (
        shared.groupby(["resolution","contexts"])
        .size()
        .reset_index(name="n_window_time_events")
        .sort_values(["resolution","contexts"])
    )
    print(s.to_string(index=False))

# ----------------------------------------------------------
# Unique physical loci, collapsing recurrence across time
# ----------------------------------------------------------

if len(shared):

    locus_cols = [
        "gene_id","resolution","chr","start","end"
    ]

    locus_rows = []

    for vals, z in shared.groupby(locus_cols):

        gene, resolution, chrom, start, end = vals

        contexts_union = set()
        for x in z["contexts"]:
            contexts_union.update(x.split("+"))

        contexts_union = sorted(
            contexts_union,
            key=lambda x: ["CG","CHG","CHH"].index(x)
        )

        times = sorted(
            set(z["time"]),
            key=lambda x: ["6h","12h","24h"].index(x)
        )

        locus_rows.append({
            "gene_id": gene,
            "resolution": resolution,
            "chr": chrom,
            "start": start,
            "end": end,
            "contexts_across_same_time_events":
                "+".join(contexts_union),
            "n_same_time_points": len(times),
            "same_time_points": ",".join(times)
        })

    loci = pd.DataFrame(locus_rows)

    locfile = OUT / "heat_same_time_unique_crosscontext_loci_1pct.tsv"
    loci.to_csv(locfile, sep="\t", index=False)

    print("\n========================================")
    print("UNIQUE PHYSICAL LOCI")
    print("========================================")
    print(loci.to_string(index=False))

    print("\nUnique physical cross-context loci:",
          len(loci))
    print("Genes containing them:",
          loci["gene_id"].nunique())

# ----------------------------------------------------------
# Genes with same-time multi-context events
# ----------------------------------------------------------

if len(shared):

    gene_summary = (
        shared.groupby("gene_id")
        .agg(
            n_window_time_events=("gene_id","size"),
            n_times=("time","nunique"),
            n_resolutions=("resolution","nunique"),
            max_contexts=("n_contexts","max")
        )
        .reset_index()
        .sort_values(
            ["max_contexts","n_window_time_events"],
            ascending=[False,False]
        )
    )

    genefile = OUT / "heat_same_time_crosscontext_gene_summary_1pct.tsv"
    gene_summary.to_csv(genefile, sep="\t", index=False)

    print("\n========================================")
    print("GENE SUMMARY")
    print("========================================")
    print(gene_summary.to_string(index=False))

print("\nSaved:", outfile)
print("DONE")
