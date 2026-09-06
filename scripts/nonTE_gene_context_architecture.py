#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/home/igor/arabidopsis_siRNA_methylation")
OUT  = ROOT / "results/nonTE_gene_analysis"

INFILE = OUT / "all_TE_distant_1kb_concordant_gene_windows.tsv"

# ------------------------------------------------------------
# Load standardized 1% joint-tail candidates
# ------------------------------------------------------------

d = pd.read_csv(INFILE, sep="\t", low_memory=False)
d["tail_pct"] = pd.to_numeric(d["tail_pct"], errors="coerce")

d = d[d["tail_pct"] == 1].copy()

print(f"1% TE-distant window/gene records: {len(d):,}")
print("Columns:", ", ".join(d.columns))

# Gene presence in a context is sufficient here.
# Collapse windows, resolutions and time points.
g = (
    d.groupby(
        ["stress", "direction", "feature", "gene_id", "context"],
        observed=True
    )
    .agg(
        n_records=("gene_id", "size"),
        n_resolutions=("resolution", "nunique"),
        n_times=("time", "nunique")
    )
    .reset_index()
)

# ------------------------------------------------------------
# Context membership
# ------------------------------------------------------------

presence = (
    g.assign(present=1)
     .pivot_table(
         index=["stress", "direction", "feature", "gene_id"],
         columns="context",
         values="present",
         aggfunc="max",
         fill_value=0
     )
     .reset_index()
)

for c in ["CG", "CHG", "CHH"]:
    if c not in presence.columns:
        presence[c] = 0
    presence[c] = presence[c].astype(int)


def architecture(r):
    x = [c for c in ["CG", "CHG", "CHH"] if r[c] == 1]
    return "+".join(x)


presence["context_architecture"] = presence.apply(
    architecture, axis=1
)

presence["n_contexts"] = (
    presence[["CG", "CHG", "CHH"]].sum(axis=1)
)

presence.to_csv(
    OUT / "TE_distant_1kb_gene_context_architecture.tsv",
    sep="\t", index=False
)

# ------------------------------------------------------------
# Architecture summary
# ------------------------------------------------------------

summary = (
    presence.groupby(
        ["stress", "direction", "feature", "context_architecture"],
        observed=True
    )
    .size()
    .reset_index(name="n_genes")
)

totals = (
    presence.groupby(
        ["stress", "direction", "feature"],
        observed=True
    )
    .size()
    .reset_index(name="total_genes")
)

summary = summary.merge(
    totals,
    on=["stress", "direction", "feature"],
    how="left"
)

summary["percent"] = (
    100 * summary["n_genes"] / summary["total_genes"]
)

summary.to_csv(
    OUT / "TE_distant_1kb_context_architecture_summary.tsv",
    sep="\t", index=False
)

# ------------------------------------------------------------
# Heat gene-body detailed output
# ------------------------------------------------------------

heat = presence[
    (presence["stress"] == "heat") &
    (presence["feature"] == "gene_body")
].copy()

# Add symbols from Araport GFF if possible
import gzip
import re

gff = ROOT / "genome/Araport11_GFF3_genes_transposons.current.gff.gz"

symbols = {}

with gzip.open(
    gff, "rt",
    encoding="utf-8",
    errors="replace"
) as f:
    for line in f:
        if line.startswith("#"):
            continue

        x = line.rstrip().split("\t")
        if len(x) < 9 or x[2] != "gene":
            continue

        attrs = {}
        for z in x[8].split(";"):
            if "=" in z:
                k,v = z.split("=",1)
                attrs[k] = v

        gid = attrs.get("ID")
        if gid:
            symbols[gid] = attrs.get("symbol","")

heat["symbol"] = heat["gene_id"].map(symbols).fillna("")

heat.to_csv(
    OUT / "heat_TE_distant_gene_context_architecture.tsv",
    sep="\t", index=False
)

# ------------------------------------------------------------
# Specific HSP / photosynthesis sets
# ------------------------------------------------------------

HSP = {
    "AT1G16030",
    "AT1G74310",
    "AT3G09440",
    "AT3G12580",
    "AT5G12020",
    "AT5G52640",
    "AT5G56010"
}

PHOTO = {
    "AT1G06680",
    "AT1G29920",
    "AT1G29930",
    "AT1G61520",
    "AT1G67090",
    "AT1G74470",
    "AT3G08940",
    "AT3G12780",
    "AT3G47470",
    "AT3G60750",
    "AT4G10340",
    "AT5G54270",
    "AT5G66570"
}

focus = heat[
    heat["gene_id"].isin(HSP | PHOTO)
].copy()

focus["module"] = np.where(
    focus["gene_id"].isin(HSP),
    "heat_response",
    "photosynthesis"
)

focus = focus.sort_values(
    ["module", "direction", "gene_id"]
)

focus.to_csv(
    OUT / "heat_HSP_photosynthesis_contexts.tsv",
    sep="\t", index=False
)

# ------------------------------------------------------------
# Cross-stress recurrence
# ------------------------------------------------------------

cross = (
    presence.groupby(
        ["direction", "feature", "gene_id"],
        observed=True
    )
    .agg(
        n_stresses=("stress", "nunique"),
        stresses=("stress",
                  lambda x: ";".join(sorted(set(x)))),
        architectures=("context_architecture",
                       lambda x: ";".join(sorted(set(x))))
    )
    .reset_index()
)

cross = cross[cross["n_stresses"] >= 2].copy()

cross["symbol"] = cross["gene_id"].map(symbols).fillna("")

cross = cross.sort_values(
    ["n_stresses", "direction", "feature", "gene_id"],
    ascending=[False, True, True, True]
)

cross.to_csv(
    OUT / "TE_distant_cross_stress_recurrent_genes.tsv",
    sep="\t", index=False
)

# ------------------------------------------------------------
# Console report
# ------------------------------------------------------------

print("\n============================================")
print("CONTEXT ARCHITECTURE BY STRESS")
print("============================================")

for keys, x in summary.groupby(
    ["stress", "direction", "feature"],
    observed=True
):
    stress, direction, feature = keys

    print(
        f"\n### {stress} | {direction} | {feature}"
    )

    x = x.sort_values("n_genes", ascending=False)

    for _, r in x.iterrows():
        print(
            f"{r.context_architecture:11s} "
            f"{int(r.n_genes):4d} "
            f"({r.percent:5.1f}%)"
        )


print("\n============================================")
print("HEAT RESPONSE MODULE")
print("============================================")

x = focus[
    (focus["module"] == "heat_response") &
    (focus["direction"] == "gain")
]

for _, r in x.iterrows():
    label = (
        f"{r.gene_id} ({r.symbol})"
        if r.symbol else r.gene_id
    )
    print(
        f"{label:30s} {r.context_architecture}"
    )


print("\n============================================")
print("PHOTOSYNTHESIS MODULE")
print("============================================")

x = focus[
    (focus["module"] == "photosynthesis") &
    (focus["direction"] == "loss")
]

for _, r in x.iterrows():
    label = (
        f"{r.gene_id} ({r.symbol})"
        if r.symbol else r.gene_id
    )
    print(
        f"{label:30s} {r.context_architecture}"
    )


print("\n============================================")
print("CROSS-STRESS RECURRENT GENES")
print("============================================")

print(
    f"Genes appearing in >=2 stresses: "
    f"{cross['gene_id'].nunique():,}"
)

if len(cross):
    print("\nTop recurrent:")
    for _, r in cross.head(40).iterrows():
        label = (
            f"{r.gene_id} ({r.symbol})"
            if r.symbol else r.gene_id
        )
        print(
            f"{label:30s} "
            f"{r.direction:4s} "
            f"{r.feature:9s} "
            f"n={r.n_stresses} "
            f"{r.stresses}"
        )

print("\nWrote:")
for f in [
    "TE_distant_1kb_gene_context_architecture.tsv",
    "TE_distant_1kb_context_architecture_summary.tsv",
    "heat_TE_distant_gene_context_architecture.tsv",
    "heat_HSP_photosynthesis_contexts.tsv",
    "TE_distant_cross_stress_recurrent_genes.tsv"
]:
    print(" ", OUT / f)

print("\nDONE")
