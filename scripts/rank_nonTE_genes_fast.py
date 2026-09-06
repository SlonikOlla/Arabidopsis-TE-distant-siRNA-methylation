#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/home/igor/arabidopsis_siRNA_methylation")
OUT = ROOT / "results/nonTE_gene_analysis"

inputs = {
    "nonTE": OUT / "all_nonTE_concordant_gene_windows.tsv",
    "TE_distant_1kb": OUT / "all_TE_distant_1kb_concordant_gene_windows.tsv"
}

def rank_file(path, label):

    print(f"\nLoading {label}: {path}", flush=True)
    d = pd.read_csv(path, sep="\t")

    print(
        f"  records={len(d):,}; genes={d.gene_id.nunique():,}",
        flush=True
    )

    # --------------------------------------------------------
    # Important: 1%, 5%, 10% tails are nested.
    # Collapse repeated representation of the same physical
    # window/test to the most stringent tail membership.
    # --------------------------------------------------------

    key = [
        "stress","time","context","resolution","direction",
        "chr","start","end","gene_id","feature"
    ]

    d["tail_pct"] = pd.to_numeric(d["tail_pct"])

    d = (
        d.sort_values("tail_pct")
         .drop_duplicates(key, keep="first")
         .copy()
    )

    print(f"  unique window/test records={len(d):,}", flush=True)

    # --------------------------------------------------------
    # Vectorized aggregation
    # --------------------------------------------------------

    group = ["gene_id","stress","direction","feature"]

    basic = (
        d.groupby(group, observed=True)
         .agg(
             n_unique_windows_tests=("gene_id","size"),
             n_contexts=("context","nunique"),
             n_resolutions=("resolution","nunique"),
             n_times=("time","nunique"),
             max_joint_score=("joint_score","max"),
             mean_abs_delta24=("delta24",
                 lambda x: np.mean(np.abs(x))),
             mean_abs_deltaMeth=("deltaMeth",
                 lambda x: np.mean(np.abs(x))),
             min_TE_distance_bp=("nearest_TE_distance_bp","min"),
             min_tail_pct=("tail_pct","min")
         )
         .reset_index()
    )

    # Number of 100- and 500-bp supporting records
    counts = (
        d.assign(
            is100=(d["resolution"] == 100).astype(int),
            is500=(d["resolution"] == 500).astype(int)
        )
        .groupby(group, observed=True)[["is100","is500"]]
        .sum()
        .reset_index()
        .rename(columns={"is100":"n_100bp","is500":"n_500bp"})
    )

    # Compact unique-value strings
    contexts = (
        d.groupby(group, observed=True)["context"]
         .agg(lambda x: ";".join(sorted(set(map(str,x)))))
         .reset_index(name="contexts")
    )

    times = (
        d.groupby(group, observed=True)["time"]
         .agg(lambda x: ";".join(sorted(set(map(str,x)))))
         .reset_index(name="times")
    )

    # Tail support has to come from ORIGINAL table because a 1%
    # window is also evidence that it passes 5% and 10%.
    original = pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "gene_id","stress","direction","feature","tail_pct"
        ]
    )

    tails = (
        original.groupby(group, observed=True)["tail_pct"]
        .agg(
            lambda x: ";".join(
                map(str, sorted(set(pd.to_numeric(x).astype(int))))
            )
        )
        .reset_index(name="tail_support")
    )

    del original

    R = basic.merge(counts, on=group, how="left")
    R = R.merge(contexts, on=group, how="left")
    R = R.merge(times, on=group, how="left")
    R = R.merge(tails, on=group, how="left")

    # --------------------------------------------------------
    # Reproducibility score
    # --------------------------------------------------------

    R["score"] = (
        2.0 * R["n_contexts"] +
        2.0 * R["n_resolutions"] +
        1.5 * R["n_times"] +
        np.log1p(R["n_unique_windows_tests"]) +
        3.0 * R["max_joint_score"]
    )

    # --------------------------------------------------------
    # Tiers
    # --------------------------------------------------------

    R["tier"] = "Tier3"

    tier1 = (
        (R["min_tail_pct"] <= 1) &
        (R["n_resolutions"] == 2) &
        (
            (R["n_times"] >= 2) |
            (R["n_contexts"] >= 2)
        )
    )

    tier2 = (
        ~tier1 &
        (R["min_tail_pct"] <= 5) &
        (
            (R["n_resolutions"] == 2) |
            (R["n_times"] >= 2) |
            (R["n_contexts"] >= 2)
        )
    )

    R.loc[tier1, "tier"] = "Tier1"
    R.loc[tier2, "tier"] = "Tier2"

    tier_order = {"Tier1":0, "Tier2":1, "Tier3":2}
    R["_tier_order"] = R["tier"].map(tier_order)

    R = (
        R.sort_values(
            ["_tier_order","score","max_joint_score"],
            ascending=[True,False,False]
        )
        .drop(columns="_tier_order")
    )

    outfile = OUT / f"{label}_gene_rankings.tsv"
    R.to_csv(outfile, sep="\t", index=False)

    print(f"  ranking rows={len(R):,}", flush=True)
    print(
        "  unique ranked genes="
        f"{R.gene_id.nunique():,}",
        flush=True
    )
    print(
        "  tiers:",
        R.groupby("tier")["gene_id"].nunique().to_dict(),
        flush=True
    )
    print(f"  wrote {outfile}", flush=True)

    return R


rankings = {}

for label, path in inputs.items():
    rankings[label] = rank_file(path, label)


# ------------------------------------------------------------
# Pathway-ready gene sets
# ------------------------------------------------------------

rows = []

for label, R in rankings.items():

    for keys, g in R.groupby(
        ["stress","direction","feature","tier"],
        observed=True
    ):

        stress, direction, feature, tier = keys

        genes = sorted(set(g["gene_id"].astype(str)))

        rows.append({
            "filter": label,
            "stress": stress,
            "direction": direction,
            "feature": feature,
            "tier": tier,
            "n_genes": len(genes),
            "genes": ";".join(genes)
        })

P = pd.DataFrame(rows)

P.to_csv(
    OUT / "pathway_input_gene_sets.tsv",
    sep="\t",
    index=False
)

print(
    "\nWrote pathway_input_gene_sets.tsv",
    flush=True
)

# ------------------------------------------------------------
# Top-gene summary
# ------------------------------------------------------------

for label, R in rankings.items():

    print("\n========================================")
    print(label)
    print("========================================")

    print("\nGenes by stress/direction:")
    print(
        R.groupby(["stress","direction"])["gene_id"]
         .nunique()
         .to_string()
    )

    print("\nTop 30:")
    cols = [
        "gene_id","stress","direction","feature","tier",
        "score","contexts","times","tail_support",
        "n_100bp","n_500bp","min_TE_distance_bp"
    ]

    print(
        R[cols].head(30).to_string(index=False)
    )

print("\nDONE.", flush=True)
