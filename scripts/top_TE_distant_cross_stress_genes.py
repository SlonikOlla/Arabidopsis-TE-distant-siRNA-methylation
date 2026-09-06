import pandas as pd
import numpy as np
from pathlib import Path

IN = Path("results/nonTE_gene_analysis/all_TE_distant_1kb_concordant_gene_windows.tsv")
OUT = Path("results/nonTE_gene_analysis")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN, sep="\t")

# Primary analysis: TE-distant gene bodies only
x = df[
    (df["feature"] == "gene_body") &
    (df["TE_distant_1kb"] == True) &
    (df["TE_overlap"] == False)
].copy()

# ------------------------------------------------------------
# Collapse nested 1/5/10 tail duplication.
# Same biological window/test may appear at multiple tail cutoffs.
# Keep the most extreme representation (smallest tail_pct).
# ------------------------------------------------------------
key = [
    "stress","time","context","resolution","direction",
    "chr","start","end","gene_id"
]

x = (
    x.sort_values(["tail_pct","joint_score"], ascending=[True,False])
     .drop_duplicates(key, keep="first")
)

# ------------------------------------------------------------
# Gene-level summary
#
# We retain several measures:
#   max_joint_score       strongest window
#   mean_top3_joint_score average of up to 3 strongest windows
#   n_supporting_windows  distinct supporting windows/tests
#   contexts              CG/CHG/CHH support
#   resolutions           100/500 bp support
#   times                 stress-specific time support
#
# Ranking prioritizes repeated support, not one isolated window.
# ------------------------------------------------------------

def summarize(g):
    g = g.sort_values("joint_score", ascending=False)
    top3 = g.head(3)

    return pd.Series({
        "max_joint_score": g["joint_score"].max(),
        "mean_top3_joint_score": top3["joint_score"].mean(),
        "n_supporting_windows": len(g),
        "n_contexts": g["context"].nunique(),
        "contexts": ";".join(sorted(g["context"].astype(str).unique())),
        "n_resolutions": g["resolution"].nunique(),
        "resolutions": ";".join(map(str, sorted(g["resolution"].unique()))),
        "n_times": g["time"].nunique(),
        "times": ";".join(sorted(g["time"].astype(str).unique())),
        "max_abs_delta24": g["delta24"].abs().max(),
        "max_abs_deltaMeth": g["deltaMeth"].abs().max(),
        "nearest_TE_distance_bp": g["nearest_TE_distance_bp"].min(),
        "best_chr": g.iloc[0]["chr"],
        "best_start": g.iloc[0]["start"],
        "best_end": g.iloc[0]["end"],
        "best_context": g.iloc[0]["context"],
        "best_resolution": g.iloc[0]["resolution"],
        "best_time": g.iloc[0]["time"],
        "best_delta24": g.iloc[0]["delta24"],
        "best_deltaMeth": g.iloc[0]["deltaMeth"],
        "best_tail_pct": g.iloc[0]["tail_pct"]
    })

summary = (
    x.groupby(["stress","direction","gene_id"], sort=False)
     .apply(summarize, include_groups=False)
     .reset_index()
)

# Composite recurrence-aware ranking.
# joint_score remains primary, with modest bonuses for independent support.
summary["support_score"] = (
    summary["mean_top3_joint_score"] *
    (1 + 0.10*np.log1p(summary["n_supporting_windows"])) *
    (1 + 0.10*(summary["n_contexts"]-1)) *
    (1 + 0.05*(summary["n_resolutions"]-1))
)

summary["rank_within_stress_direction"] = (
    summary.groupby(["stress","direction"])["support_score"]
           .rank(method="first", ascending=False)
           .astype(int)
)

summary = summary.sort_values(
    ["stress","direction","rank_within_stress_direction"]
)

summary.to_csv(
    OUT / "TE_distant_gene_body_cross_stress_rankings.tsv",
    sep="\t", index=False
)

# Top 20 gain and top 20 loss per stress
top = summary[
    summary["rank_within_stress_direction"] <= 20
].copy()

# ------------------------------------------------------------
# Cross-stress recurrence
# Direction-specific and direction-independent
# ------------------------------------------------------------

gene_stresses = (
    summary.groupby("gene_id")["stress"]
           .agg(lambda z: ";".join(sorted(set(z))))
)

gene_nstress = (
    summary.groupby("gene_id")["stress"]
           .nunique()
)

direction_stresses = (
    summary.groupby(["gene_id","direction"])["stress"]
           .agg(lambda z: ";".join(sorted(set(z))))
)

direction_nstress = (
    summary.groupby(["gene_id","direction"])["stress"]
           .nunique()
)

top["all_stresses_for_gene"] = top["gene_id"].map(gene_stresses)
top["n_stresses_for_gene"] = top["gene_id"].map(gene_nstress)

top = top.merge(
    direction_stresses.rename("same_direction_stresses"),
    on=["gene_id","direction"],
    how="left"
)

top = top.merge(
    direction_nstress.rename("n_same_direction_stresses"),
    on=["gene_id","direction"],
    how="left"
)

top.to_csv(
    OUT / "TOP20_TE_distant_genes_each_stress_gain_loss.tsv",
    sep="\t", index=False
)

# ------------------------------------------------------------
# Overlap table using all ranked genes
# ------------------------------------------------------------

overlap = summary[
    summary["gene_id"].map(gene_nstress) >= 2
].copy()

overlap["all_stresses_for_gene"] = overlap["gene_id"].map(gene_stresses)
overlap["n_stresses_for_gene"] = overlap["gene_id"].map(gene_nstress)

overlap = overlap.sort_values(
    ["n_stresses_for_gene","gene_id","stress"],
    ascending=[False,True,True]
)

overlap.to_csv(
    OUT / "TE_distant_cross_stress_overlapping_genes.tsv",
    sep="\t", index=False
)

# ------------------------------------------------------------
# Top-20 overlap: stricter and easier to interpret
# ------------------------------------------------------------

top20_counts = top.groupby("gene_id")["stress"].nunique()

top20_overlap = top[
    top["gene_id"].map(top20_counts) >= 2
].copy()

top20_overlap["n_stresses_top20"] = \
    top20_overlap["gene_id"].map(top20_counts)

top20_overlap = top20_overlap.sort_values(
    ["n_stresses_top20","gene_id","stress"],
    ascending=[False,True,True]
)

top20_overlap.to_csv(
    OUT / "TOP20_cross_stress_overlapping_genes.tsv",
    sep="\t", index=False
)

# ------------------------------------------------------------
# Console summary
# ------------------------------------------------------------

print("\nTOP 10 PER STRESS / DIRECTION")
print("="*80)

showcols = [
    "stress","direction","rank_within_stress_direction",
    "gene_id","support_score","max_joint_score",
    "n_supporting_windows","contexts","resolutions","times"
]

for stress in ["heat","drought","phosphate","pathogen"]:
    for direction in ["gain","loss"]:
        z = summary[
            (summary["stress"] == stress) &
            (summary["direction"] == direction)
        ].head(10)

        print(f"\n{stress.upper()} — {direction.upper()}")
        print(z[showcols].to_string(index=False))

print("\n" + "="*80)
print("TOP-20 GENES RECURRING ACROSS >=2 STRESSES")
print("="*80)

if len(top20_overlap):
    print(top20_overlap[
        ["gene_id","stress","direction",
         "rank_within_stress_direction",
         "n_stresses_top20",
         "all_stresses_for_gene"]
    ].to_string(index=False))
else:
    print("No genes recur in the top 20 of >=2 stresses.")

print("\nSaved:")
print(OUT / "TE_distant_gene_body_cross_stress_rankings.tsv")
print(OUT / "TOP20_TE_distant_genes_each_stress_gain_loss.tsv")
print(OUT / "TE_distant_cross_stress_overlapping_genes.tsv")
print(OUT / "TOP20_cross_stress_overlapping_genes.tsv")
