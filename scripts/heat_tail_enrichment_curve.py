
#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/igor/arabidopsis_siRNA_methylation")
INDIR = ROOT / "results/nonTE_gene_analysis"
OUTDIR = INDIR / "heat_tail_sensitivity"
OUTDIR.mkdir(parents=True, exist_ok=True)

FILES = {
    1: INDIR / "heat_gene_body_window_opportunity_GO_joint1pct_200perm.tsv",
    5: INDIR / "heat_gene_body_window_opportunity_GO_joint5pct_200perm.tsv",
    10: INDIR / "heat_gene_body_window_opportunity_GO_joint10pct_200perm.tsv",
}

TERMS = {
    "GO:0015979": "Photosynthesis",
    "GO:0009765": "Light harvesting",
    "GO:0009408": "Response to heat",
    "GO:0006457": "Protein folding",
}

rows = []

for tail, path in FILES.items():
    df = pd.read_csv(path, sep="\t")

    for go_id, label in TERMS.items():
        x = df[df["GO_ID"] == go_id].copy()

        for _, r in x.iterrows():
            rows.append({
                "tail_pct": tail,
                "context": r["context"],
                "direction": r["direction"],
                "GO_ID": go_id,
                "GO_term_short": label,
                "observed_hits": r["observed_hits"],
                "null_mean_hits": r["null_mean_hits"],
                "fold_vs_window_null": r["fold_vs_window_null"],
                "empirical_p": r["empirical_p"],
                "FDR_BH": r["FDR_BH"],
            })

out = pd.DataFrame(rows)

out.to_csv(
    OUTDIR / "heat_tail_enrichment_summary.tsv",
    sep="\t",
    index=False
)

focus = [
    ("Photosynthesis", "CG", "loss"),
    ("Photosynthesis", "CHG", "loss"),
    ("Photosynthesis", "CHH", "loss"),
    ("Light harvesting", "CG", "loss"),
    ("Light harvesting", "CHG", "loss"),
    ("Light harvesting", "CHH", "loss"),
    ("Response to heat", "CHH", "gain"),
    ("Protein folding", "CHH", "gain"),
]

print("\nHEAT TAIL-SENSITIVITY\n")

for term, ctx, direction in focus:
    x = out[
        (out["GO_term_short"] == term) &
        (out["context"] == ctx) &
        (out["direction"] == direction)
    ].sort_values("tail_pct")

    print(f"{term} | {ctx} {direction}")

    for _, r in x.iterrows():
        print(
            f"  {int(r['tail_pct'])}% "
            f"fold={r['fold_vs_window_null']:.3f} "
            f"obs={int(r['observed_hits'])} "
            f"null={r['null_mean_hits']:.3f} "
            f"P={r['empirical_p']:.4g} "
            f"FDR={r['FDR_BH']:.4g}"
        )

    print()

# Photosynthesis loss
fig, ax = plt.subplots(figsize=(7,5))

for ctx in ["CG","CHG","CHH"]:
    x = out[
        (out["GO_term_short"] == "Photosynthesis") &
        (out["context"] == ctx) &
        (out["direction"] == "loss")
    ].sort_values("tail_pct")

    ax.plot(
        x["tail_pct"],
        x["fold_vs_window_null"],
        marker="o",
        label=ctx
    )

ax.axhline(1, linestyle="--")
ax.set_xticks([1,5,10])
ax.set_xlabel("Joint-tail threshold (%)")
ax.set_ylabel("Fold enrichment vs window-opportunity null")
ax.set_title("Heat: photosynthesis — concordant loss")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(
    OUTDIR / "heat_photosynthesis_loss_tail_curve.png",
    dpi=300
)
plt.close(fig)

# Light harvesting loss
fig, ax = plt.subplots(figsize=(7,5))

for ctx in ["CG","CHG","CHH"]:
    x = out[
        (out["GO_term_short"] == "Light harvesting") &
        (out["context"] == ctx) &
        (out["direction"] == "loss")
    ].sort_values("tail_pct")

    ax.plot(
        x["tail_pct"],
        x["fold_vs_window_null"],
        marker="o",
        label=ctx
    )

ax.axhline(1, linestyle="--")
ax.set_xticks([1,5,10])
ax.set_xlabel("Joint-tail threshold (%)")
ax.set_ylabel("Fold enrichment vs window-opportunity null")
ax.set_title("Heat: light harvesting — concordant loss")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(
    OUTDIR / "heat_light_harvesting_loss_tail_curve.png",
    dpi=300
)
plt.close(fig)

# CHH gain heat-response module
fig, ax = plt.subplots(figsize=(7,5))

for term in ["Response to heat","Protein folding"]:
    x = out[
        (out["GO_term_short"] == term) &
        (out["context"] == "CHH") &
        (out["direction"] == "gain")
    ].sort_values("tail_pct")

    ax.plot(
        x["tail_pct"],
        x["fold_vs_window_null"],
        marker="o",
        label=term
    )

ax.axhline(1, linestyle="--")
ax.set_xticks([1,5,10])
ax.set_xlabel("Joint-tail threshold (%)")
ax.set_ylabel("Fold enrichment vs window-opportunity null")
ax.set_title("Heat: CHH concordant gain")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(
    OUTDIR / "heat_CHH_gain_heat_response_tail_curve.png",
    dpi=300
)
plt.close(fig)

print("Saved results to:", OUTDIR)
