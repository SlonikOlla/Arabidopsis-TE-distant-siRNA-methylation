#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

ROOT = Path.home() / "arabidopsis_siRNA_methylation"
OUT = ROOT / "results/locus_plot_data/AT1G61520"
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# AT1G61520 / LHCA3
# ============================================================
CHR = 1
GENE_START = 22699714
GENE_END   = 22701412
STRAND = "+"

# Tight locus display
XMIN = 22699500
XMAX = 22701500

# Validated recurrent 500-bp loss window
LOSS_START = 22700500
LOSS_END   = 22701000

# ============================================================
# Source matrices
# ============================================================
srna100_file = ROOT / "heat_consensus/results/heat_24nt_100bp_matrix.tsv"
cg100_file   = ROOT / "heat_consensus/results/heat_CG_100bp_matrix.tsv"
chg100_file  = ROOT / "heat_consensus/results/heat_CHG_100bp_matrix.tsv"
chh100_file  = ROOT / "heat_consensus/results/heat_CHH_100bp_matrix.tsv"

cg500_file   = ROOT / "heat_consensus/results/heat_CG_500bp_matrix.tsv"
chg500_file  = ROOT / "heat_consensus/results/heat_CHG_500bp_matrix.tsv"
chh500_file  = ROOT / "heat_consensus/results/heat_CHH_500bp_matrix.tsv"

# ============================================================
# Read + subset
# ============================================================
def read_region(path):
    df = pd.read_csv(path, sep="\t")
    return df[
        (df["chr"].astype(str) == str(CHR)) &
        (df["end"] >= XMIN) &
        (df["start"] <= XMAX)
    ].copy()

srna = read_region(srna100_file)
cg   = read_region(cg100_file)
chg  = read_region(chg100_file)
chh  = read_region(chh100_file)

cg500  = pd.read_csv(cg500_file, sep="\t")
chg500 = pd.read_csv(chg500_file, sep="\t")
chh500 = pd.read_csv(chh500_file, sep="\t")

# Save exact plotting inputs
srna.to_csv(OUT / "AT1G61520_24nt_100bp.tsv", sep="\t", index=False)
cg.to_csv(OUT / "AT1G61520_CG_100bp.tsv", sep="\t", index=False)
chg.to_csv(OUT / "AT1G61520_CHG_100bp.tsv", sep="\t", index=False)
chh.to_csv(OUT / "AT1G61520_CHH_100bp.tsv", sep="\t", index=False)

def midpoint(df):
    return (df["start"] + df["end"]) / 2.0

# ============================================================
# Extract methylation delta for exact 500-bp loss window
# ============================================================
def get_window_deltas(df, context):
    r = df[
        (df["chr"].astype(str) == str(CHR)) &
        (df["start"] == LOSS_START) &
        (df["end"] == LOSS_END)
    ]

    if r.empty:
        return [np.nan, np.nan, np.nan]

    r = r.iloc[0]

    if context == "CG":
        return [
            r["deltaCG_6h"],
            r["deltaCG_12h"],
            r["deltaCG_24h"]
        ]

    elif context == "CHG":
        return [
            r["deltaCHG_6h"],
            r["deltaCHG_12h"],
            r["deltaCHG_24h"]
        ]

    else:
        return [
            r["deltaCHH_6h"],
            r["deltaCHH_12h"],
            r["deltaCHH_24h"]
        ]

loss = {
    "CG":  get_window_deltas(cg500, "CG"),
    "CHG": get_window_deltas(chg500, "CHG"),
    "CHH": get_window_deltas(chh500, "CHH")
}

def fmt(v):
    return "NA" if pd.isna(v) else f"{v:+.3f}"

# ============================================================
# Figure
# ============================================================
fig = plt.figure(figsize=(14.0, 9.5))

gs = fig.add_gridspec(
    5, 2,
    width_ratios=[5.0, 1.65],
    height_ratios=[0.62, 1.2, 1.2, 1.2, 1.2],
    hspace=0.18,
    wspace=0.08
)

ax_gene = fig.add_subplot(gs[0, 0])
ax_srna = fig.add_subplot(gs[1, 0], sharex=ax_gene)
ax_cg   = fig.add_subplot(gs[2, 0], sharex=ax_gene)
ax_chg  = fig.add_subplot(gs[3, 0], sharex=ax_gene)
ax_chh  = fig.add_subplot(gs[4, 0], sharex=ax_gene)
ax_right = fig.add_subplot(gs[:, 1])

ax_right.axis("off")

data_axes = [ax_srna, ax_cg, ax_chg, ax_chh]

# ============================================================
# Shade validated loss window
# ============================================================
for ax in [ax_gene] + data_axes:
    ax.axvspan(
        LOSS_START,
        LOSS_END,
        color="0.80",
        zorder=0
    )

# ============================================================
# Gene boundaries through all data panels
# ============================================================
for ax in data_axes:
    ax.axvline(
        GENE_START,
        color="0.35",
        linestyle="--",
        linewidth=0.9,
        alpha=0.75
    )
    ax.axvline(
        GENE_END,
        color="0.35",
        linestyle="--",
        linewidth=0.9,
        alpha=0.75
    )

# ============================================================
# Gene model
# ============================================================
ax_gene.set_ylim(0, 1)
ax_gene.set_yticks([])
ax_gene.set_xlim(XMIN, XMAX)

gene_y = 0.48

ax_gene.plot(
    [GENE_START, GENE_END],
    [gene_y, gene_y],
    linewidth=6,
    solid_capstyle="butt"
)

ax_gene.annotate(
    "",
    xy=(GENE_END, gene_y),
    xytext=(GENE_END - 250, gene_y),
    arrowprops=dict(
        arrowstyle="-|>",
        lw=2,
        mutation_scale=16
    )
)

ax_gene.text(
    (GENE_START + GENE_END) / 2,
    0.78,
    "AT1G61520 (LHCA3)   + strand",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold"
)

ax_gene.text(
    (LOSS_START + LOSS_END) / 2,
    0.08,
    "Methylation loss",
    ha="center",
    fontsize=9,
    fontweight="bold"
)

for s in ["left", "right", "top", "bottom"]:
    ax_gene.spines[s].set_visible(False)

# ============================================================
# 24-nt siRNA bars
# ============================================================
x = midpoint(srna)

bar_width = 40
offset = 22

ax_srna.bar(
    x - offset,
    srna["CTRLmean"],
    width=bar_width,
    label="Ct",
    alpha=0.80
)

ax_srna.bar(
    x + offset,
    srna["HEATmean"],
    width=bar_width,
    label="Heat",
    alpha=0.80
)

ax_srna.set_ylabel("24-nt siRNA\nCPM")

# siRNA legend can remain in its own panel
ax_srna.legend(
    frameon=False,
    ncol=2,
    loc="upper left"
)

# ============================================================
# Methylation
# ============================================================
def methylation_panel(ax, df, context):

    x = midpoint(df)

    if context in ["CG", "CHG"]:
        cols = [
            ("CTRLmean", "Ct"),
            ("HEAT6mean", "6 h"),
            ("HEAT12mean", "12 h"),
            ("HEAT24mean", "24 h")
        ]
    else:
        cols = [
            ("CTRLmean", "Ct"),
            ("H6mean", "6 h"),
            ("H12mean", "12 h"),
            ("H24mean", "24 h")
        ]

    for col, label in cols:
        ax.plot(
            x,
            df[col],
            marker="o",
            markersize=3.4,
            linewidth=1.35,
            label=label
        )

    ax.set_ylabel(f"{context}\nmethylation")
    ax.set_ylim(-0.03, 1.03)

methylation_panel(ax_cg,  cg,  "CG")
methylation_panel(ax_chg, chg, "CHG")
methylation_panel(ax_chh, chh, "CHH")

# ============================================================
# Axis formatting
# ============================================================
for ax in data_axes:
    ax.grid(axis="y", alpha=0.18)
    ax.set_xlim(XMIN, XMAX)

for ax in [ax_gene, ax_srna, ax_cg, ax_chg]:
    plt.setp(ax.get_xticklabels(), visible=False)

ax_chh.xaxis.set_major_formatter(
    FuncFormatter(lambda value, pos: f"{value/1000:,.1f}")
)

ax_chh.set_xlabel("Chr1 position (kb)")

# ============================================================
# RIGHT SIDE — loss table
# ============================================================
ax_right.text(
    0.5, 0.93,
    "500-bp window summary",
    ha="center",
    va="top",
    fontsize=11,
    fontweight="bold",
    transform=ax_right.transAxes
)

ax_right.text(
    0.5, 0.885,
    "Δ methylation = heat − control",
    ha="center",
    va="top",
    fontsize=9,
    transform=ax_right.transAxes
)

ax_right.text(
    0.5, 0.825,
    "Methylation loss",
    ha="center",
    fontsize=10,
    fontweight="bold",
    transform=ax_right.transAxes
)

ax_right.text(
    0.5, 0.795,
    f"Chr1:{LOSS_START:,}–{LOSS_END:,}",
    ha="center",
    fontsize=8.5,
    transform=ax_right.transAxes
)

loss_data = [
    [ctx] + [fmt(v) for v in loss[ctx]]
    for ctx in ["CG", "CHG", "CHH"]
]

loss_table = ax_right.table(
    cellText=loss_data,
    colLabels=["Context", "6 h", "12 h", "24 h"],
    cellLoc="center",
    colLoc="center",
    bbox=[0.02, 0.52, 0.96, 0.25]
)

loss_table.auto_set_font_size(False)
loss_table.set_fontsize(9)

# ============================================================
# Methylation legend below table
# ============================================================
handles = [
    Line2D([0], [0], marker="o", linewidth=1.5, label="Ct"),
    Line2D([0], [0], marker="o", linewidth=1.5, label="6 h"),
    Line2D([0], [0], marker="o", linewidth=1.5, label="12 h"),
    Line2D([0], [0], marker="o", linewidth=1.5, label="24 h"),
]

ax_right.legend(
    handles=handles,
    labels=["Ct", "6 h", "12 h", "24 h"],
    frameon=True,
    loc="center",
    bbox_to_anchor=(0.50, 0.32),
    fontsize=9
)

# ============================================================
# Title / outputs
# ============================================================
fig.suptitle(
    "AT1G61520 (LHCA3): heat-associated 24-nt siRNA and DNA methylation",
    fontsize=14,
    fontweight="bold",
    y=0.985
)

png = OUT / "AT1G61520_heat_locus_FINAL.png"
pdf = OUT / "AT1G61520_heat_locus_FINAL.pdf"

fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")

print("\nSaved:")
print(png)
print(pdf)

print("\nLOSS WINDOW")
print(f"Chr1:{LOSS_START:,}-{LOSS_END:,}")

for ctx in ["CG", "CHG", "CHH"]:
    print(ctx, [fmt(v) for v in loss[ctx]])

