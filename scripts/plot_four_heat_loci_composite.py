#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Global publication-size fonts
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
})

ROOT = Path.home() / "arabidopsis_siRNA_methylation"
OUT = ROOT / "results/locus_plot_data/master_four_loci"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

srna = pd.read_csv(
    ROOT / "heat_consensus/results/heat_24nt_100bp_matrix.tsv",
    sep="\t"
)

CG = pd.read_csv(
    ROOT / "heat_consensus/results/heat_CG_100bp_matrix.tsv",
    sep="\t"
)

CHG = pd.read_csv(
    ROOT / "heat_consensus/results/heat_CHG_100bp_matrix.tsv",
    sep="\t"
)

CHH = pd.read_csv(
    ROOT / "heat_consensus/results/heat_CHH_100bp_matrix.tsv",
    sep="\t"
)

DATA = {
    "CG": CG,
    "CHG": CHG,
    "CHH": CHH
}

# ------------------------------------------------------------
# Locus definitions
# ------------------------------------------------------------

LOCI = [
    {
        "gene": "AT1G29930",
        "symbol": "CAB1/LHCB1.3",
        "strand": "+",
        "start": 10477884,
        "end": 10479114,
        "xmin": 10477000,
        "xmax": 10479500,
        "windows": [
            (10478000, 10478500, "gain"),
            (10478500, 10479000, "loss")
        ]
    },

    {
        "gene": "AT1G61520",
        "symbol": "LHCA3",
        "strand": "+",
        "start": 22699714,
        "end": 22701412,
        "xmin": 22699500,
        "xmax": 22701500,
        "windows": [
            (22700500, 22701000, "loss")
        ]
    },

    {
        "gene": "AT1G67090",
        "symbol": "RBCS1A",
        "strand": "-",
        "start": 25048157,
        "end": 25049667,
        "xmin": 25048000,
        "xmax": 25050000,
        "windows": [
            (25048500, 25049000, "loss")
        ]
    },

    {
        "gene": "AT1G74470",
        "symbol": "",
        "strand": "+",
        "start": 27991164,
        "end": 27993445,
        "xmin": 27991000,
        "xmax": 27993500,
        "windows": [
            (27991500, 27992000, "loss"),
            (27991600, 27991700, "core")
        ]
    }
]

COLORS = {
    "Ct": "C0",
    "Heat": "C1",
    "6 h": "C1",
    "12 h": "C2",
    "24 h": "C3"
}

PANEL = ["A", "B", "C", "D"]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def region(df, cfg):
    return df[
        (df["chr"].astype(str) == "1") &
        (df["end"] >= cfg["xmin"]) &
        (df["start"] <= cfg["xmax"])
    ].copy()

def midpoint(df):
    return (df["start"] + df["end"]) / 2.0

def kb_formatter(x, pos):
    return f"{x/1000:,.1f}"

def shade(ax, cfg):
    for start, end, typ in cfg["windows"]:

        if typ == "gain":
            ax.axvspan(
                start, end,
                color="0.91",
                zorder=0
            )

        elif typ == "loss":
            ax.axvspan(
                start, end,
                color="0.82",
                zorder=0
            )

        elif typ == "core":
            ax.axvspan(
                start, end,
                color="0.62",
                zorder=0
            )

def gene_boundaries(ax, cfg):
    ax.axvline(
        cfg["start"],
        linestyle="--",
        color="0.35",
        linewidth=0.75,
        alpha=0.65
    )

    ax.axvline(
        cfg["end"],
        linestyle="--",
        color="0.35",
        linewidth=0.75,
        alpha=0.65
    )

# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------

fig = plt.figure(figsize=(15.5, 12.7))

outer = fig.add_gridspec(
    2, 2,
    left=0.075,
    right=0.955,
    bottom=0.185,
    top=0.94,
    wspace=0.16,
    hspace=0.20
)

for i, cfg in enumerate(LOCI):

    row = i // 2
    col = i % 2

    inner = outer[row, col].subgridspec(
        5, 1,
        height_ratios=[0.62, 1.0, 1.0, 1.0, 1.0],
        hspace=0.08
    )

    ax_gene = fig.add_subplot(inner[0])
    ax_si   = fig.add_subplot(inner[1], sharex=ax_gene)
    ax_cg   = fig.add_subplot(inner[2], sharex=ax_gene)
    ax_chg  = fig.add_subplot(inner[3], sharex=ax_gene)
    ax_chh  = fig.add_subplot(inner[4], sharex=ax_gene)

    axes = [ax_gene, ax_si, ax_cg, ax_chg, ax_chh]

    for ax in axes:
        ax.set_xlim(cfg["xmin"], cfg["xmax"])
        shade(ax, cfg)

    # --------------------------------------------------------
    # Gene model
    # --------------------------------------------------------

    ax_gene.set_ylim(0, 1)
    ax_gene.set_yticks([])

    y = 0.48

    ax_gene.plot(
        [cfg["start"], cfg["end"]],
        [y, y],
        linewidth=6,
        solid_capstyle="butt"
    )

    if cfg["strand"] == "+":
        ax_gene.annotate(
            "",
            xy=(cfg["end"], y),
            xytext=(cfg["end"] - 180, y),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.8,
                mutation_scale=15
            )
        )
    else:
        ax_gene.annotate(
            "",
            xy=(cfg["start"], y),
            xytext=(cfg["start"] + 180, y),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.8,
                mutation_scale=15
            )
        )

    symbol = f" ({cfg['symbol']})" if cfg["symbol"] else ""

    ax_gene.text(
        0.01,
        0.90,
        f"{PANEL[i]}   {cfg['gene']}{symbol}",
        transform=ax_gene.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold"
    )

    ax_gene.text(
        0.99,
        0.90,
        f"{cfg['strand']} strand",
        transform=ax_gene.transAxes,
        ha="right",
        va="top",
        fontsize=11
    )

    # Window labels
    for start, end, typ in cfg["windows"]:

        if typ == "gain":
            ax_gene.text(
                (start + end) / 2,
                0.10,
                "gain",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

        elif typ == "loss":

            # AT1G74470 contains a narrow 100-bp core,
            # so shift the broad loss label slightly right.
            if cfg["gene"] == "AT1G74470":
                xpos = start + 0.70 * (end - start)
            else:
                xpos = (start + end) / 2

            ax_gene.text(
                xpos,
                0.10,
                "recurrent loss",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

        elif typ == "core":
            # Narrow 100-bp core is shown by the darker band.
            # Do not label it in the gene strip because it
            # collides with the AT1G74470 title.
            pass

    for spine in ax_gene.spines.values():
        spine.set_visible(False)

    # --------------------------------------------------------
    # siRNA
    # --------------------------------------------------------

    d = region(srna, cfg)
    x = midpoint(d)

    ax_si.bar(
        x - 22,
        d["CTRLmean"],
        width=40,
        color=COLORS["Ct"],
        alpha=0.82
    )

    ax_si.bar(
        x + 22,
        d["HEATmean"],
        width=40,
        color=COLORS["Heat"],
        alpha=0.82
    )

    ax_si.set_ylabel(
        "24-nt siRNA\nCPM",
        fontsize=17,
        fontweight="bold"
    )

    if cfg["gene"] == "AT1G74470":
        ymax = ax_si.get_ylim()[1]
        ax_si.text(
            (27991600 + 27991700) / 2,
            ymax * 0.93,
            "100-bp core",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            rotation=90
        )

    gene_boundaries(ax_si, cfg)

    # --------------------------------------------------------
    # DNA methylation
    # --------------------------------------------------------

    methyl_axes = {
        "CG": ax_cg,
        "CHG": ax_chg,
        "CHH": ax_chh
    }

    for context, ax in methyl_axes.items():

        d = region(DATA[context], cfg)
        x = midpoint(d)

        if context in ["CG", "CHG"]:
            columns = [
                ("CTRLmean", "Ct"),
                ("HEAT6mean", "6 h"),
                ("HEAT12mean", "12 h"),
                ("HEAT24mean", "24 h")
            ]
        else:
            columns = [
                ("CTRLmean", "Ct"),
                ("H6mean", "6 h"),
                ("H12mean", "12 h"),
                ("H24mean", "24 h")
            ]

        for column, label in columns:

            ax.plot(
                x,
                d[column],
                color=COLORS[label],
                marker="o",
                markersize=2.8,
                linewidth=1.15
            )

        ax.set_ylim(-0.03, 1.03)

        ax.set_ylabel(
            context,
            fontsize=17,
            fontweight="bold",
            labelpad=8
        )

        gene_boundaries(ax, cfg)

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    for ax in [ax_si, ax_cg, ax_chg, ax_chh]:
        ax.grid(
            axis="y",
            alpha=0.18
        )

        # Explicit publication-size numerical tick labels
        ax.tick_params(
            axis="x",
            which="major",
            labelsize=16,
            width=1.2,
            length=5
        )

        ax.tick_params(
            axis="y",
            which="major",
            labelsize=16,
            width=1.2,
            length=5
        )

        # Force font size in case rcParams are overridden
        for label in ax.get_xticklabels():
            label.set_fontsize(16)

        for label in ax.get_yticklabels():
            label.set_fontsize(16)

        # Fewer ticks = more room for publication-size labels
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    for ax in [ax_gene, ax_si, ax_cg, ax_chg]:
        plt.setp(
            ax.get_xticklabels(),
            visible=False
        )

    ax_chh.xaxis.set_major_locator(
        MaxNLocator(nbins=6)
    )
    ax_chh.xaxis.set_major_formatter(
        FuncFormatter(kb_formatter)
    )

    ax_chh.set_xlabel(
        "Chr1 position (kb)",
        fontsize=17,
        fontweight="bold"
    )

# ------------------------------------------------------------
# Common legends
# ------------------------------------------------------------

srna_handles = [
    plt.Rectangle(
        (0, 0), 1, 1,
        color=COLORS["Ct"],
        alpha=0.82,
        label="siRNA control"
    ),
    plt.Rectangle(
        (0, 0), 1, 1,
        color=COLORS["Heat"],
        alpha=0.82,
        label="siRNA heat"
    )
]

meth_handles = [
    Line2D(
        [0], [0],
        color=COLORS["Ct"],
        marker="o",
        lw=1.4,
        label="Control"
    ),
    Line2D(
        [0], [0],
        color=COLORS["6 h"],
        marker="o",
        lw=1.4,
        label="6 h"
    ),
    Line2D(
        [0], [0],
        color=COLORS["12 h"],
        marker="o",
        lw=1.4,
        label="12 h"
    ),
    Line2D(
        [0], [0],
        color=COLORS["24 h"],
        marker="o",
        lw=1.4,
        label="24 h"
    )
]

# One common legend in a dedicated bottom band
all_handles = srna_handles + meth_handles

fig.legend(
    handles=all_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.075),
    ncol=6,
    frameon=False,
    fontsize=15,
    columnspacing=1.8,
    handletextpad=0.6
)

fig.suptitle(
    "Recurrent heat-associated 24-nt siRNA and DNA methylation changes at four Arabidopsis loci",
    fontsize=20,
    fontweight="bold",
    y=0.978
)

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

png = OUT / "four_heat_loci_COMPOSITE.png"
pdf = OUT / "four_heat_loci_COMPOSITE.pdf"

fig.savefig(
    png,
    dpi=300
)

fig.savefig(
    pdf
)

plt.close(fig)

print("Saved:")
print(png)
print(pdf)
