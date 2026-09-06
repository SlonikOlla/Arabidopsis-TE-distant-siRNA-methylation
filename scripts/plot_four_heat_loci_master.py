#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path.home() / "arabidopsis_siRNA_methylation"
OUT = ROOT / "results/locus_plot_data/master_four_loci"
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# INPUTS
# ============================================================

FILES = {
    "srna100": ROOT / "heat_consensus/results/heat_24nt_100bp_matrix.tsv",

    "CG100":  ROOT / "heat_consensus/results/heat_CG_100bp_matrix.tsv",
    "CHG100": ROOT / "heat_consensus/results/heat_CHG_100bp_matrix.tsv",
    "CHH100": ROOT / "heat_consensus/results/heat_CHH_100bp_matrix.tsv",

    "CG500":  ROOT / "heat_consensus/results/heat_CG_500bp_matrix.tsv",
    "CHG500": ROOT / "heat_consensus/results/heat_CHG_500bp_matrix.tsv",
    "CHH500": ROOT / "heat_consensus/results/heat_CHH_500bp_matrix.tsv",
}

# ============================================================
# LOCUS DEFINITIONS
# ============================================================

LOCI = {
    "AT1G29930": {
        "symbol": "CAB1/LHCB1.3",
        "chr": 1,
        "start": 10477884,
        "end": 10479114,
        "strand": "+",
        "xmin": 10477000,
        "xmax": 10479500,
        "windows": [
            {
                "start": 10478000,
                "end": 10478500,
                "resolution": 500,
                "type": "gain",
                "label": "Heat-associated gain"
            },
            {
                "start": 10478500,
                "end": 10479000,
                "resolution": 500,
                "type": "loss",
                "label": "Recurrent heat-associated loss"
            },
        ],
        "table_windows": [
            ("gain", 10478000, 10478500),
            ("loss", 10478500, 10479000),
        ]
    },

    "AT1G61520": {
        "symbol": "LHCA3",
        "chr": 1,
        "start": 22699714,
        "end": 22701412,
        "strand": "+",
        "xmin": 22699500,
        "xmax": 22701500,
        "windows": [
            {
                "start": 22700500,
                "end": 22701000,
                "resolution": 500,
                "type": "loss",
                "label": "Recurrent heat-associated loss"
            },
        ],
        "table_windows": [
            ("loss", 22700500, 22701000),
        ]
    },

    "AT1G67090": {
        "symbol": "RBCS1A",
        "chr": 1,
        "start": 25048157,
        "end": 25049667,
        "strand": "-",
        "xmin": 25048000,
        "xmax": 25050000,
        "windows": [
            {
                "start": 25048500,
                "end": 25049000,
                "resolution": 500,
                "type": "loss",
                "label": "Recurrent heat-associated loss"
            },
        ],
        "table_windows": [
            ("loss", 25048500, 25049000),
        ]
    },

    "AT1G74470": {
        "symbol": "",
        "chr": 1,
        "start": 27991164,
        "end": 27993445,
        "strand": "+",
        "xmin": 27991000,
        "xmax": 27993500,
        "windows": [
            {
                "start": 27991500,
                "end": 27992000,
                "resolution": 500,
                "type": "loss",
                "label": "Recurrent heat-associated loss"
            },
            {
                "start": 27991600,
                "end": 27991700,
                "resolution": 100,
                "type": "core",
                "label": "100-bp recurrent core"
            },
        ],
        "table_windows": [
            ("loss", 27991500, 27992000),
        ]
    }
}

# ============================================================
# COLORS
# ============================================================

COLORS = {
    "Ct":   "C0",
    "Heat": "C1",
    "6 h":  "C1",
    "12 h": "C2",
    "24 h": "C3",
}

# ============================================================
# READ MATRICES
# ============================================================

DATA = {k: pd.read_csv(v, sep="\t") for k, v in FILES.items()}

def chr_match(df, chrom):
    return df["chr"].astype(str) == str(chrom)

def subset(df, cfg):
    return df[
        chr_match(df, cfg["chr"]) &
        (df["end"] >= cfg["xmin"]) &
        (df["start"] <= cfg["xmax"])
    ].copy()

def midpoint(df):
    return (df["start"] + df["end"]) / 2

# ============================================================
# DELTA FUNCTIONS
# ============================================================

def get_500_delta(context, chrom, start, end):
    df = DATA[f"{context}500"]

    r = df[
        chr_match(df, chrom) &
        (df["start"] == start) &
        (df["end"] == end)
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

    if context == "CHG":
        return [
            r["deltaCHG_6h"],
            r["deltaCHG_12h"],
            r["deltaCHG_24h"]
        ]

    return [
        r["deltaCHH_6h"],
        r["deltaCHH_12h"],
        r["deltaCHH_24h"]
    ]

def get_100_core(context, chrom, start, end):
    df = DATA[f"{context}100"]

    r = df[
        chr_match(df, chrom) &
        (df["start"] == start) &
        (df["end"] == end)
    ]

    if r.empty:
        return None

    r = r.iloc[0]

    if context == "CG":
        return {
            "control": r["CTRLmean"],
            "6h": r["HEAT6mean"],
            "12h": r["HEAT12mean"],
            "24h": r["HEAT24mean"],
            "d6": r["deltaCG_6h"],
            "d12": r["deltaCG_12h"],
            "d24": r["deltaCG_24h"]
        }

    if context == "CHG":
        return {
            "control": r["CTRLmean"],
            "6h": r["HEAT6mean"],
            "12h": r["HEAT12mean"],
            "24h": r["HEAT24mean"],
            "d6": r["deltaCHG_6h"],
            "d12": r["deltaCHG_12h"],
            "d24": r["deltaCHG_24h"]
        }

    return {
        "control": r["CTRLmean"],
        "6h": r["H6mean"],
        "12h": r["H12mean"],
        "24h": r["H24mean"],
        "d6": r["deltaCHH_6h"],
        "d12": r["deltaCHH_12h"],
        "d24": r["deltaCHH_24h"]
    }

def fmt(v):
    return "NA" if pd.isna(v) else f"{v:+.3f}"

# ============================================================
# PLOT HELPERS
# ============================================================

def format_kb(x, pos):
    return f"{x/1000:,.1f}"

def add_window_shading(ax, cfg):
    for w in cfg["windows"]:
        if w["type"] == "gain":
            ax.axvspan(w["start"], w["end"], color="0.90", zorder=0)

        elif w["type"] == "loss":
            ax.axvspan(w["start"], w["end"], color="0.82", zorder=0)

        elif w["type"] == "core":
            ax.axvspan(w["start"], w["end"], color="0.65", zorder=0)

def draw_gene(ax, gene, cfg):
    ax.set_xlim(cfg["xmin"], cfg["xmax"])
    ax.set_ylim(0, 1)
    ax.set_yticks([])

    add_window_shading(ax, cfg)

    y = 0.52

    ax.plot(
        [cfg["start"], cfg["end"]],
        [y, y],
        linewidth=7,
        solid_capstyle="butt"
    )

    if cfg["strand"] == "+":
        ax.annotate(
            "",
            xy=(cfg["end"], y),
            xytext=(cfg["end"] - 220, y),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=2.0,
                mutation_scale=18
            )
        )
    else:
        ax.annotate(
            "",
            xy=(cfg["start"], y),
            xytext=(cfg["start"] + 220, y),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=2.0,
                mutation_scale=18
            )
        )

    symbol = f" ({cfg['symbol']})" if cfg["symbol"] else ""

    ax.text(
        (cfg["start"] + cfg["end"]) / 2,
        0.82,
        f"{gene}{symbol}    {cfg['strand']} strand",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold"
    )

    ax.text(
        cfg["start"],
        0.97,
        f"{cfg['start']:,}",
        ha="center",
        va="bottom",
        fontsize=8
    )

    ax.text(
        cfg["end"],
        0.97,
        f"{cfg['end']:,}",
        ha="center",
        va="bottom",
        fontsize=8
    )

    for w in cfg["windows"]:
        if w["type"] == "gain":
            ax.text(
                (w["start"] + w["end"]) / 2,
                0.14,
                "Heat-associated gain\n(500 bp)",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold"
            )

        elif w["type"] == "loss":
            txt = "Recurrent heat-associated\nloss window"
            if w["resolution"] == 500:
                txt += "\n(500 bp)"

            ax.text(
                (w["start"] + w["end"]) / 2,
                0.14,
                txt,
                ha="center",
                va="center",
                fontsize=8.3,
                fontweight="bold"
            )

        elif w["type"] == "core":
            ax.text(
                (w["start"] + w["end"]) / 2,
                0.04,
                "100-bp\ncore",
                ha="center",
                va="bottom",
                fontsize=7.3,
                fontweight="bold"
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

def draw_srna(ax, cfg):
    df = subset(DATA["srna100"], cfg)
    x = midpoint(df)

    add_window_shading(ax, cfg)

    width = 40
    offset = 22

    ax.bar(
        x - offset,
        df["CTRLmean"],
        width=width,
        color=COLORS["Ct"],
        alpha=0.82,
        label="Ct"
    )

    ax.bar(
        x + offset,
        df["HEATmean"],
        width=width,
        color=COLORS["Heat"],
        alpha=0.82,
        label="Heat"
    )

    ax.set_ylabel("24-nt siRNA\n(CPM)")
    ax.grid(axis="y", alpha=0.20)
    ax.set_xlim(cfg["xmin"], cfg["xmax"])

    ax.axvline(cfg["start"], linestyle="--", linewidth=0.8, color="0.35", alpha=0.7)
    ax.axvline(cfg["end"], linestyle="--", linewidth=0.8, color="0.35", alpha=0.7)

def draw_methylation(ax, cfg, context):
    df = subset(DATA[f"{context}100"], cfg)
    x = midpoint(df)

    add_window_shading(ax, cfg)

    if context in ("CG", "CHG"):
        cols = [
            ("CTRLmean", "Ct"),
            ("HEAT6mean", "6 h"),
            ("HEAT12mean", "12 h"),
            ("HEAT24mean", "24 h"),
        ]
    else:
        cols = [
            ("CTRLmean", "Ct"),
            ("H6mean", "6 h"),
            ("H12mean", "12 h"),
            ("H24mean", "24 h"),
        ]

    for col, label in cols:
        ax.plot(
            x,
            df[col],
            marker="o",
            markersize=3.6,
            linewidth=1.35,
            color=COLORS[label],
            label=label
        )

    ax.axvline(cfg["start"], linestyle="--", linewidth=0.8, color="0.35", alpha=0.7)
    ax.axvline(cfg["end"], linestyle="--", linewidth=0.8, color="0.35", alpha=0.7)

    ax.set_ylabel(f"{context}\nmethylation")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(cfg["xmin"], cfg["xmax"])
    ax.grid(axis="y", alpha=0.20)

def draw_right_panel(ax, gene, cfg):
    ax.axis("off")

    ax.text(
        0.5, 0.97,
        "500-bp window summary",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        transform=ax.transAxes
    )

    ax.text(
        0.5, 0.925,
        "Δ methylation = heat − control",
        ha="center",
        va="top",
        fontsize=8.7,
        transform=ax.transAxes
    )

    y = 0.84

    for typ, start, end in cfg["table_windows"]:

        heading = (
            "Heat-associated gain"
            if typ == "gain"
            else "Recurrent heat-associated loss"
        )

        ax.text(
            0.5, y,
            heading,
            ha="center",
            fontsize=9.6,
            fontweight="bold",
            transform=ax.transAxes
        )

        ax.text(
            0.5, y - 0.035,
            f"Chr1:{start:,}–{end:,}",
            ha="center",
            fontsize=8,
            transform=ax.transAxes
        )

        rows = []
        for context in ["CG", "CHG", "CHH"]:
            vals = get_500_delta(context, cfg["chr"], start, end)
            rows.append([context] + [fmt(v) for v in vals])

        table = ax.table(
            cellText=rows,
            colLabels=["Context", "6 h", "12 h", "24 h"],
            cellLoc="center",
            colLoc="center",
            bbox=[0.03, y - 0.26, 0.94, 0.19]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.3)

        y -= 0.34

    handles = [
        Line2D([0], [0], color=COLORS["Ct"], marker="o", lw=1.5, label="Ct"),
        Line2D([0], [0], color=COLORS["6 h"], marker="o", lw=1.5, label="6 h"),
        Line2D([0], [0], color=COLORS["12 h"], marker="o", lw=1.5, label="12 h"),
        Line2D([0], [0], color=COLORS["24 h"], marker="o", lw=1.5, label="24 h"),
    ]

    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08),
        frameon=True,
        fontsize=8.7
    )

    if gene == "AT1G74470":
        ax.text(
            0.5, 0.25,
            "Exact 100-bp recurrent core",
            ha="center",
            fontsize=9,
            fontweight="bold",
            transform=ax.transAxes
        )

        ax.text(
            0.5, 0.215,
            "Chr1:27,991,600–27,991,700",
            ha="center",
            fontsize=8,
            transform=ax.transAxes
        )

        ax.text(
            0.5, 0.18,
            "12 h: CG + CHG + CHH",
            ha="center",
            fontsize=8,
            transform=ax.transAxes
        )

# ============================================================
# INDIVIDUAL FULL FIGURES
# ============================================================

def make_full_figure(gene, cfg):

    fig = plt.figure(figsize=(14.2, 9.3))

    gs = fig.add_gridspec(
        5, 2,
        width_ratios=[5.0, 1.7],
        height_ratios=[0.65, 1.18, 1.15, 1.15, 1.15],
        hspace=0.15,
        wspace=0.08
    )

    ax_gene = fig.add_subplot(gs[0, 0])
    ax_srna = fig.add_subplot(gs[1, 0], sharex=ax_gene)
    ax_cg = fig.add_subplot(gs[2, 0], sharex=ax_gene)
    ax_chg = fig.add_subplot(gs[3, 0], sharex=ax_gene)
    ax_chh = fig.add_subplot(gs[4, 0], sharex=ax_gene)
    ax_right = fig.add_subplot(gs[:, 1])

    draw_gene(ax_gene, gene, cfg)
    draw_srna(ax_srna, cfg)
    draw_methylation(ax_cg, cfg, "CG")
    draw_methylation(ax_chg, cfg, "CHG")
    draw_methylation(ax_chh, cfg, "CHH")
    draw_right_panel(ax_right, gene, cfg)

    for ax in [ax_gene, ax_srna, ax_cg, ax_chg]:
        plt.setp(ax.get_xticklabels(), visible=False)

    ax_chh.xaxis.set_major_formatter(FuncFormatter(format_kb))
    ax_chh.set_xlabel("Chr1 position (kb)")

    ax_srna.legend(
        frameon=False,
        ncol=2,
        loc="upper left",
        fontsize=8.7
    )

    symbol = f" ({cfg['symbol']})" if cfg["symbol"] else ""

    fig.suptitle(
        f"{gene}{symbol}: heat-associated 24-nt siRNA and DNA methylation",
        fontsize=14,
        fontweight="bold",
        y=0.985
    )

    png = OUT / f"{gene}_heat_locus_MASTER.png"
    pdf = OUT / f"{gene}_heat_locus_MASTER.pdf"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# QC TABLE
# ============================================================

qc_rows = []

for gene, cfg in LOCI.items():

    srna = subset(DATA["srna100"], cfg)

    for w in cfg["windows"]:

        if w["resolution"] == 500:
            for context in ["CG", "CHG", "CHH"]:
                vals = get_500_delta(
                    context,
                    cfg["chr"],
                    w["start"],
                    w["end"]
                )

                qc_rows.append({
                    "gene": gene,
                    "symbol": cfg["symbol"],
                    "strand": cfg["strand"],
                    "window_type": w["type"],
                    "resolution_bp": 500,
                    "chr": cfg["chr"],
                    "start": w["start"],
                    "end": w["end"],
                    "context": context,
                    "delta_6h": vals[0],
                    "delta_12h": vals[1],
                    "delta_24h": vals[2],
                    "CTRL_24nt_CPM": np.nan,
                    "HEAT_24nt_CPM": np.nan,
                    "delta24_CPM": np.nan
                })

        elif w["resolution"] == 100 and w["type"] == "core":

            sr = srna[
                (srna["start"] == w["start"]) &
                (srna["end"] == w["end"])
            ]

            if sr.empty:
                ctrl24 = heat24 = d24 = np.nan
            else:
                sr = sr.iloc[0]
                ctrl24 = sr["CTRLmean"]
                heat24 = sr["HEATmean"]
                d24 = sr["delta24_CPM"]

            for context in ["CG", "CHG", "CHH"]:
                m = get_100_core(
                    context,
                    cfg["chr"],
                    w["start"],
                    w["end"]
                )

                qc_rows.append({
                    "gene": gene,
                    "symbol": cfg["symbol"],
                    "strand": cfg["strand"],
                    "window_type": "exact_100bp_core",
                    "resolution_bp": 100,
                    "chr": cfg["chr"],
                    "start": w["start"],
                    "end": w["end"],
                    "context": context,
                    "delta_6h": m["d6"],
                    "delta_12h": m["d12"],
                    "delta_24h": m["d24"],
                    "CTRL_24nt_CPM": ctrl24,
                    "HEAT_24nt_CPM": heat24,
                    "delta24_CPM": d24
                })

qc = pd.DataFrame(qc_rows)

qc.to_csv(
    OUT / "four_heat_loci_window_QC.tsv",
    sep="\t",
    index=False
)

# ============================================================
# GENERATE INDIVIDUAL FIGURES
# ============================================================

for gene, cfg in LOCI.items():
    print("Plotting", gene)
    make_full_figure(gene, cfg)

print("\nSaved master figures and QC table to:")
print(OUT)
