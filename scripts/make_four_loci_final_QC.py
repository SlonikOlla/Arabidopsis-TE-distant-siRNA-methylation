#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.home() / "arabidopsis_siRNA_methylation"
OUT = ROOT / "results/locus_plot_data/master_four_loci"

srna100 = pd.read_csv(
    ROOT / "heat_consensus/results/heat_24nt_100bp_matrix.tsv",
    sep="\t"
)

m500 = {
    "CG": pd.read_csv(
        ROOT / "heat_consensus/results/heat_CG_500bp_matrix.tsv",
        sep="\t"
    ),
    "CHG": pd.read_csv(
        ROOT / "heat_consensus/results/heat_CHG_500bp_matrix.tsv",
        sep="\t"
    ),
    "CHH": pd.read_csv(
        ROOT / "heat_consensus/results/heat_CHH_500bp_matrix.tsv",
        sep="\t"
    ),
}

m100 = {
    "CG": pd.read_csv(
        ROOT / "heat_consensus/results/heat_CG_100bp_matrix.tsv",
        sep="\t"
    ),
    "CHG": pd.read_csv(
        ROOT / "heat_consensus/results/heat_CHG_100bp_matrix.tsv",
        sep="\t"
    ),
    "CHH": pd.read_csv(
        ROOT / "heat_consensus/results/heat_CHH_100bp_matrix.tsv",
        sep="\t"
    ),
}

WINDOWS = [
    ("AT1G29930", "CAB1/LHCB1.3", "+", "gain",
     1, 10478000, 10478500),

    ("AT1G29930", "CAB1/LHCB1.3", "+", "loss",
     1, 10478500, 10479000),

    ("AT1G61520", "LHCA3", "+", "loss",
     1, 22700500, 22701000),

    ("AT1G67090", "RBCS1A", "-", "loss",
     1, 25048500, 25049000),

    ("AT1G74470", "", "+", "loss",
     1, 27991500, 27992000),
]

def methyl_500(context, chrom, start, end):

    df = m500[context]

    r = df[
        (df["chr"].astype(str) == str(chrom)) &
        (df["start"] == start) &
        (df["end"] == end)
    ]

    if r.empty:
        return [np.nan] * 7

    r = r.iloc[0]

    if context in ("CG", "CHG"):
        return [
            r["CTRLmean"],
            r["HEAT6mean"],
            r["HEAT12mean"],
            r["HEAT24mean"],
            r[f"delta{context}_6h"],
            r[f"delta{context}_12h"],
            r[f"delta{context}_24h"],
        ]

    return [
        r["CTRLmean"],
        r["H6mean"],
        r["H12mean"],
        r["H24mean"],
        r["deltaCHH_6h"],
        r["deltaCHH_12h"],
        r["deltaCHH_24h"],
    ]

rows = []

for gene, symbol, strand, wtype, chrom, start, end in WINDOWS:

    # Sum constituent 100-bp CPM bins across each 500-bp interval
    s = srna100[
        (srna100["chr"].astype(str) == str(chrom)) &
        (srna100["start"] >= start) &
        (srna100["end"] <= end)
    ].copy()

    ctrl24 = s["CTRLmean"].sum()
    heat24 = s["HEATmean"].sum()
    d24 = heat24 - ctrl24

    for context in ["CG", "CHG", "CHH"]:

        (
            ctrl_m,
            h6,
            h12,
            h24,
            d6,
            d12,
            d24m
        ) = methyl_500(context, chrom, start, end)

        rows.append({
            "gene": gene,
            "symbol": symbol,
            "strand": strand,
            "window_type": wtype,
            "resolution_bp": 500,
            "chr": chrom,
            "start": start,
            "end": end,
            "context": context,

            "CTRL_methylation": ctrl_m,
            "HEAT6_methylation": h6,
            "HEAT12_methylation": h12,
            "HEAT24_methylation": h24,

            "deltaMeth_6h": d6,
            "deltaMeth_12h": d12,
            "deltaMeth_24h": d24m,

            "n_100bp_sRNA_bins": len(s),
            "CTRL_24nt_CPM_sum": ctrl24,
            "HEAT_24nt_CPM_sum": heat24,
            "delta24_CPM_sum": d24
        })

# ------------------------------------------------------------
# Exact AT1G74470 100-bp recurrent core
# ------------------------------------------------------------

core_start = 27991600
core_end = 27991700

s = srna100[
    (srna100["chr"].astype(str) == "1") &
    (srna100["start"] == core_start) &
    (srna100["end"] == core_end)
].iloc[0]

for context in ["CG", "CHG", "CHH"]:

    d = m100[context]

    r = d[
        (d["chr"].astype(str) == "1") &
        (d["start"] == core_start) &
        (d["end"] == core_end)
    ].iloc[0]

    if context in ("CG", "CHG"):
        ctrl = r["CTRLmean"]
        h6   = r["HEAT6mean"]
        h12  = r["HEAT12mean"]
        h24  = r["HEAT24mean"]

        d6  = r[f"delta{context}_6h"]
        d12 = r[f"delta{context}_12h"]
        d24 = r[f"delta{context}_24h"]

    else:
        ctrl = r["CTRLmean"]
        h6   = r["H6mean"]
        h12  = r["H12mean"]
        h24  = r["H24mean"]

        d6  = r["deltaCHH_6h"]
        d12 = r["deltaCHH_12h"]
        d24 = r["deltaCHH_24h"]

    rows.append({
        "gene": "AT1G74470",
        "symbol": "",
        "strand": "+",
        "window_type": "exact_100bp_core",
        "resolution_bp": 100,
        "chr": 1,
        "start": core_start,
        "end": core_end,
        "context": context,

        "CTRL_methylation": ctrl,
        "HEAT6_methylation": h6,
        "HEAT12_methylation": h12,
        "HEAT24_methylation": h24,

        "deltaMeth_6h": d6,
        "deltaMeth_12h": d12,
        "deltaMeth_24h": d24,

        "n_100bp_sRNA_bins": 1,
        "CTRL_24nt_CPM_sum": s["CTRLmean"],
        "HEAT_24nt_CPM_sum": s["HEATmean"],
        "delta24_CPM_sum": s["delta24_CPM"]
    })

out = pd.DataFrame(rows)

outfile = OUT / "four_heat_loci_FINAL_QC.tsv"
out.to_csv(outfile, sep="\t", index=False)

print(out.to_string(index=False))
print("\nSaved:")
print(outfile)
