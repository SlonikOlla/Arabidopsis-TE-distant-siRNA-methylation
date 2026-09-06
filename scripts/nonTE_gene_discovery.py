#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
from bisect import bisect_left

ROOT = Path("/home/igor/arabidopsis_siRNA_methylation")
OUT = ROOT / "results/nonTE_gene_analysis"
OUT.mkdir(parents=True, exist_ok=True)

TAILS = [1, 5, 10]
CONTEXTS = ["CG", "CHG", "CHH"]
RESOLUTIONS = [100, 500]

# ------------------------------------------------------------
# Annotation
# ------------------------------------------------------------

def read_bed(path, names):
    return pd.read_csv(path, sep="\t", header=None, names=names)

genes = read_bed(
    ROOT / "annotation/Araport11_genes.bed",
    ["chr","start","end","gene_id","strand"]
)

promoters = read_bed(
    ROOT / "annotation/Araport11_promoters_1kb.bed",
    ["chr","start","end","gene_id","strand"]
)

tes = read_bed(
    ROOT / "annotation/TAIR10_TEs_with_family_superfamily.bed",
    ["chr","start","end","TE_id","TE_family","TE_superfamily"]
)

for d in (genes, promoters, tes):
    d["chr"] = d["chr"].astype(str)
    d["start"] = pd.to_numeric(d["start"])
    d["end"] = pd.to_numeric(d["end"])

# interval dictionaries
gene_by_chr = defaultdict(list)
prom_by_chr = defaultdict(list)
te_by_chr = defaultdict(list)

for r in genes.itertuples():
    gene_by_chr[r.chr].append((r.start, r.end, r.gene_id))

for r in promoters.itertuples():
    prom_by_chr[r.chr].append((r.start, r.end, r.gene_id))

for r in tes.itertuples():
    te_by_chr[r.chr].append(
        (r.start, r.end, r.TE_id, r.TE_family, r.TE_superfamily)
    )

for D in (gene_by_chr, prom_by_chr, te_by_chr):
    for c in D:
        D[c].sort()

# TE starts for fast nearest-distance calculations
te_starts = {}
for c, arr in te_by_chr.items():
    te_starts[c] = [x[0] for x in arr]


def overlaps(intervals, start, end):
    """Return records overlapping [start,end)."""
    out = []
    # simple scan around insertion point; annotations are modest in size
    starts = [x[0] for x in intervals]
    i = bisect_left(starts, end)

    j = i - 1
    while j >= 0 and intervals[j][1] > start:
        if intervals[j][0] < end:
            out.append(intervals[j])
        j -= 1

    return out


def te_status(chrom, start, end):
    arr = te_by_chr.get(chrom, [])
    if not arr:
        return False, np.inf, ""

    ov = overlaps(arr, start, end)
    if ov:
        ids = ";".join(sorted(set(x[2] for x in ov)))
        return True, 0, ids

    # nearest TE distance
    starts = te_starts[chrom]
    pos = bisect_left(starts, end)

    distances = []

    # TE beginning after window
    for idx in [pos, pos+1]:
        if 0 <= idx < len(arr):
            s,e,*_ = arr[idx]
            if s >= end:
                distances.append(s-end)

    # TE ending before window
    for idx in [pos-1, pos-2]:
        if 0 <= idx < len(arr):
            s,e,*_ = arr[idx]
            if e <= start:
                distances.append(start-e)

    dist = min(distances) if distances else np.inf
    return False, dist, ""


def map_genes(chrom, start, end):
    gb = overlaps(gene_by_chr.get(chrom, []), start, end)
    pr = overlaps(prom_by_chr.get(chrom, []), start, end)

    rows = []

    for x in gb:
        rows.append((x[2], "gene_body"))

    for x in pr:
        rows.append((x[2], "promoter"))

    return list(dict.fromkeys(rows))


# ------------------------------------------------------------
# Input definitions
# ------------------------------------------------------------

def methyl_path(stress, context, res):
    if stress == "phosphate":
        return ROOT / f"results/root_methylation/root_{context}_{res}bp_matrix.tsv"
    return ROOT / f"{stress}_consensus/results/{stress}_{context}_{res}bp_matrix.tsv"


def srna_path(stress, res):
    if stress == "phosphate":
        return ROOT / f"results/root_siRNA/root_24nt_{res}bp_matrix.tsv"
    return ROOT / f"{stress}_consensus/results/{stress}_24nt_{res}bp_matrix.tsv"


def load_srna(stress, res):
    d = pd.read_csv(srna_path(stress, res), sep="\t")
    d["chr"] = d["chr"].astype(str)

    if stress == "phosphate":
        col = "delta24_CPM" if res == 100 else "delta_CPM"
        return {"single": d[["chr","start","end",col]].rename(columns={col:"delta24"})}

    if stress == "drought":
        return {
            "single":
            d[["chr","start","end","delta24_CPM"]]
            .rename(columns={"delta24_CPM":"delta24"})
        }

    if stress == "heat":
        # one 1-h sRNA contrast paired with 6/12/24-h methylomes
        q = d[["chr","start","end","delta24_CPM"]].rename(
            columns={"delta24_CPM":"delta24"}
        )
        return {"6h":q.copy(), "12h":q.copy(), "24h":q.copy()}

    if stress == "pathogen":
        m6 = (
            d["GSM491567_mock_6hpi_rep1"] +
            d["GSM491568_mock_6hpi_rep2"]
        ) / 2.0

        m14 = (
            d["GSM491572_mock_14hpi_rep1"] +
            d["GSM491573_mock_14hpi_rep2"]
        ) / 2.0

        q6 = d[["chr","start","end"]].copy()
        q6["delta24"] = d["GSM491570_ev_6hpi"] - m6

        q14 = d[["chr","start","end"]].copy()
        q14["delta24"] = d["GSM491576_ev_14hpi"] - m14

        return {"6h":q6, "14h":q14}

    raise ValueError(stress)


def load_methyl(stress, context, res):
    d = pd.read_csv(methyl_path(stress, context, res), sep="\t")
    d["chr"] = d["chr"].astype(str)

    if stress == "heat":
        out = {}
        for t in ["6h","12h","24h"]:
            col = f"delta{context}_{t}"
            out[t] = d[["chr","start","end",col]].rename(
                columns={col:"deltaMeth"}
            )
        return out

    col = f"delta{context}"

    # Legacy pathogen CHH matrices use delta_CHH
    if col not in d.columns:
        alt = f"delta_{context}"
        if alt in d.columns:
            col = alt
        else:
            raise KeyError(
                f"No delta column for {stress} {context} {res}bp. "
                f"Available columns: {list(d.columns)}"
            )

    if stress == "pathogen":
        q = d[["chr","start","end",col]].rename(columns={col:"deltaMeth"})
        # same methylome paired to 6-h and 14-h sRNA
        return {"6h":q.copy(), "14h":q.copy()}

    return {
        "single":
        d[["chr","start","end",col]]
        .rename(columns={col:"deltaMeth"})
    }


# ------------------------------------------------------------
# Extreme-tail discovery
# ------------------------------------------------------------

all_windows = []

for stress in ["phosphate","drought","heat","pathogen"]:
    for res in RESOLUTIONS:

        srna_sets = load_srna(stress, res)

        for context in CONTEXTS:
            meth_sets = load_methyl(stress, context, res)

            common_times = sorted(set(srna_sets) & set(meth_sets))

            for time in common_times:
                s = srna_sets[time]
                m = meth_sets[time]

                x = s.merge(
                    m,
                    on=["chr","start","end"],
                    how="inner"
                )

                x = x.replace([np.inf,-np.inf], np.nan)
                x = x.dropna(subset=["delta24","deltaMeth"])

                if len(x) == 0:
                    continue

                n = len(x)

                # percentile ranks
                x["rank24"] = x["delta24"].rank(method="average", pct=True)
                x["rankMeth"] = x["deltaMeth"].rank(method="average", pct=True)

                for pct in TAILS:
                    f = pct / 100.0

                    gain = x[
                        (x["rank24"] >= 1-f) &
                        (x["rankMeth"] >= 1-f) &
                        (x["delta24"] > 0) &
                        (x["deltaMeth"] > 0)
                    ].copy()

                    loss = x[
                        (x["rank24"] <= f) &
                        (x["rankMeth"] <= f) &
                        (x["delta24"] < 0) &
                        (x["deltaMeth"] < 0)
                    ].copy()

                    for direction, z in [("gain",gain),("loss",loss)]:
                        if z.empty:
                            continue

                        z["stress"] = stress
                        z["time"] = time
                        z["context"] = context
                        z["resolution"] = res
                        z["tail_pct"] = pct
                        z["direction"] = direction

                        # joint extremeness: conservative weaker rank
                        if direction == "gain":
                            z["joint_score"] = np.minimum(
                                z["rank24"], z["rankMeth"]
                            )
                        else:
                            z["joint_score"] = np.minimum(
                                1-z["rank24"], 1-z["rankMeth"]
                            )

                        keep = [
                            "stress","time","context","resolution",
                            "tail_pct","direction",
                            "chr","start","end",
                            "delta24","deltaMeth",
                            "rank24","rankMeth","joint_score"
                        ]

                        all_windows.append(z[keep])

if not all_windows:
    raise RuntimeError("No concordant windows found.")

W = pd.concat(all_windows, ignore_index=True)

# ------------------------------------------------------------
# TE exclusion + gene/promoter mapping
# ------------------------------------------------------------

mapped = []

for i, r in enumerate(W.itertuples(index=False), 1):
    chrom = str(r.chr)

    te_overlap, te_distance, te_ids = te_status(
        chrom, int(r.start), int(r.end)
    )

    # user requested eliminating all TE-overlapping windows
    if te_overlap:
        continue

    gene_hits = map_genes(
        chrom, int(r.start), int(r.end)
    )

    if not gene_hits:
        continue

    for gene_id, feature in gene_hits:
        mapped.append({
            "stress": r.stress,
            "time": r.time,
            "context": r.context,
            "resolution": r.resolution,
            "tail_pct": r.tail_pct,
            "direction": r.direction,
            "chr": chrom,
            "start": int(r.start),
            "end": int(r.end),
            "delta24": r.delta24,
            "deltaMeth": r.deltaMeth,
            "rank24": r.rank24,
            "rankMeth": r.rankMeth,
            "joint_score": r.joint_score,
            "gene_id": gene_id,
            "feature": feature,
            "TE_overlap": False,
            "nearest_TE_distance_bp": te_distance,
            "TE_distant_1kb": bool(te_distance >= 1000)
        })

M = pd.DataFrame(mapped)

M.to_csv(
    OUT / "all_nonTE_concordant_gene_windows.tsv",
    sep="\t", index=False
)

M[M["TE_distant_1kb"]].to_csv(
    OUT / "all_TE_distant_1kb_concordant_gene_windows.tsv",
    sep="\t", index=False
)

# ------------------------------------------------------------
# Gene ranking
# ------------------------------------------------------------

def make_gene_ranking(d, label):
    if d.empty:
        return pd.DataFrame()

    # Avoid counting 1%, 5%, 10% membership as independent evidence.
    # First collapse same physical window/test to its most stringent tail.
    collapsed = (
        d.sort_values("tail_pct")
         .drop_duplicates(
             subset=[
                 "stress","time","context","resolution",
                 "direction","chr","start","end",
                 "gene_id","feature"
             ],
             keep="first"
         )
    )

    group_cols = [
        "gene_id","stress","direction","feature"
    ]

    rows = []

    for key, g in collapsed.groupby(group_cols):
        gene_id, stress, direction, feature = key

        contexts = sorted(g["context"].unique())
        resolutions = sorted(g["resolution"].unique())
        times = sorted(g["time"].unique())
        tails = sorted(
            d[
                (d["gene_id"] == gene_id) &
                (d["stress"] == stress) &
                (d["direction"] == direction) &
                (d["feature"] == feature)
            ]["tail_pct"].unique()
        )

        n100 = int((g["resolution"] == 100).sum())
        n500 = int((g["resolution"] == 500).sum())

        # reproducibility-oriented score
        score = (
            2.0 * len(contexts) +
            2.0 * len(resolutions) +
            1.5 * len(times) +
            np.log1p(len(g)) +
            3.0 * float(g["joint_score"].max())
        )

        tier = "Tier3"

        # Tier1: at least one 1% event plus both resolutions
        # and recurrence in time or context
        if (
            1 in tails and
            len(resolutions) == 2 and
            (len(times) >= 2 or len(contexts) >= 2)
        ):
            tier = "Tier1"

        # Tier2: at least 5% plus both resolutions OR multiple contexts/times
        elif (
            min(tails) <= 5 and
            (
                len(resolutions) == 2 or
                len(times) >= 2 or
                len(contexts) >= 2
            )
        ):
            tier = "Tier2"

        rows.append({
            "gene_id": gene_id,
            "stress": stress,
            "direction": direction,
            "feature": feature,
            "tier": tier,
            "score": score,
            "n_unique_windows_tests": len(g),
            "n_100bp": n100,
            "n_500bp": n500,
            "n_contexts": len(contexts),
            "contexts": ";".join(contexts),
            "n_times": len(times),
            "times": ";".join(times),
            "tail_support": ";".join(map(str,tails)),
            "max_joint_score": g["joint_score"].max(),
            "mean_abs_delta24": g["delta24"].abs().mean(),
            "mean_abs_deltaMeth": g["deltaMeth"].abs().mean(),
            "min_TE_distance_bp": g["nearest_TE_distance_bp"].min()
        })

    R = pd.DataFrame(rows)

    tier_order = {"Tier1":0, "Tier2":1, "Tier3":2}
    R["_tier"] = R["tier"].map(tier_order)

    R = R.sort_values(
        ["_tier","score","max_joint_score"],
        ascending=[True,False,False]
    ).drop(columns="_tier")

    R.to_csv(
        OUT / f"{label}_gene_rankings.tsv",
        sep="\t", index=False
    )

    return R


R0 = make_gene_ranking(
    M,
    "nonTE"
)

R1 = make_gene_ranking(
    M[M["TE_distant_1kb"]],
    "TE_distant_1kb"
)

# ------------------------------------------------------------
# Pathway-ready gene sets
# ------------------------------------------------------------

sets = []

for label, R in [
    ("nonTE",R0),
    ("TE_distant_1kb",R1)
]:
    if R.empty:
        continue

    for (stress,direction,feature,tier), g in R.groupby(
        ["stress","direction","feature","tier"]
    ):
        sets.append({
            "filter": label,
            "stress": stress,
            "direction": direction,
            "feature": feature,
            "tier": tier,
            "n_genes": g["gene_id"].nunique(),
            "genes": ";".join(sorted(g["gene_id"].unique()))
        })

pd.DataFrame(sets).to_csv(
    OUT / "pathway_input_gene_sets.tsv",
    sep="\t", index=False
)

# ------------------------------------------------------------
# Compact summaries
# ------------------------------------------------------------

def print_summary(name, R):
    print("\n==========", name, "==========")
    if R.empty:
        print("No genes.")
        return

    print("Genes:", R["gene_id"].nunique())

    print("\nBy tier:")
    print(R.groupby("tier")["gene_id"].nunique().to_string())

    print("\nBy stress/direction:")
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
    print(R[cols].head(30).to_string(index=False))


print_summary("ALL TE-FREE", R0)
print_summary(">=1 kb FROM TE", R1)

print("\nWritten to:", OUT)
