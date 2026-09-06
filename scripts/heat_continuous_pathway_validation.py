#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import argparse
import numpy as np
import pandas as pd

ROOT = Path("/home/igor/arabidopsis_siRNA_methylation")
OUT = ROOT / "results/nonTE_gene_analysis"
CACHE = OUT / "heat_window_opportunity_cache_gene_body"
GO_DIR = ROOT / "annotation/GO"

SRNA = {
    100: ROOT / "heat_consensus/results/heat_24nt_100bp_matrix.tsv",
    500: ROOT / "heat_consensus/results/heat_24nt_500bp_matrix.tsv",
}

METH = {
    ("CG",100):  ROOT / "heat_consensus/results/heat_CG_100bp_matrix.tsv",
    ("CG",500):  ROOT / "heat_consensus/results/heat_CG_500bp_matrix.tsv",
    ("CHG",100): ROOT / "heat_consensus/results/heat_CHG_100bp_matrix.tsv",
    ("CHG",500): ROOT / "heat_consensus/results/heat_CHG_500bp_matrix.tsv",
    ("CHH",100): ROOT / "heat_consensus/results/heat_CHH_100bp_matrix.tsv",
    ("CHH",500): ROOT / "heat_consensus/results/heat_CHH_500bp_matrix.tsv",
}

TARGETS = {
    "GO:0015979": "Photosynthesis",
    "GO:0009765": "Light harvesting",
    "GO:0009408": "Response to heat",
    "GO:0006457": "Protein folding",
}

parser = argparse.ArgumentParser()
parser.add_argument("--nperm", type=int, default=10000)
parser.add_argument("--seed", type=int, default=20260831)
args = parser.parse_args()

RESULT_DIR = OUT / "heat_continuous_pathway"
RESULT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# GO ontology
# ------------------------------------------------------------

obo = GO_DIR / "go-basic.obo"
gaf = GO_DIR / "gene_association.tair"

parents = defaultdict(set)
namespace = {}
names = {}

current = None

with open(obo) as fh:
    for line in fh:
        line = line.rstrip()

        if line == "[Term]":
            current = None
            continue

        if line.startswith("id: GO:"):
            current = line.split("id: ",1)[1]
            continue

        if current is None:
            continue

        if line.startswith("name: "):
            names[current] = line.split("name: ",1)[1]

        elif line.startswith("namespace: "):
            namespace[current] = line.split("namespace: ",1)[1]

        elif line.startswith("is_a: GO:"):
            p = line.split()[1]
            parents[current].add(p)

        elif line.startswith("relationship: part_of GO:"):
            p = line.split()[2]
            parents[current].add(p)

ancestor_cache = {}

def ancestors(go):
    if go in ancestor_cache:
        return ancestor_cache[go]

    out = {go}
    for p in parents.get(go, []):
        out |= ancestors(p)

    ancestor_cache[go] = out
    return out

# ------------------------------------------------------------
# TAIR GO annotations, propagated through BP ontology
# ------------------------------------------------------------

gene2go = defaultdict(set)

with open(gaf) as fh:
    for line in fh:
        if line.startswith("!"):
            continue

        x = line.rstrip("\n").split("\t")
        if len(x) < 5:
            continue

        gene = x[1]
        qualifier = x[3]
        go = x[4]

        if "NOT" in qualifier.split("|"):
            continue

        if namespace.get(go) != "biological_process":
            continue

        for a in ancestors(go):
            if namespace.get(a) == "biological_process" and a != "GO:0008150":
                gene2go[gene].add(a)

bp_genes = set(gene2go)

print(f"GO BP annotated genes: {len(bp_genes):,}")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def methyl_col(context, time):
    return f"delta{context}_{time}"

def bh(p):
    p = np.asarray(p, dtype=float)
    n = len(p)

    order = np.argsort(p)
    ranked = p[order]

    q = ranked * n / np.arange(1, n+1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)

    out = np.empty(n)
    out[order] = q
    return out

# ------------------------------------------------------------
# Continuous window scores
#
# gain:
#   high positive delta24 + high positive deltaMeth
#
# loss:
#   strong negative delta24 + strong negative deltaMeth
#
# Rank-product lies from ~0 to 1.
# Non-concordant windows receive 0.
# ------------------------------------------------------------

gene_stratum = []

for context in ["CG","CHG","CHH"]:

    print(f"\n=== {context} ===", flush=True)

    for resolution in [100,500]:

        s = pd.read_csv(
            SRNA[resolution],
            sep="\t",
            usecols=["chr","start","end","delta24_CPM"]
        )

        s["chr"] = s["chr"].astype(str)
        s["delta24_CPM"] = pd.to_numeric(
            s["delta24_CPM"], errors="coerce"
        )

        s["window_id"] = (
            s["chr"] + ":" +
            s["start"].astype(str) + "-" +
            s["end"].astype(str)
        )

        for time in ["6h","12h","24h"]:

            tag = f"{context}_{resolution}_{time}"

            print(f"  {tag}", flush=True)

            mf = METH[(context,resolution)]
            mcol = methyl_col(context,time)

            m = pd.read_csv(
                mf,
                sep="\t",
                usecols=["chr","start","end",mcol]
            )

            m["chr"] = m["chr"].astype(str)
            m[mcol] = pd.to_numeric(
                m[mcol], errors="coerce"
            )

            m["window_id"] = (
                m["chr"] + ":" +
                m["start"].astype(str) + "-" +
                m["end"].astype(str)
            )

            d = s[["window_id","delta24_CPM"]].merge(
                m[["window_id",mcol]],
                on="window_id",
                how="inner"
            )

            d = d[
                np.isfinite(d["delta24_CPM"]) &
                np.isfinite(d[mcol]) &
                (d[mcol].abs() <= 1)
            ].copy()

            mapfile = CACHE / f"eligible_{tag}_gene_map.tsv"

            mp = pd.read_csv(
                mapfile,
                sep="\t",
                usecols=["window_id","gene_id"]
            ).drop_duplicates()

            # Restrict ranks to the exact eligible TE-distant
            # gene-body window universe.
            eligible_ids = mp[["window_id"]].drop_duplicates()

            d = eligible_ids.merge(
                d,
                on="window_id",
                how="inner"
            )

            n = len(d)

            # Positive-tail ranks: largest value approaches 1.
            r24_gain = d["delta24_CPM"].rank(
                method="average",
                pct=True,
                ascending=True
            )

            rm_gain = d[mcol].rank(
                method="average",
                pct=True,
                ascending=True
            )

            # Negative-tail ranks: most negative value approaches 1.
            r24_loss = d["delta24_CPM"].rank(
                method="average",
                pct=True,
                ascending=False
            )

            rm_loss = d[mcol].rank(
                method="average",
                pct=True,
                ascending=False
            )

            gain_mask = (
                (d["delta24_CPM"] > 0) &
                (d[mcol] > 0)
            )

            loss_mask = (
                (d["delta24_CPM"] < 0) &
                (d[mcol] < 0)
            )

            d["gain_score"] = 0.0
            d["loss_score"] = 0.0

            d.loc[gain_mask, "gain_score"] = (
                r24_gain[gain_mask] *
                rm_gain[gain_mask]
            )

            d.loc[loss_mask, "loss_score"] = (
                r24_loss[loss_mask] *
                rm_loss[loss_mask]
            )

            x = mp.merge(
                d[["window_id","gain_score","loss_score"]],
                on="window_id",
                how="inner"
            )

            for direction in ["gain","loss"]:

                scorecol = f"{direction}_score"

                g = (
                    x.groupby("gene_id")
                    .agg(
                        score=(scorecol,"mean"),
                        n_windows=("window_id","nunique")
                    )
                    .reset_index()
                )

                g["context"] = context
                g["resolution"] = resolution
                g["time"] = time
                g["direction"] = direction

                gene_stratum.append(g)

            print(
                f"    eligible windows={n:,}; "
                f"genes={mp.gene_id.nunique():,}",
                flush=True
            )

# ------------------------------------------------------------
# Combine six strata per context:
# 100/500 bp x 6/12/24 h
#
# Equal weight to each experimental stratum.
# Opportunity = total eligible gene-window observations.
# ------------------------------------------------------------

gs = pd.concat(gene_stratum, ignore_index=True)

combined = (
    gs.groupby(["context","direction","gene_id"])
    .agg(
        continuous_score=("score","mean"),
        opportunity=("n_windows","sum"),
        n_strata=("score","size")
    )
    .reset_index()
)

combined["rank"] = (
    combined.groupby(["context","direction"])
    ["continuous_score"]
    .rank(method="average", ascending=False)
)

combined.to_csv(
    RESULT_DIR / "heat_continuous_gene_rankings.tsv",
    sep="\t",
    index=False
)

# ------------------------------------------------------------
# Opportunity-matched pathway validation
#
# Random genes are matched to pathway genes by deciles of
# eligible-window opportunity.
# ------------------------------------------------------------

rng = np.random.default_rng(args.seed)
results = []

for context in ["CG","CHG","CHH"]:
    for direction in ["gain","loss"]:

        z = combined[
            (combined.context == context) &
            (combined.direction == direction) &
            (combined.gene_id.isin(bp_genes))
        ].copy()

        # Opportunity bins.
        z["opp_bin"] = pd.qcut(
            np.log1p(z["opportunity"]),
            q=10,
            labels=False,
            duplicates="drop"
        )

        print(
            f"\n{context} {direction}: "
            f"{len(z):,} BP-annotated genes",
            flush=True
        )

        for go, label in TARGETS.items():

            pathway = {
                g for g in bp_genes
                if go in gene2go[g]
            }

            obs = z[z.gene_id.isin(pathway)].copy()

            if len(obs) < 3:
                continue

            observed = obs["continuous_score"].mean()

            required = (
                obs.groupby("opp_bin")
                .size()
                .to_dict()
            )

            null = np.zeros(args.nperm, dtype=float)

            for i in range(args.nperm):

                scores = []

                for b, k in required.items():

                    pool = z[z.opp_bin == b]

                    idx = rng.choice(
                        len(pool),
                        size=int(k),
                        replace=False
                    )

                    scores.extend(
                        pool.iloc[idx]["continuous_score"].values
                    )

                null[i] = np.mean(scores)

            null_mean = null.mean()

            p_emp = (
                1 + np.sum(null >= observed)
            ) / (args.nperm + 1)

            fold = (
                observed / null_mean
                if null_mean > 0
                else np.nan
            )

            results.append({
                "context": context,
                "direction": direction,
                "GO_ID": go,
                "GO_term": label,
                "n_pathway_genes": len(obs),
                "observed_mean_score": observed,
                "null_mean_score": null_mean,
                "fold_enrichment": fold,
                "empirical_P": p_emp,
                "pathway_genes": ";".join(
                    sorted(obs.gene_id)
                )
            })

res = pd.DataFrame(results)

if len(res):

    res["targeted_FDR"] = np.nan

    for (context,direction), idx in res.groupby(
        ["context","direction"]
    ).groups.items():

        idx = list(idx)

        res.loc[idx,"targeted_FDR"] = bh(
            res.loc[idx,"empirical_P"].values
        )

    res = res.sort_values(
        ["context","direction","empirical_P"]
    )

res.to_csv(
    RESULT_DIR / "heat_continuous_targeted_pathways.tsv",
    sep="\t",
    index=False
)

print("\n\nHEAT CONTINUOUS PATHWAY VALIDATION")

for _, r in res.iterrows():
    print(
        f"\n{r.GO_term} | "
        f"{r.context} {r.direction}\n"
        f"  genes={int(r.n_pathway_genes)} "
        f"obs={r.observed_mean_score:.6f} "
        f"null={r.null_mean_score:.6f} "
        f"fold={r.fold_enrichment:.3f} "
        f"P={r.empirical_P:.6g} "
        f"targeted_FDR={r.targeted_FDR:.6g}"
    )

print(
    "\nSaved to:",
    RESULT_DIR
)
