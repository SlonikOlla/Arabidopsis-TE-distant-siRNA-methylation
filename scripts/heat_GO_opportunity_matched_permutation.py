#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

ROOT = Path("/home/igor/arabidopsis_siRNA_methylation")
OUT  = ROOT / "results/nonTE_gene_analysis"

BGFILE = OUT / "eligible_gene_background_support.tsv"
WINFILE = OUT / "all_TE_distant_1kb_concordant_gene_windows.tsv"
GOFILE = OUT / "GO_BP_enrichment_all.tsv"

N_PERM = 10000
SEED = 20260830
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------
# Eligible opportunity structure
# ------------------------------------------------------------

b = pd.read_csv(BGFILE, sep="\t")

b = b[
    (b["filter"] == "TE_distant_1kb") &
    (b["stress"] == "heat") &
    (b["feature"] == "gene_body")
].copy()

# Count distinct measurable context/resolution/time opportunities.
opp = (
    b.drop_duplicates(
        ["gene_id","context","resolution","time"]
    )
    .groupby("gene_id")
    .agg(
        n_opportunities=("gene_id","size"),
        n_contexts=("context","nunique"),
        n_resolutions=("resolution","nunique"),
        n_times=("time","nunique")
    )
    .reset_index()
)

# Exact opportunity strata.
opp["stratum"] = (
    opp["n_opportunities"].astype(str) + "|" +
    opp["n_contexts"].astype(str) + "|" +
    opp["n_resolutions"].astype(str) + "|" +
    opp["n_times"].astype(str)
)

gene_stratum = dict(zip(opp.gene_id, opp.stratum))

stratum_genes = {
    s: x.gene_id.to_numpy()
    for s,x in opp.groupby("stratum")
}

print("Eligible heat gene-body genes:", len(opp))
print("\nOpportunity strata:")
print(
    opp.groupby(
        ["n_opportunities","n_contexts","n_resolutions","n_times"]
    ).size().sort_values(ascending=False).head(20)
)

# ------------------------------------------------------------
# Candidate genes: TE-distant joint 1% tail
# ------------------------------------------------------------

w = pd.read_csv(WINFILE, sep="\t", low_memory=False)
w["tail_pct"] = pd.to_numeric(w["tail_pct"], errors="coerce")

w = w[
    (w["stress"] == "heat") &
    (w["feature"] == "gene_body") &
    (w["tail_pct"] == 1)
].copy()

# ------------------------------------------------------------
# Get GO memberships from existing enrichment file
#
# We only need the specific biologically central terms.
# Gene membership is recovered from genes_with_symbols/genes
# if present.
# ------------------------------------------------------------

go = pd.read_csv(GOFILE, sep="\t", low_memory=False)

TARGETS = {
    "GO:0009408": "response to heat",
    "GO:0006457": "protein folding",
    "GO:0015979": "photosynthesis",
    "GO:0009765": "photosynthesis, light harvesting",
    "GO:0006091": "generation of precursor metabolites and energy",
}

# Build GO -> genes from the existing table.
go2genes = defaultdict(set)

gene_col = None
for candidate in ["genes", "genes_with_symbols"]:
    if candidate in go.columns:
        gene_col = candidate
        break

if gene_col is None:
    raise RuntimeError(
        "Could not find genes or genes_with_symbols in GO table"
    )

for _,r in go[go["GO_ID"].isin(TARGETS)].iterrows():
    val = r.get(gene_col)
    if pd.isna(val):
        continue

    for z in str(val).split(";"):
        # Remove symbol if AT1G...(...)
        gid = z.split("(")[0].strip()
        if gid.startswith("AT"):
            go2genes[r["GO_ID"]].add(gid)

# ------------------------------------------------------------
# Matched sampler
# ------------------------------------------------------------

def matched_random_set(selected):
    """
    For every selected gene, draw a random eligible gene from
    the same opportunity stratum. Avoid duplicate genes where
    possible.
    """
    chosen = set()

    for gene in selected:
        s = gene_stratum.get(gene)
        if s is None:
            continue

        pool = stratum_genes[s]

        available = np.array(
            [g for g in pool if g not in chosen],
            dtype=object
        )

        if len(available):
            pick = rng.choice(available)
        else:
            pick = rng.choice(pool)

        chosen.add(pick)

    return chosen


# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

rows = []

for context in ["CG","CHG","CHH"]:
    for direction in ["gain","loss"]:

        selected = set(
            w[
                (w["context"] == context) &
                (w["direction"] == direction)
            ]["gene_id"].astype(str)
        )

        selected &= set(gene_stratum)

        print(
            f"\n{context} {direction}: "
            f"{len(selected)} candidate genes"
        )

        for goid,name in TARGETS.items():

            target_genes = go2genes.get(goid, set())

            obs = len(selected & target_genes)

            # Only test biologically relevant combinations if
            # at least 3 observed genes.
            if obs < 3:
                continue

            null = np.empty(N_PERM, dtype=int)

            for i in range(N_PERM):
                rand = matched_random_set(selected)
                null[i] = len(rand & target_genes)

            p = (
                1 + np.sum(null >= obs)
            ) / (N_PERM + 1)

            mean_null = null.mean()

            fold = (
                obs / mean_null
                if mean_null > 0
                else np.inf
            )

            rows.append({
                "context": context,
                "direction": direction,
                "GO_ID": goid,
                "GO_term": name,
                "n_selected": len(selected),
                "observed_hits": obs,
                "null_mean_hits": mean_null,
                "null_sd": null.std(ddof=1),
                "fold_vs_matched_null": fold,
                "empirical_p": p,
                "null_ge_observed": int(
                    np.sum(null >= obs)
                )
            })

            print(
                f"  {goid} {name}: "
                f"obs={obs}, "
                f"null={mean_null:.3f}, "
                f"fold={fold:.2f}, "
                f"P={p:.6g}"
            )

res = pd.DataFrame(rows)

if len(res):
    # BH over these targeted tests
    p = res["empirical_p"].to_numpy()
    order = np.argsort(p)
    n = len(p)

    q = p[order] * n / np.arange(1,n+1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q,1)

    fdr = np.empty(n)
    fdr[order] = q
    res["FDR_BH"] = fdr

    res.to_csv(
        OUT / "heat_GO_opportunity_matched_permutation.tsv",
        sep="\t", index=False
    )

    print("\n======================================")
    print("FINAL RESULTS")
    print("======================================")
    print(
        res.sort_values("empirical_p").to_string(index=False)
    )

print("\nDONE")
