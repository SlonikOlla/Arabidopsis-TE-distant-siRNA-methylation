#!/usr/bin/env python3

from pathlib import Path
from functools import lru_cache
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

ROOT = Path("/home/igor/arabidopsis_siRNA_methylation")
OUT  = ROOT / "results/nonTE_gene_analysis"
GO   = ROOT / "annotation/GO"

BGFILE  = OUT / "eligible_gene_background_support.tsv"
WINFILE = OUT / "all_TE_distant_1kb_concordant_gene_windows.tsv"
GAF     = GO / "gene_association.tair"
OBO     = GO / "go-basic.obo"

N_PERM = 10000
SEED = 20260830
rng = np.random.default_rng(SEED)

TARGETS = {
    "GO:0009408": "response to heat",
    "GO:0006457": "protein folding",
    "GO:0015979": "photosynthesis",
    "GO:0009765": "photosynthesis, light harvesting",
    "GO:0006091":
        "generation of precursor metabolites and energy",
}

# ------------------------------------------------------------
# Parse GO ontology
# ------------------------------------------------------------

print("Reading GO ontology...", flush=True)

terms = {}
cur = None

with open(OBO, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip()

        if line == "[Term]":
            if cur and "id" in cur:
                terms[cur["id"]] = cur
            cur = {
                "parents": [],
                "obsolete": False
            }
            continue

        if cur is None:
            continue

        if line.startswith("id: "):
            cur["id"] = line[4:]

        elif line.startswith("name: "):
            cur["name"] = line[6:]

        elif line.startswith("namespace: "):
            cur["namespace"] = line[11:]

        elif line.startswith("is_a: "):
            cur["parents"].append(
                line[6:].split()[0]
            )

        elif line.startswith(
            "relationship: part_of "
        ):
            cur["parents"].append(
                line.split()[2]
            )

        elif line == "is_obsolete: true":
            cur["obsolete"] = True

if cur and "id" in cur:
    terms[cur["id"]] = cur


@lru_cache(maxsize=None)
def ancestors(go):
    out = {go}

    if go in terms:
        for p in terms[go]["parents"]:
            out.update(ancestors(p))

    return frozenset(out)


# ------------------------------------------------------------
# Parse TAIR GAF
# ------------------------------------------------------------

print("Reading TAIR GO annotations...", flush=True)

direct = defaultdict(set)

with open(GAF, encoding="utf-8") as f:
    for line in f:

        if line.startswith("!"):
            continue

        x = line.rstrip().split("\t")

        if len(x) < 9:
            continue

        gene = x[1]
        qualifier = x[3]
        go = x[4]
        aspect = x[8]

        if aspect != "P":
            continue

        if "NOT" in qualifier.split("|"):
            continue

        if go not in terms:
            continue

        if terms[go].get("obsolete"):
            continue

        direct[gene].add(go)


gene2go = {}

for gene, gos in direct.items():

    z = set()

    for go in gos:
        z.update(ancestors(go))

    z = {
        go for go in z
        if go in terms
        and terms[go].get("namespace")
            == "biological_process"
        and not terms[go].get("obsolete")
    }

    z.discard("GO:0008150")

    gene2go[gene] = z


go2genes = defaultdict(set)

for gene, gos in gene2go.items():
    for go in gos:
        if go in TARGETS:
            go2genes[go].add(gene)

print(
    f"Genes with BP annotation: "
    f"{len(gene2go):,}",
    flush=True
)

for go, name in TARGETS.items():
    print(
        f"{go} {name}: "
        f"{len(go2genes[go])} annotated genes",
        flush=True
    )


# ------------------------------------------------------------
# Heat TE-distant gene-body background
# ------------------------------------------------------------

print("\nReading eligible background...", flush=True)

b = pd.read_csv(BGFILE, sep="\t")

b = b[
    (b["filter"] == "TE_distant_1kb") &
    (b["stress"] == "heat") &
    (b["feature"] == "gene_body")
].drop_duplicates().copy()


# ------------------------------------------------------------
# Candidate windows
# ------------------------------------------------------------

print("Reading candidate windows...", flush=True)

w = pd.read_csv(
    WINFILE,
    sep="\t",
    low_memory=False
)

w["tail_pct"] = pd.to_numeric(
    w["tail_pct"],
    errors="coerce"
)

w = w[
    (w["stress"] == "heat") &
    (w["feature"] == "gene_body") &
    (w["tail_pct"] == 1)
].copy()


# ------------------------------------------------------------
# Fast context-specific matched test
#
# Matching variables:
# number of measurable resolutions and time points
# WITHIN the tested methylation context.
# ------------------------------------------------------------

rows = []

for context in ["CG", "CHG", "CHH"]:

    bc = b[b["context"] == context].copy()

    opp = (
        bc.groupby("gene_id")
        .agg(
            n_resolutions=(
                "resolution", "nunique"
            ),
            n_times=("time", "nunique")
        )
        .reset_index()
    )

    opp["stratum"] = (
        opp["n_resolutions"].astype(str)
        + "|"
        + opp["n_times"].astype(str)
    )

    gene_stratum = dict(
        zip(opp["gene_id"], opp["stratum"])
    )

    pools = {
        s: np.array(x["gene_id"], dtype=object)
        for s, x in opp.groupby("stratum")
    }

    universe = set(opp["gene_id"])

    print(
        f"\n{context}: "
        f"{len(universe):,} eligible genes",
        flush=True
    )

    for direction in ["gain", "loss"]:

        selected = set(
            w[
                (w["context"] == context) &
                (w["direction"] == direction)
            ]["gene_id"].astype(str)
        )

        selected &= universe

        # Number selected from each opportunity stratum
        need = Counter(
            gene_stratum[g]
            for g in selected
        )

        print(
            f"  {direction}: "
            f"{len(selected)} candidates; "
            f"{len(need)} strata",
            flush=True
        )

        for goid, name in TARGETS.items():

            target = go2genes[goid] & universe
            obs = len(selected & target)

            if obs < 3:
                continue

            # For each stratum, determine number of
            # GO-positive genes in pool.
            stratum_info = []

            for s, kdraw in need.items():

                pool = pools[s]

                M = len(pool)

                K = sum(
                    g in target
                    for g in pool
                )

                stratum_info.append(
                    (M, K, kdraw)
                )

            # Vectorized hypergeometric sampling.
            null = np.zeros(
                N_PERM,
                dtype=np.int16
            )

            for M, K, kdraw in stratum_info:

                if K == 0 or kdraw == 0:
                    continue

                null += rng.hypergeometric(
                    ngood=K,
                    nbad=M-K,
                    nsample=kdraw,
                    size=N_PERM
                )

            ge = int(np.sum(null >= obs))

            p = (ge + 1) / (N_PERM + 1)

            mean_null = float(null.mean())
            sd_null = float(null.std(ddof=1))

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
                "null_sd": sd_null,
                "fold_vs_matched_null": fold,
                "empirical_p": p,
                "null_ge_observed": ge
            })

            print(
                f"    {name}: "
                f"obs={obs}, "
                f"null={mean_null:.3f}, "
                f"fold={fold:.2f}, "
                f"P={p:.6g}",
                flush=True
            )


# ------------------------------------------------------------
# BH correction
# ------------------------------------------------------------

res = pd.DataFrame(rows)

if len(res):

    p = res["empirical_p"].to_numpy()

    order = np.argsort(p)
    n = len(p)

    q = (
        p[order]
        * n
        / np.arange(1, n+1)
    )

    q = np.minimum.accumulate(
        q[::-1]
    )[::-1]

    q = np.minimum(q, 1)

    fdr = np.empty(n)
    fdr[order] = q

    res["FDR_BH"] = fdr

    res = res.sort_values(
        ["empirical_p",
         "context",
         "direction"]
    )

    outfile = (
        OUT /
        "heat_GO_opportunity_matched_fast.tsv"
    )

    res.to_csv(
        outfile,
        sep="\t",
        index=False
    )

    print(
        "\n===================================="
    )
    print("FINAL RESULTS")
    print(
        "===================================="
    )

    print(
        res.to_string(index=False),
        flush=True
    )

    print(
        f"\nSaved: {outfile}",
        flush=True
    )

print("\nDONE", flush=True)
