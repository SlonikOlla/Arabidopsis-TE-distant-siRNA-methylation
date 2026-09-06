#!/usr/bin/env python3

from pathlib import Path
from functools import lru_cache
import pandas as pd
import numpy as np

ROOT = Path("/home/igor/arabidopsis_siRNA_methylation")
OUT  = ROOT / "results/nonTE_gene_analysis"
GO   = ROOT / "annotation/GO"

WINDOWS = OUT / "all_TE_distant_1kb_concordant_gene_windows.tsv"
BACKGROUND = OUT / "eligible_gene_background_support.tsv"
GAF = GO / "gene_association.tair"
OBO = GO / "go-basic.obo"

# ------------------------------------------------------------
# GO ontology
# ------------------------------------------------------------

terms = {}
cur = None

with open(OBO) as f:
    for line in f:
        line = line.rstrip()

        if line == "[Term]":
            if cur and "id" in cur:
                terms[cur["id"]] = cur
            cur = {"parents": [], "obsolete": False}
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
            cur["parents"].append(line[6:].split()[0])
        elif line.startswith("relationship: part_of "):
            cur["parents"].append(line.split()[2])
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
# GAF: Biological Process
# ------------------------------------------------------------

direct = {}

with open(GAF) as f:
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

        direct.setdefault(gene, set()).add(go)


gene2go = {}

for gene, gos in direct.items():
    z = set()
    for go in gos:
        z.update(ancestors(go))

    z = {
        go for go in z
        if go in terms
        and terms[go].get("namespace") == "biological_process"
        and not terms[go].get("obsolete")
    }

    z.discard("GO:0008150")
    gene2go[gene] = z


# ------------------------------------------------------------
# Load candidate windows
# ------------------------------------------------------------

w = pd.read_csv(WINDOWS, sep="\t", low_memory=False)

w["tail_pct"] = pd.to_numeric(w["tail_pct"], errors="coerce")

w = w[
    (w["stress"] == "heat") &
    (w["feature"] == "gene_body") &
    (w["tail_pct"] == 1)
].copy()

# ------------------------------------------------------------
# Background support
# ------------------------------------------------------------

b = pd.read_csv(BACKGROUND, sep="\t")

b = b[
    (b["filter"] == "TE_distant_1kb") &
    (b["stress"] == "heat") &
    (b["feature"] == "gene_body")
].copy()


# ------------------------------------------------------------
# Fisher/hypergeometric enrichment
# ------------------------------------------------------------

from scipy.stats import hypergeom

def bh(p):
    p = np.asarray(p, dtype=float)
    n = len(p)

    order = np.argsort(p)
    x = p[order] * n / np.arange(1, n+1)
    x = np.minimum.accumulate(x[::-1])[::-1]
    x = np.minimum(x, 1)

    out = np.empty(n)
    out[order] = x
    return out


rows = []
summaries = []

for context in ["CG", "CHG", "CHH"]:
    for direction in ["gain", "loss"]:

        # Candidate genes in this exact context
        sel = set(
            w[
                (w["context"] == context) &
                (w["direction"] == direction)
            ]["gene_id"].astype(str)
        )

        # Eligible genes must have measurable support in this
        # context. We accept either resolution/time because the
        # candidate definition is gene-level presence in the 1% tail.
        universe = set(
            b[b["context"] == context]["gene_id"].astype(str)
        )

        sel &= universe

        universe_go = universe & set(gene2go)
        sel_go = sel & universe_go

        N = len(universe_go)
        n = len(sel_go)

        summaries.append({
            "context": context,
            "direction": direction,
            "selected_total": len(sel),
            "selected_BP": n,
            "background_BP": N
        })

        print(
            f"{context:3s} {direction:4s}: "
            f"selected={len(sel):4d}, "
            f"BP={n:4d}, background={N:5d}",
            flush=True
        )

        if n < 3:
            continue

        all_go = set()
        for gene in universe_go:
            all_go.update(gene2go[gene])

        tmp = []

        for go in all_go:
            bg_genes = {
                g for g in universe_go
                if go in gene2go[g]
            }

            K = len(bg_genes)

            if K < 10:
                continue

            hits = sel_go & bg_genes
            k = len(hits)

            if k < 3:
                continue

            p = hypergeom.sf(k-1, N, K, n)
            expected = n * K / N
            fold = (k/n)/(K/N)

            tmp.append({
                "context": context,
                "direction": direction,
                "GO_ID": go,
                "GO_term": terms[go].get("name", ""),
                "hits": k,
                "selected_BP": n,
                "background_hits": K,
                "background_BP": N,
                "expected": expected,
                "fold_enrichment": fold,
                "p_value": p,
                "genes": ";".join(sorted(hits))
            })

        if tmp:
            z = pd.DataFrame(tmp)
            z["FDR_BH"] = bh(z["p_value"])
            rows.append(z)


summary = pd.DataFrame(summaries)

summary.to_csv(
    OUT / "heat_context_specific_GO_summary.tsv",
    sep="\t", index=False
)

if rows:
    result = pd.concat(rows, ignore_index=True)

    result = result.sort_values(
        ["context", "direction", "FDR_BH", "fold_enrichment"],
        ascending=[True, True, True, False]
    )

    result.to_csv(
        OUT / "heat_context_specific_GO_all.tsv",
        sep="\t", index=False
    )

    sig = result[
        (result["FDR_BH"] < 0.05) &
        (result["fold_enrichment"] > 1)
    ].copy()

    sig.to_csv(
        OUT / "heat_context_specific_GO_significant.tsv",
        sep="\t", index=False
    )

    print("\n===== SIGNIFICANT GO TERMS =====")

    for (context, direction), g in sig.groupby(
        ["context", "direction"]
    ):
        print(f"\n### {context} | {direction}")

        q = g.sort_values(
            ["FDR_BH", "fold_enrichment"],
            ascending=[True, False]
        ).head(20)

        for _, r in q.iterrows():
            print(
                f"{r.GO_ID}\t{r.GO_term}\t"
                f"hits={int(r.hits)}\t"
                f"fold={r.fold_enrichment:.2f}\t"
                f"FDR={r.FDR_BH:.3g}\t"
                f"{r.genes}"
            )

else:
    print("No testable GO terms.")

print("\nDONE")
