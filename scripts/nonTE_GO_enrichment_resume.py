#!/usr/bin/env python3
from pathlib import Path
from functools import lru_cache
import re
import gzip
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = Path("/home/igor/arabidopsis_siRNA_methylation")
ANN  = ROOT / "annotation"
GO   = ANN / "GO"
OUT  = ROOT / "results/nonTE_gene_analysis"

GAF = GO / "gene_association.tair"
OBO = GO / "go-basic.obo"

RANK_FILES = {
    "nonTE": OUT / "nonTE_gene_rankings.tsv",
    "TE_distant_1kb": OUT / "TE_distant_1kb_gene_rankings.tsv"
}

WINDOW_FILES = {
    "nonTE": OUT / "all_nonTE_concordant_gene_windows.tsv",
    "TE_distant_1kb": OUT / "all_TE_distant_1kb_concordant_gene_windows.tsv"
}

support = pd.read_csv(
    OUT / "eligible_gene_background_support.tsv",
    sep="\t"
)

opp = pd.read_csv(
    OUT / "eligible_gene_backgrounds.tsv",
    sep="\t"
)

print("Reusing completed eligible backgrounds:")
print(f"  support rows = {len(support):,}")
print(f"  opportunity rows = {len(opp):,}")

# ============================================================
# Parse Araport11 descriptions
# ============================================================

print("\n===== PARSING GENE SYMBOLS =====", flush=True)

gff = ROOT / "genome/Araport11_GFF3_genes_transposons.current.gff.gz"

gene_info = {}

import gzip
with gzip.open(gff, "rt", encoding="utf-8", errors="replace") as f:
    for line in f:
        if line.startswith("#"):
            continue

        fields = line.rstrip("\n").split("\t")
        if len(fields) < 9 or fields[2] != "gene":
            continue

        attrs = {}
        for item in fields[8].split(";"):
            if "=" in item:
                k, v = item.split("=", 1)
                attrs[k] = v

        gid = attrs.get("ID")
        if not gid:
            continue

        gene_info[gid] = {
            "symbol": attrs.get("symbol", ""),
            "description": attrs.get(
                "computational_description",
                attrs.get("Note", "")
            )
        }


# ============================================================
# Parse GO ontology
# ============================================================

print("===== PARSING GO ONTOLOGY =====", flush=True)

terms = {}
current = None

with open(OBO) as f:
    for line in f:
        line = line.rstrip("\n")

        if line == "[Term]":
            if current and "id" in current:
                terms[current["id"]] = current
            current = {
                "parents": [],
                "alt_ids": [],
                "obsolete": False
            }
            continue

        if current is None:
            continue

        if line.startswith("id: "):
            current["id"] = line[4:]

        elif line.startswith("name: "):
            current["name"] = line[6:]

        elif line.startswith("namespace: "):
            current["namespace"] = line[11:]

        elif line.startswith("alt_id: "):
            current["alt_ids"].append(line[8:])

        elif line.startswith("is_a: "):
            current["parents"].append(
                line[6:].split()[0]
            )

        elif line.startswith("relationship: part_of "):
            current["parents"].append(
                line.split()[2]
            )

        elif line == "is_obsolete: true":
            current["obsolete"] = True

if current and "id" in current:
    terms[current["id"]] = current


alt_to_primary = {}
for tid, t in terms.items():
    for alt in t.get("alt_ids", []):
        alt_to_primary[alt] = tid


@lru_cache(maxsize=None)
def ancestors(goid):
    goid = alt_to_primary.get(goid, goid)

    if goid not in terms:
        return frozenset([goid])

    t = terms[goid]

    out = {goid}

    for p in t.get("parents", []):
        out.update(ancestors(p))

    return frozenset(out)


# ============================================================
# Parse GAF — Biological Process only
# ============================================================

print("===== PARSING TAIR GAF =====", flush=True)

direct = {}

with open(GAF) as f:
    for line in f:
        if line.startswith("!"):
            continue

        x = line.rstrip("\n").split("\t")
        if len(x) < 9:
            continue

        gene = x[1]
        qualifier = x[3]
        goid = x[4]
        aspect = x[8]

        if aspect != "P":
            continue

        if "NOT" in qualifier.split("|"):
            continue

        goid = alt_to_primary.get(goid, goid)

        if goid not in terms:
            continue

        if terms[goid].get("obsolete"):
            continue

        if terms[goid].get("namespace") != "biological_process":
            continue

        direct.setdefault(gene, set()).add(goid)


gene_to_go = {}

for gene, gos in direct.items():
    expanded = set()

    for goid in gos:
        expanded.update(ancestors(goid))

    expanded = {
        x for x in expanded
        if x in terms
        and terms[x].get("namespace") == "biological_process"
        and not terms[x].get("obsolete")
    }

    # Remove root BP term from testing
    expanded.discard("GO:0008150")

    gene_to_go[gene] = expanded


go_to_genes = {}

for gene, gos in gene_to_go.items():
    for goid in gos:
        go_to_genes.setdefault(goid, set()).add(gene)

print(
    f"Genes with propagated BP annotations: "
    f"{len(gene_to_go):,}",
    flush=True
)


# ============================================================
# Candidate gene sets
# ============================================================

print("\n===== BUILDING CANDIDATE SETS =====", flush=True)

candidate_sets = {}


# Tier 1
for filt, path in RANK_FILES.items():
    r = pd.read_csv(path, sep="\t")

    r = r[r["tier"] == "Tier1"].copy()

    for keys, g in r.groupby(
        ["stress", "direction", "feature"],
        observed=True
    ):
        stress, direction, feature = keys

        candidate_sets[
            (filt, stress, direction, feature, "Tier1")
        ] = set(g["gene_id"].astype(str))


# Standardized joint 1% sensitivity
for filt, path in WINDOW_FILES.items():
    w = pd.read_csv(
        path, sep="\t",
        usecols=[
            "stress", "direction", "feature",
            "gene_id", "tail_pct"
        ]
    )

    w["tail_pct"] = pd.to_numeric(
        w["tail_pct"], errors="coerce"
    )

    w = w[w["tail_pct"] == 1].copy()

    for keys, g in w.groupby(
        ["stress", "direction", "feature"],
        observed=True
    ):
        stress, direction, feature = keys

        candidate_sets[
            (filt, stress, direction, feature, "Joint1pct")
        ] = set(g["gene_id"].astype(str))


# ============================================================
# BH correction
# ============================================================

def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    n = len(p)

    order = np.argsort(p)
    ranked = p[order]

    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)

    out = np.empty(n)
    out[order] = q
    return out


# ============================================================
# GO enrichment
# ============================================================

print("\n===== RUNNING GO BP ENRICHMENT =====", flush=True)

all_results = []
set_summary = []

for key, selected_raw in sorted(candidate_sets.items()):

    filt, stress, direction, feature, setdef = key

    o = opp[
        (opp["filter"] == filt) &
        (opp["stress"] == stress) &
        (opp["feature"] == feature)
    ].copy()

    if setdef == "Tier1":
        universe = set(
            o.loc[o["eligible_tier1"], "gene_id"].astype(str)
        )
    else:
        universe = set(o["gene_id"].astype(str))

    selected = set(selected_raw) & universe

    missing_from_bg = set(selected_raw) - universe

    # GO analysis universe = eligible genes carrying at least
    # one Biological Process annotation
    universe_go = universe & set(gene_to_go)
    selected_go = selected & universe_go

    N = len(universe_go)
    n = len(selected_go)

    set_summary.append({
        "filter": filt,
        "stress": stress,
        "direction": direction,
        "feature": feature,
        "gene_set": setdef,
        "selected_total": len(selected_raw),
        "selected_in_background": len(selected),
        "selected_BP_annotated": n,
        "background_total": len(universe),
        "background_BP_annotated": N,
        "selected_missing_from_background": len(missing_from_bg)
    })

    print(
        f"{filt:15s} {stress:9s} {direction:4s} "
        f"{feature:9s} {setdef:9s}: "
        f"selected={len(selected):4d}, "
        f"BP={n:4d}, background={N:5d}",
        flush=True
    )

    if N == 0 or n < 3:
        continue

    rows = []

    for goid, global_genes in go_to_genes.items():

        bg_genes = global_genes & universe_go
        K = len(bg_genes)

        # avoid extremely tiny terms
        if K < 10:
            continue

        hit_genes = selected_go & bg_genes
        k = len(hit_genes)

        if k < 3:
            continue

        p = hypergeom.sf(k - 1, N, K, n)

        expected = n * K / N
        fold = (
            (k / n) /
            (K / N)
            if n > 0 and K > 0
            else np.nan
        )

        gene_labels = []
        for gid in sorted(hit_genes):
            sym = gene_info.get(gid, {}).get("symbol", "")
            if sym:
                gene_labels.append(f"{gid}({sym})")
            else:
                gene_labels.append(gid)

        rows.append({
            "filter": filt,
            "stress": stress,
            "direction": direction,
            "feature": feature,
            "gene_set": setdef,
            "GO_ID": goid,
            "GO_term": terms[goid].get("name", ""),
            "selected_hits": k,
            "selected_annotated": n,
            "background_hits": K,
            "background_annotated": N,
            "expected_hits": expected,
            "fold_enrichment": fold,
            "p_value": p,
            "genes": ";".join(sorted(hit_genes)),
            "genes_with_symbols": ";".join(gene_labels)
        })

    if not rows:
        continue

    z = pd.DataFrame(rows)
    z["FDR_BH"] = bh_adjust(z["p_value"].values)

    all_results.append(z)


summary = pd.DataFrame(set_summary)
summary.to_csv(
    OUT / "GO_gene_set_summary.tsv",
    sep="\t", index=False
)

if all_results:
    results = pd.concat(all_results, ignore_index=True)

    results = results.sort_values(
        [
            "filter", "stress", "direction",
            "feature", "gene_set",
            "FDR_BH", "fold_enrichment"
        ],
        ascending=[
            True, True, True,
            True, True,
            True, False
        ]
    )

    results.to_csv(
        OUT / "GO_BP_enrichment_all.tsv",
        sep="\t", index=False
    )

    sig = results[
        (results["FDR_BH"] < 0.05) &
        (results["fold_enrichment"] > 1)
    ].copy()

    sig.to_csv(
        OUT / "GO_BP_enrichment_significant.tsv",
        sep="\t", index=False
    )

else:
    results = pd.DataFrame()
    sig = pd.DataFrame()


# ============================================================
# Compact top-term report
# ============================================================

report = OUT / "GO_BP_top_terms.txt"

with open(report, "w") as f:

    f.write("NON-TE GENE GO BIOLOGICAL PROCESS ENRICHMENT\n")
    f.write("============================================\n\n")

    for _, row in summary.iterrows():
        f.write(
            f"{row['filter']} | {row['stress']} | "
            f"{row['direction']} | {row['feature']} | "
            f"{row['gene_set']}\n"
        )
        f.write(
            f"selected={row['selected_in_background']}; "
            f"BP annotated={row['selected_BP_annotated']}; "
            f"eligible BP background="
            f"{row['background_BP_annotated']}\n"
        )

        if len(sig):
            q = sig[
                (sig["filter"] == row["filter"]) &
                (sig["stress"] == row["stress"]) &
                (sig["direction"] == row["direction"]) &
                (sig["feature"] == row["feature"]) &
                (sig["gene_set"] == row["gene_set"])
            ].copy()

            q = q.sort_values(
                ["FDR_BH", "fold_enrichment"],
                ascending=[True, False]
            ).head(15)

            if len(q):
                for _, x in q.iterrows():
                    f.write(
                        f"  {x.GO_ID} | {x.GO_term} | "
                        f"{int(x.selected_hits)}/"
                        f"{int(x.selected_annotated)} | "
                        f"fold={x.fold_enrichment:.2f} | "
                        f"FDR={x.FDR_BH:.3g}\n"
                    )
            else:
                f.write("  No GO BP terms at FDR < 0.05\n")
        else:
            f.write("  No GO BP terms at FDR < 0.05\n")

        f.write("\n")


print("\n===== FINISHED =====")
print("Wrote:")
print(" ", OUT / "eligible_gene_background_support.tsv")
print(" ", OUT / "eligible_gene_backgrounds.tsv")
print(" ", OUT / "GO_gene_set_summary.tsv")
print(" ", OUT / "GO_BP_enrichment_all.tsv")
print(" ", OUT / "GO_BP_enrichment_significant.tsv")
print(" ", OUT / "GO_BP_top_terms.txt")

if len(sig):
    print(
        f"\nSignificant enriched GO BP rows: {len(sig):,}"
    )
else:
    print("\nNo significant enriched GO BP rows.")
