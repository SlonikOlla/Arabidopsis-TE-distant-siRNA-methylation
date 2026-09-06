#!/usr/bin/env python3

from pathlib import Path
from functools import lru_cache
import subprocess
import tempfile
import re
import math

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = Path("/home/igor/arabidopsis_siRNA_methylation")
ANN  = ROOT / "annotation"
GO   = ANN / "GO"
OUT  = ROOT / "results/nonTE_gene_analysis"
OUT.mkdir(parents=True, exist_ok=True)

TE_BED       = ANN / "Araport11_TEs.bed"
GENE_BED     = ANN / "Araport11_genes.bed"
PROMOTER_BED = ANN / "Araport11_promoters_1kb.bed"

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

contexts = ["CG", "CHG", "CHH"]
resolutions = [100, 500]

stress_times = {
    "drought": ["single"],
    "heat": ["6h", "12h", "24h"],
    "pathogen": ["6h", "14h"],
    "phosphate": ["single"]
}


# ============================================================
# Helpers
# ============================================================

def norm_chr(x):
    x = str(x)
    x = re.sub(r"^(Chr|chr)", "", x)
    return x


def read_coords_delta(path, delta_col):
    d = pd.read_csv(
        path, sep="\t",
        usecols=["chr", "start", "end", delta_col]
    )
    d = d.rename(columns={delta_col: "delta"})
    d["chr"] = d["chr"].map(norm_chr)
    d["start"] = pd.to_numeric(d["start"], errors="coerce")
    d["end"] = pd.to_numeric(d["end"], errors="coerce")
    d["delta"] = pd.to_numeric(d["delta"], errors="coerce")

    d = d[
        d["start"].notna() &
        d["end"].notna() &
        np.isfinite(d["delta"])
    ].copy()

    d["start"] = d["start"].astype(int)
    d["end"] = d["end"].astype(int)
    return d


def discover_srna(stress, res):
    if stress == "phosphate":
        p = ROOT / f"results/root_siRNA/root_24nt_{res}bp_matrix.tsv"
        if p.exists():
            return p
        raise FileNotFoundError(p)

    candidates = [
        ROOT / f"{stress}_consensus/results/{stress}_24nt_{res}bp_matrix.tsv",
        ROOT / f"results/{stress}_siRNA/{stress}_24nt_{res}bp_matrix.tsv",
    ]

    for p in candidates:
        if p.exists():
            return p

    hits = sorted(ROOT.glob(f"**/*{stress}*24nt*{res}bp*matrix.tsv"))
    if not hits:
        hits = sorted(ROOT.glob(f"**/*24nt*{res}bp*matrix.tsv"))

    stress_hits = [
        p for p in hits
        if stress.lower() in str(p).lower()
    ]

    if len(stress_hits) == 1:
        return stress_hits[0]

    if len(stress_hits) > 1:
        print(f"\nMultiple sRNA candidates for {stress} {res} bp:")
        for p in stress_hits:
            print(" ", p)
        raise RuntimeError("Ambiguous sRNA matrix")

    raise FileNotFoundError(
        f"No 24-nt matrix found for {stress}, {res} bp"
    )


def srna_delta(stress, res, time):
    path = discover_srna(stress, res)
    h = pd.read_csv(path, sep="\t", nrows=0)
    cols = list(h.columns)

    # Ordinary matrices with an existing delta column
    if stress != "pathogen":
        for c in ["delta24_CPM", "delta_CPM", "delta24"]:
            if c in cols:
                d = read_coords_delta(path, c)
                return d, path

        raise KeyError(
            f"No recognized 24-nt delta column in {path}\n"
            f"Columns: {cols}"
        )

    # Pathogen: derive 6h or 14h contrast
    d = pd.read_csv(path, sep="\t")
    d["chr"] = d["chr"].map(norm_chr)

    if time == "6h":
        mock = [
            "GSM491567_mock_6hpi_rep1",
            "GSM491568_mock_6hpi_rep2"
        ]
        ev = "GSM491570_ev_6hpi"
    elif time == "14h":
        mock = [
            "GSM491572_mock_14hpi_rep1",
            "GSM491573_mock_14hpi_rep2"
        ]
        ev = "GSM491576_ev_14hpi"
    else:
        raise ValueError(time)

    required = ["chr", "start", "end"] + mock + [ev]
    missing = [x for x in required if x not in d.columns]
    if missing:
        raise KeyError(
            f"Missing pathogen sRNA columns in {path}: {missing}"
        )

    for c in mock + [ev]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["delta"] = (
        d[ev] -
        d[mock].mean(axis=1)
    )

    d["start"] = pd.to_numeric(d["start"], errors="coerce")
    d["end"] = pd.to_numeric(d["end"], errors="coerce")

    d = d[
        d["start"].notna() &
        d["end"].notna() &
        np.isfinite(d["delta"])
    ][["chr", "start", "end", "delta"]].copy()

    d["start"] = d["start"].astype(int)
    d["end"] = d["end"].astype(int)

    return d, path


def methyl_path(stress, context, res):
    if stress == "phosphate":
        return ROOT / (
            f"results/root_methylation/"
            f"root_{context}_{res}bp_matrix.tsv"
        )

    return ROOT / (
        f"{stress}_consensus/results/"
        f"{stress}_{context}_{res}bp_matrix.tsv"
    )


def methyl_delta(stress, context, res, time):
    path = methyl_path(stress, context, res)
    h = pd.read_csv(path, sep="\t", nrows=0)
    cols = list(h.columns)

    if stress == "heat":
        wanted = f"delta{context}_{time}"
        if wanted not in cols:
            raise KeyError(
                f"{wanted} missing in {path}; columns={cols}"
            )
        col = wanted

    else:
        alternatives = [
            f"delta{context}",
            f"delta_{context}"
        ]
        col = next((x for x in alternatives if x in cols), None)
        if col is None:
            raise KeyError(
                f"No delta column for {stress} {context} "
                f"{res} bp; columns={cols}"
            )

    d = read_coords_delta(path, col)

    # methylation fractions must give differences within [-1,1].
    # This also removes legacy sentinel values such as -999.
    d = d[d["delta"].between(-1, 1, inclusive="both")].copy()

    return d, path


def run_cmd(cmd):
    subprocess.run(cmd, check=True)


def write_bed(df, path):
    x = df[["chr", "start", "end"]].drop_duplicates().copy()
    x.to_csv(path, sep="\t", header=False, index=False)


def filter_bed(inbed, outbed, filter_name):
    if filter_name == "nonTE":
        run_cmd([
            "bedtools", "intersect",
            "-a", str(inbed),
            "-b", str(TE_BED),
            "-v"
        ],)
        # subprocess above cannot redirect, do directly below
    else:
        pass


def bedtools_to_file(cmd, outfile):
    with open(outfile, "w") as f:
        subprocess.run(cmd, stdout=f, check=True)


def genes_for_feature(bedfile, annotation):
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".tsv", delete=False
    ) as tmp:
        out = Path(tmp.name)

    try:
        bedtools_to_file([
            "bedtools", "intersect",
            "-a", str(bedfile),
            "-b", str(annotation),
            "-wa", "-wb"
        ], out)

        if out.stat().st_size == 0:
            return set()

        z = pd.read_csv(out, sep="\t", header=None)

        # A has 3 columns; B gene/promoter BED has:
        # chr,start,end,gene_id,strand
        if z.shape[1] < 7:
            raise RuntimeError(
                f"Unexpected intersect output: {z.shape[1]} columns"
            )

        return set(z.iloc[:, 6].astype(str))
    finally:
        out.unlink(missing_ok=True)


# ============================================================
# Build measurable gene backgrounds
# ============================================================

print("\n===== BUILDING ELIGIBLE BACKGROUNDS =====", flush=True)

support_rows = []
srna_cache = {}

with tempfile.TemporaryDirectory() as td:
    td = Path(td)

    for stress, times in stress_times.items():
        for res in resolutions:

            # one sRNA matrix can be reused for contexts
            for time in times:
                key = (stress, res, time)

                if key not in srna_cache:
                    s, spath = srna_delta(stress, res, time)
                    srna_cache[key] = s
                    print(
                        f"sRNA {stress} {time} {res}: "
                        f"{spath.name} n={len(s):,}",
                        flush=True
                    )

                s = srna_cache[key]

                for context in contexts:
                    m, mpath = methyl_delta(
                        stress, context, res, time
                    )

                    j = s.merge(
                        m,
                        on=["chr", "start", "end"],
                        how="inner",
                        suffixes=("_24", "_meth")
                    )

                    j = j[
                        np.isfinite(j["delta_24"]) &
                        np.isfinite(j["delta_meth"])
                    ].copy()

                    coords = (
                        j[["chr", "start", "end"]]
                        .drop_duplicates()
                    )

                    print(
                        f"joint {stress:9s} {time:6s} "
                        f"{context:3s} {res:3d} bp: "
                        f"{len(coords):,}",
                        flush=True
                    )

                    rawbed = td / (
                        f"{stress}_{time}_{context}_{res}_raw.bed"
                    )
                    write_bed(coords, rawbed)

                    for filt in ["nonTE", "TE_distant_1kb"]:

                        filtbed = td / (
                            f"{stress}_{time}_{context}_{res}_{filt}.bed"
                        )

                        if filt == "nonTE":
                            bedtools_to_file([
                                "bedtools", "intersect",
                                "-a", str(rawbed),
                                "-b", str(TE_BED),
                                "-v"
                            ], filtbed)

                        else:
                            # nearest TE must be >=1000 bp away.
                            # -w 999 removes anything 0–999 bp away.
                            bedtools_to_file([
                                "bedtools", "window",
                                "-a", str(rawbed),
                                "-b", str(TE_BED),
                                "-w", "999",
                                "-v"
                            ], filtbed)

                        for feature, ann in [
                            ("gene_body", GENE_BED),
                            ("promoter", PROMOTER_BED)
                        ]:
                            genes = genes_for_feature(filtbed, ann)

                            for gene in genes:
                                support_rows.append({
                                    "filter": filt,
                                    "stress": stress,
                                    "context": context,
                                    "resolution": res,
                                    "time": time,
                                    "feature": feature,
                                    "gene_id": gene
                                })


support = pd.DataFrame(support_rows).drop_duplicates()

support.to_csv(
    OUT / "eligible_gene_background_support.tsv",
    sep="\t", index=False
)

print(
    f"\nBackground support records: {len(support):,}",
    flush=True
)


# Collapse to gene-level opportunity
opp = (
    support.groupby(
        ["filter", "stress", "feature", "gene_id"],
        observed=True
    )
    .agg(
        n_contexts=("context", "nunique"),
        n_resolutions=("resolution", "nunique"),
        n_times=("time", "nunique")
    )
    .reset_index()
)

opp["eligible_any"] = True

opp["eligible_tier1"] = (
    (opp["n_resolutions"] == 2) &
    (
        (opp["n_contexts"] >= 2) |
        (opp["n_times"] >= 2)
    )
)

opp.to_csv(
    OUT / "eligible_gene_backgrounds.tsv",
    sep="\t", index=False
)


print("\nEligible genes by stress/filter/feature:")
print(
    opp.groupby(["filter", "stress", "feature"])
       ["gene_id"].nunique()
       .to_string()
)


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
