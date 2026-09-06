#!/usr/bin/env python3

import argparse
import subprocess
import tempfile
from pathlib import Path
from functools import lru_cache
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path("/home/igor/arabidopsis_siRNA_methylation")
OUT  = ROOT / "results/nonTE_gene_analysis"
ANN  = ROOT / "annotation"
GO   = ANN / "GO"

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

DELTA = {
    ("CG","6h"):   "deltaCG_6h",
    ("CG","12h"):  "deltaCG_12h",
    ("CG","24h"):  "deltaCG_24h",
    ("CHG","6h"):  "deltaCHG_6h",
    ("CHG","12h"): "deltaCHG_12h",
    ("CHG","24h"): "deltaCHG_24h",
    ("CHH","6h"):  "deltaCHH_6h",
    ("CHH","12h"): "deltaCHH_12h",
    ("CHH","24h"): "deltaCHH_24h",
}

GENES = ANN / "Araport11_genes.bed"
TES   = ANN / "TAIR10_transposable_elements.bed"
GAF   = GO / "gene_association.tair"
OBO   = GO / "go-basic.obo"

CANDIDATES = OUT / "all_TE_distant_1kb_concordant_gene_windows.tsv"

parser = argparse.ArgumentParser()
parser.add_argument("--nperm", type=int, default=200)
parser.add_argument("--seed", type=int, default=20260830)
parser.add_argument("--batch", type=int, default=200)
args = parser.parse_args()

rng = np.random.default_rng(args.seed)

# ============================================================
# GO ontology
# ============================================================

print("Reading GO ontology...", flush=True)

terms = {}
cur = None

with open(OBO, encoding="utf-8") as f:
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

for gene,gset in direct.items():
    z = set()

    for go in gset:
        z.update(ancestors(go))

    z = {
        go for go in z
        if go in terms
        and terms[go].get("namespace") == "biological_process"
        and not terms[go].get("obsolete")
    }

    z.discard("GO:0008150")
    gene2go[gene] = z

print(
    f"Genes with BP annotation: {len(gene2go):,}",
    flush=True
)

# ============================================================
# Build TE ±1 kb exclusion BED
# ============================================================

cache = OUT / "heat_window_opportunity_cache"
cache.mkdir(exist_ok=True)

genome_file = cache / "heat_chrom_sizes.txt"
te1kb_file = cache / "TAIR10_TEs_plus_1kb_merged.bed"

if not genome_file.exists():
    x = pd.read_csv(
        SRNA[100],
        sep="\t",
        usecols=["chr","end"]
    )

    sizes = (
        x.groupby("chr")["end"]
        .max()
        .reset_index()
    )

    sizes.to_csv(
        genome_file,
        sep="\t",
        header=False,
        index=False
    )

if not te1kb_file.exists():

    tmp = cache / "TE_plus1kb_raw.bed"

    subprocess.run(
        [
            "bedtools","slop",
            "-i",str(TES),
            "-g",str(genome_file),
            "-b","1000"
        ],
        check=True,
        stdout=open(tmp,"w")
    )

    subprocess.run(
        [
            "bedtools","sort",
            "-i",str(tmp)
        ],
        check=True,
        stdout=open(cache/"TE_plus1kb_sorted.bed","w")
    )

    subprocess.run(
        [
            "bedtools","merge",
            "-i",str(cache/"TE_plus1kb_sorted.bed")
        ],
        check=True,
        stdout=open(te1kb_file,"w")
    )

# ============================================================
# Reconstruct eligible gene-body windows
# ============================================================

def make_mapping(context, resolution, time):

    tag = f"{context}_{resolution}_{time}"
    mapped_file = cache / f"eligible_{tag}_gene_map.tsv"

    if mapped_file.exists():
        return pd.read_csv(mapped_file, sep="\t")

    print(f"Building {tag}...", flush=True)

    s = pd.read_csv(
        SRNA[resolution],
        sep="\t",
        usecols=["chr","start","end","delta24_CPM"]
    )

    mcol = DELTA[(context,time)]

    m = pd.read_csv(
        METH[(context,resolution)],
        sep="\t",
        usecols=["chr","start","end",mcol]
    )

    s["chr"] = s["chr"].astype(str)
    m["chr"] = m["chr"].astype(str)

    d = s.merge(
        m,
        on=["chr","start","end"],
        how="inner"
    )

    d["delta24_CPM"] = pd.to_numeric(
        d["delta24_CPM"],
        errors="coerce"
    )

    d[mcol] = pd.to_numeric(
        d[mcol],
        errors="coerce"
    )

    d = d[
        np.isfinite(d["delta24_CPM"]) &
        np.isfinite(d[mcol]) &
        (d[mcol].abs() <= 1)
    ].copy()

    d["window_id"] = (
        d["chr"].astype(str) + ":" +
        d["start"].astype(str) + "-" +
        d["end"].astype(str)
    )

    bed0 = cache / f"{tag}.valid.bed"
    bed1 = cache / f"{tag}.TEdistant.bed"
    bed2 = cache / f"{tag}.gene_intersect.bed"

    d[
        ["chr","start","end","window_id"]
    ].to_csv(
        bed0,
        sep="\t",
        header=False,
        index=False
    )

    with open(bed1,"w") as fh:
        subprocess.run(
            [
                "bedtools","intersect",
                "-a",str(bed0),
                "-b",str(te1kb_file),
                "-v"
            ],
            check=True,
            stdout=fh
        )

    with open(bed2,"w") as fh:
        subprocess.run(
            [
                "bedtools","intersect",
                "-a",str(bed1),
                "-b",str(GENES),
                "-wa","-wb"
            ],
            check=True,
            stdout=fh
        )

    # A has 4 columns.
    # Gene BED:
    # chr start end gene_id strand
    z = pd.read_csv(
        bed2,
        sep="\t",
        header=None
    )

    if z.shape[1] < 9:
        raise RuntimeError(
            f"Unexpected gene intersection format: "
            f"{z.shape[1]} columns"
        )

    z = z.iloc[:,[0,1,2,3,7]]
    z.columns = [
        "chr","start","end",
        "window_id","gene_id"
    ]

    z = z.drop_duplicates(
        ["window_id","gene_id"]
    )

    z.to_csv(
        mapped_file,
        sep="\t",
        index=False
    )

    print(
        f"  {len(d):,} jointly measurable windows; "
        f"{z.window_id.nunique():,} TE-distant "
        f"gene-body windows; "
        f"{z.gene_id.nunique():,} genes",
        flush=True
    )

    return z


maps = {}

for context in ["CG","CHG","CHH"]:
    for resolution in [100,500]:
        for time in ["6h","12h","24h"]:
            maps[(context,resolution,time)] = \
                make_mapping(context,resolution,time)

# ============================================================
# Observed joint-1% candidate windows
# ============================================================

cand = pd.read_csv(
    CANDIDATES,
    sep="\t",
    low_memory=False
)

cand["tail_pct"] = pd.to_numeric(
    cand["tail_pct"],
    errors="coerce"
)

cand = cand[
    (cand["stress"] == "heat") &
    (cand["feature"] == "gene_body") &
    (cand["tail_pct"] == 1)
].copy()

cand["chr"] = cand["chr"].astype(str)

cand["window_id"] = (
    cand["chr"].astype(str) + ":" +
    cand["start"].astype(str) + "-" +
    cand["end"].astype(str)
)

print(
    "\nObserved candidate structure:",
    flush=True
)

for context in ["CG","CHG","CHH"]:
    for direction in ["gain","loss"]:
        print(
            f"\n{context} {direction}",
            flush=True
        )

        for resolution in [100,500]:
            for time in ["6h","12h","24h"]:

                q = cand[
                    (cand["context"] == context) &
                    (cand["direction"] == direction) &
                    (cand["resolution"] == resolution) &
                    (cand["time"].astype(str) == time)
                ]

                nwin = q["window_id"].nunique()

                print(
                    f"  {resolution:3d} {time}: "
                    f"{nwin:4d} windows",
                    flush=True
                )

# ============================================================
# GO sparse matrix helper
# ============================================================

def BH(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    n = len(p)

    q = p[order] * n / np.arange(1,n+1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q,1)

    out = np.empty(n)
    out[order] = q
    return out


results = []

# ============================================================
# Permutation separately by context + direction
# ============================================================

for context in ["CG","CHG","CHH"]:

    # Universe of all genes that could be hit
    allgenes = set()

    for resolution in [100,500]:
        for time in ["6h","12h","24h"]:
            allgenes.update(
                maps[
                    (context,resolution,time)
                ]["gene_id"].astype(str)
            )

    allgenes &= set(gene2go)

    gene_list = sorted(allgenes)
    gene_index = {
        g:i for i,g in enumerate(gene_list)
    }

    # GO terms available in this universe
    term_genes = defaultdict(set)

    for g in gene_list:
        for go in gene2go[g]:
            term_genes[go].add(g)

    go_list = sorted(
        go for go,gs in term_genes.items()
        if len(gs) >= 10
    )

    go_index = {
        go:i for i,go in enumerate(go_list)
    }

    rr = []
    cc = []

    for g in gene_list:
        gi = gene_index[g]

        for go in gene2go[g]:
            if go in go_index:
                rr.append(gi)
                cc.append(go_index[go])

    GOmat = sparse.csr_matrix(
        (
            np.ones(len(rr), dtype=np.int32),
            (rr,cc)
        ),
        shape=(len(gene_list),len(go_list))
    )

    # For each stratum:
    # unique windows + genes overlapping each window
    pools = {}

    for resolution in [100,500]:
        for time in ["6h","12h","24h"]:

            z = maps[(context,resolution,time)]

            z = z[
                z["gene_id"].isin(gene_index)
            ].copy()

            grouped = (
                z.groupby("window_id")["gene_id"]
                .apply(list)
            )

            win_ids = grouped.index.to_numpy()

            gene_arrays = [
                np.array(
                    [gene_index[g] for g in gs],
                    dtype=np.int32
                )
                for gs in grouped.values
            ]

            pools[(resolution,time)] = (
                win_ids,
                gene_arrays
            )

    for direction in ["gain","loss"]:

        print(
            f"\nPERMUTING {context} {direction}",
            flush=True
        )

        observed = cand[
            (cand["context"] == context) &
            (cand["direction"] == direction)
        ]

        observed_genes = set(
            observed["gene_id"].astype(str)
        ) & set(gene_index)

        obs_idx = np.array(
            [gene_index[g] for g in observed_genes],
            dtype=np.int32
        )

        obs_hits = np.asarray(
            GOmat[obs_idx].sum(axis=0)
        ).ravel()

        # Only terms with >=3 observed genes
        test_mask = obs_hits >= 3
        test_cols = np.where(test_mask)[0]

        if not len(test_cols):
            continue

        exceed = np.zeros(
            len(test_cols),
            dtype=np.int64
        )

        null_sum = np.zeros(
            len(test_cols),
            dtype=np.float64
        )

        null_sumsq = np.zeros(
            len(test_cols),
            dtype=np.float64
        )

        # Number of observed unique windows to draw
        ndraw = {}

        for resolution in [100,500]:
            for time in ["6h","12h","24h"]:

                q = observed[
                    (observed["resolution"] == resolution) &
                    (observed["time"].astype(str) == time)
                ]

                ndraw[(resolution,time)] = \
                    q["window_id"].nunique()

        done = 0

        while done < args.nperm:

            B = min(
                args.batch,
                args.nperm-done
            )

            prow = []
            pcol = []

            for bi in range(B):

                genes = set()

                for key,k in ndraw.items():

                    if k == 0:
                        continue

                    win_ids,gene_arrays = pools[key]

                    if k > len(win_ids):
                        raise RuntimeError(
                            f"{context} {direction} {key}: "
                            f"draw {k} > pool {len(win_ids)}"
                        )

                    chosen = rng.choice(
                        len(win_ids),
                        size=k,
                        replace=False
                    )

                    for wi in chosen:
                        genes.update(
                            gene_arrays[wi].tolist()
                        )

                if genes:
                    prow.extend([bi]*len(genes))
                    pcol.extend(genes)

            P = sparse.csr_matrix(
                (
                    np.ones(len(prow),dtype=np.int32),
                    (prow,pcol)
                ),
                shape=(B,len(gene_list))
            )

            counts = (
                P @ GOmat[:,test_cols]
            ).toarray()

            obs_test = obs_hits[test_cols]

            exceed += np.sum(
                counts >= obs_test,
                axis=0
            )

            null_sum += counts.sum(axis=0)
            null_sumsq += (
                counts.astype(float)**2
            ).sum(axis=0)

            done += B

            if (
                done == args.nperm or
                done % max(args.batch*5,1000) == 0
            ):
                print(
                    f"  {done:,}/{args.nperm:,}",
                    flush=True
                )

        mean = null_sum / args.nperm

        var = (
            null_sumsq / args.nperm
            - mean**2
        )

        var[var < 0] = 0
        sd = np.sqrt(var)

        pemp = (
            exceed + 1
        ) / (
            args.nperm + 1
        )

        tmp = []

        for j,col in enumerate(test_cols):

            go = go_list[col]
            obs_n = int(obs_hits[col])

            tmp.append({
                "context": context,
                "direction": direction,
                "GO_ID": go,
                "GO_term":
                    terms[go].get("name",""),
                "observed_hits": obs_n,
                "null_mean_hits": mean[j],
                "null_sd": sd[j],
                "fold_vs_window_null":
                    (
                        obs_n/mean[j]
                        if mean[j] > 0
                        else np.inf
                    ),
                "empirical_p": pemp[j],
                "null_ge_observed":
                    int(exceed[j]),
                "n_observed_genes":
                    len(observed_genes),
                "nperm": args.nperm
            })

        t = pd.DataFrame(tmp)
        t["FDR_BH"] = BH(
            t["empirical_p"].to_numpy()
        )

        results.append(t)

        sig = t[
            (t["FDR_BH"] < 0.05) &
            (t["fold_vs_window_null"] > 1)
        ].sort_values(
            ["FDR_BH",
             "fold_vs_window_null"],
            ascending=[True,False]
        )

        print(
            f"  observed genes: "
            f"{len(observed_genes)}",
            flush=True
        )

        print(
            f"  significant BP terms: "
            f"{len(sig)}",
            flush=True
        )

        if len(sig):
            print(
                sig[
                    [
                        "GO_ID","GO_term",
                        "observed_hits",
                        "null_mean_hits",
                        "fold_vs_window_null",
                        "empirical_p",
                        "FDR_BH"
                    ]
                ].head(15).to_string(
                    index=False
                ),
                flush=True
            )

# ============================================================
# Save
# ============================================================

res = pd.concat(
    results,
    ignore_index=True
)

outfile = OUT / (
    f"heat_window_opportunity_GO_"
    f"{args.nperm}perm.tsv"
)

res.to_csv(
    outfile,
    sep="\t",
    index=False
)

sigfile = OUT / (
    f"heat_window_opportunity_GO_"
    f"{args.nperm}perm_significant.tsv"
)

res[
    (res["FDR_BH"] < 0.05) &
    (res["fold_vs_window_null"] > 1)
].to_csv(
    sigfile,
    sep="\t",
    index=False
)

print("\n====================================")
print("KEY TERMS")
print("====================================")

targets = {
    "GO:0009408",
    "GO:0006457",
    "GO:0015979",
    "GO:0009765",
    "GO:0006091"
}

x = res[
    res["GO_ID"].isin(targets)
].sort_values(
    ["context","direction","GO_ID"]
)

print(
    x.to_string(index=False),
    flush=True
)

print(f"\nSaved: {outfile}")
print(f"Saved: {sigfile}")
print("DONE")
