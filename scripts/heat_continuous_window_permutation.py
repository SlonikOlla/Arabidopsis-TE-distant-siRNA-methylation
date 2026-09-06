#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
import argparse
import numpy as np
import pandas as pd

ROOT = Path("/home/igor/arabidopsis_siRNA_methylation")
OUT = ROOT / "results/nonTE_gene_analysis"
CACHE = OUT / "heat_window_opportunity_cache_gene_body"
PREV = OUT / "heat_continuous_pathway"
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
parser.add_argument("--nperm", type=int, default=1000)
parser.add_argument("--seed", type=int, default=20260831)
args = parser.parse_args()

DEST = OUT / "heat_continuous_window_permutation"
DEST.mkdir(exist_ok=True)

# ---------- GO ----------
parents = defaultdict(set)
namespace = {}
current = None

with open(GO_DIR / "go-basic.obo") as fh:
    for line in fh:
        line = line.rstrip()
        if line == "[Term]":
            current = None
        elif line.startswith("id: GO:"):
            current = line.split("id: ",1)[1]
        elif current and line.startswith("namespace: "):
            namespace[current] = line.split("namespace: ",1)[1]
        elif current and line.startswith("is_a: GO:"):
            parents[current].add(line.split()[1])
        elif current and line.startswith("relationship: part_of GO:"):
            parents[current].add(line.split()[2])

acache = {}
def ancestors(go):
    if go in acache:
        return acache[go]
    z = {go}
    for p in parents.get(go,()):
        z |= ancestors(p)
    acache[go] = z
    return z

gene2go = defaultdict(set)
with open(GO_DIR / "gene_association.tair") as fh:
    for line in fh:
        if line.startswith("!"):
            continue
        x = line.rstrip("\n").split("\t")
        if len(x) < 5 or "NOT" in x[3].split("|"):
            continue
        gene, go = x[1], x[4]
        if namespace.get(go) != "biological_process":
            continue
        for a in ancestors(go):
            if namespace.get(a) == "biological_process" and a != "GO:0008150":
                gene2go[gene].add(a)

target_genes = {
    go: {g for g,terms in gene2go.items() if go in terms}
    for go in TARGETS
}

# ---------- prepare each context ----------
rng = np.random.default_rng(args.seed)
all_results = []

for context in ["CG","CHG","CHH"]:
    print(f"\nPreparing {context}...", flush=True)

    # One table per resolution. Rows are gene-window mappings.
    blocks = {}

    for resolution in [100,500]:
        s = pd.read_csv(
            SRNA[resolution], sep="\t",
            usecols=["chr","start","end","delta24_CPM"]
        )
        s["chr"] = s["chr"].astype(str)
        s["window_id"] = (
            s["chr"] + ":" + s["start"].astype(str) + "-" + s["end"].astype(str)
        )
        s["delta24_CPM"] = pd.to_numeric(s["delta24_CPM"], errors="coerce")

        # Use 6h cache as base; eligible sets differ only trivially.
        maps = {}
        score_tables = {}

        for time in ["6h","12h","24h"]:
            mp = pd.read_csv(
                CACHE / f"eligible_{context}_{resolution}_{time}_gene_map.tsv",
                sep="\t", usecols=["window_id","gene_id"]
            ).drop_duplicates()
            maps[time] = mp

            mcol = f"delta{context}_{time}"
            m = pd.read_csv(
                METH[(context,resolution)], sep="\t",
                usecols=["chr","start","end",mcol]
            )
            m["chr"] = m["chr"].astype(str)
            m["window_id"] = (
                m["chr"] + ":" + m["start"].astype(str) + "-" + m["end"].astype(str)
            )
            m[mcol] = pd.to_numeric(m[mcol], errors="coerce")

            d = s[["window_id","delta24_CPM"]].merge(
                m[["window_id",mcol]], on="window_id"
            )
            ids = mp[["window_id"]].drop_duplicates()
            d = ids.merge(d,on="window_id")
            d = d[
                np.isfinite(d.delta24_CPM) &
                np.isfinite(d[mcol]) &
                (d[mcol].abs() <= 1)
            ].copy()

            rg24 = d.delta24_CPM.rank(pct=True,ascending=True)
            rgm  = d[mcol].rank(pct=True,ascending=True)
            rl24 = d.delta24_CPM.rank(pct=True,ascending=False)
            rlm  = d[mcol].rank(pct=True,ascending=False)

            d["gain"] = 0.0
            d["loss"] = 0.0

            q = (d.delta24_CPM > 0) & (d[mcol] > 0)
            d.loc[q,"gain"] = rg24[q] * rgm[q]

            q = (d.delta24_CPM < 0) & (d[mcol] < 0)
            d.loc[q,"loss"] = rl24[q] * rlm[q]

            score_tables[time] = d.set_index("window_id")[["gain","loss"]]

        blocks[resolution] = (maps,score_tables)

    # Build fixed gene universe.
    genes = sorted(set().union(*[
        set(mp.gene_id)
        for resolution in blocks
        for mp in blocks[resolution][0].values()
    ]))
    gindex = {g:i for i,g in enumerate(genes)}
    ng = len(genes)

    for direction in ["gain","loss"]:
        print(f"  {context} {direction}: {ng:,} genes", flush=True)

        # Observed sum and number of strata contributing per gene.
        obs_sum = np.zeros(ng)
        obs_n = np.zeros(ng)

        prepared = []

        for resolution in [100,500]:
            maps, score_tables = blocks[resolution]

            # Use union of window IDs so one shared permutation can be
            # applied to 6/12/24h within this resolution.
            union_ids = sorted(set().union(*[
                set(x.index) for x in score_tables.values()
            ]))
            uid = {w:i for i,w in enumerate(union_ids)}
            nw = len(union_ids)

            time_data = []

            for time in ["6h","12h","24h"]:
                mp = maps[time]
                st = score_tables[time]

                # Window score vector on union universe.
                wscore = np.zeros(nw)
                present = np.zeros(nw,dtype=bool)
                for w,v in st[direction].items():
                    j = uid[w]
                    wscore[j] = v
                    present[j] = True

                # Mapping arrays: one row per gene-window overlap.
                mm = mp[mp.window_id.isin(st.index)].copy()
                wi = mm.window_id.map(uid).to_numpy()
                gi = mm.gene_id.map(gindex).to_numpy()

                # observed gene mean for this stratum
                sums = np.bincount(
                    gi, weights=wscore[wi], minlength=ng
                )
                cnt = np.bincount(gi,minlength=ng)
                gm = np.divide(
                    sums,cnt,
                    out=np.zeros(ng),
                    where=cnt>0
                )
                obs_sum += gm
                obs_n += (cnt>0)

                time_data.append((wi,gi,cnt,present))

            prepared.append((nw,score_tables,time_data,uid))

        observed_gene = np.divide(
            obs_sum,obs_n,
            out=np.zeros(ng),
            where=obs_n>0
        )

        # Target pathway observed means.
        observed_path = {}
        target_idx = {}

        for go,label in TARGETS.items():
            idx = np.array(
                [gindex[g] for g in target_genes[go] if g in gindex],
                dtype=int
            )
            target_idx[go] = idx
            observed_path[go] = observed_gene[idx].mean()

        exceed = {go:0 for go in TARGETS}
        null_sum = {go:0.0 for go in TARGETS}

        # Permutation: same window permutation across 6/12/24h
        # within each resolution.
        for ip in range(args.nperm):
            perm_sum = np.zeros(ng)
            perm_n = np.zeros(ng)

            for nw,score_tables,time_data,uid in prepared:
                perm = rng.permutation(nw)

                for time,(wi,gi,cnt,present) in zip(
                    ["6h","12h","24h"],time_data
                ):
                    st = score_tables[time]
                    original = np.zeros(nw)

                    for w,v in st[direction].items():
                        original[uid[w]] = v

                    shuffled = original[perm]

                    sums = np.bincount(
                        gi,weights=shuffled[wi],minlength=ng
                    )
                    gm = np.divide(
                        sums,cnt,
                        out=np.zeros(ng),
                        where=cnt>0
                    )
                    perm_sum += gm
                    perm_n += (cnt>0)

            pg = np.divide(
                perm_sum,perm_n,
                out=np.zeros(ng),
                where=perm_n>0
            )

            for go in TARGETS:
                val = pg[target_idx[go]].mean()
                null_sum[go] += val
                if val >= observed_path[go]:
                    exceed[go] += 1

            if (ip+1) % 100 == 0:
                print(
                    f"    permutation {ip+1}/{args.nperm}",
                    flush=True
                )

        for go,label in TARGETS.items():
            nm = null_sum[go] / args.nperm
            op = observed_path[go]
            p = (exceed[go]+1)/(args.nperm+1)

            all_results.append({
                "context":context,
                "direction":direction,
                "GO_ID":go,
                "GO_term":label,
                "n_genes":len(target_idx[go]),
                "observed_mean":op,
                "null_mean":nm,
                "fold":op/nm if nm>0 else np.nan,
                "empirical_P":p
            })

res = pd.DataFrame(all_results)

# BH over four predefined pathways within context/direction.
res["targeted_FDR"] = np.nan
for _,idx in res.groupby(["context","direction"]).groups.items():
    idx = list(idx)
    p = res.loc[idx,"empirical_P"].to_numpy()
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1,len(p)+1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(len(p))
    out[order] = np.minimum(q,1)
    res.loc[idx,"targeted_FDR"] = out

res.to_csv(
    DEST / f"heat_continuous_window_permutation_{args.nperm}perm.tsv",
    sep="\t",index=False
)

print("\nHEAT STRICT CONTINUOUS WINDOW-PERMUTATION")
for _,r in res.sort_values(
    ["context","direction","empirical_P"]
).iterrows():
    print(
        f"\n{r.GO_term} | {r.context} {r.direction}\n"
        f"  genes={int(r.n_genes)} "
        f"obs={r.observed_mean:.6f} "
        f"null={r.null_mean:.6f} "
        f"fold={r.fold:.3f} "
        f"P={r.empirical_P:.6g} "
        f"FDR={r.targeted_FDR:.6g}"
    )

print("\nSaved to:",DEST)
