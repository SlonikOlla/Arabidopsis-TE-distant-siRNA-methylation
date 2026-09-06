#!/usr/bin/env python3

import numpy as np
import pandas as pd

SEED = 20260829
N_PERM = 10000
NBINS = 10
TAILS = [0.01, 0.05, 0.10]

OUT = "results/all_stresses_CG_CHG_extreme_tail_EXACT_RANK_10k.tsv"

rng = np.random.default_rng(SEED)


def bh(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out


def rank_bins(x, nbins=10):
    x = np.asarray(x, dtype=float)
    n = len(x)

    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)

    b = np.floor(ranks * nbins / n).astype(int)
    b[b == nbins] = nbins - 1
    return b


def exact_tail(x, frac, direction):
    x = np.asarray(x, dtype=float)
    n = len(x)
    k = int(np.floor(n * frac))

    if k < 1:
        return np.array([], dtype=int)

    order = np.argsort(x, kind="mergesort")

    if direction == "loss":
        return order[:k]
    elif direction == "gain":
        return order[-k:]
    else:
        raise ValueError(direction)


def empirical_test(selected, counterpart, counterpart_direction, strata):

    selected = np.asarray(selected, dtype=int)
    counterpart = np.asarray(counterpart, dtype=float)
    strata = np.asarray(strata)

    if counterpart_direction == "gain":
        target = counterpart > 0
    elif counterpart_direction == "loss":
        target = counterpart < 0
    else:
        raise ValueError(counterpart_direction)

    selected_mask = np.zeros(len(counterpart), dtype=bool)
    selected_mask[selected] = True

    observed = int(np.sum(target & selected_mask))

    null = np.zeros(N_PERM, dtype=np.int64)

    for st in np.unique(strata):

        idx = strata == st

        N = int(np.sum(idx))
        K = int(np.sum(target[idx]))
        n = int(np.sum(selected_mask[idx]))

        if n == 0 or K == 0:
            continue

        if K == N:
            null += n
            continue

        null += rng.hypergeometric(
            ngood=K,
            nbad=N-K,
            nsample=n,
            size=N_PERM
        )

    null_mean = float(null.mean())
    null_sd = float(null.std(ddof=1))

    p_enrich = (
        1 + np.sum(null >= observed)
    ) / (N_PERM + 1)

    p_deplete = (
        1 + np.sum(null <= observed)
    ) / (N_PERM + 1)

    oe = observed / null_mean if null_mean > 0 else np.nan

    return observed, null_mean, oe, p_enrich, p_deplete, null_sd


def merge3(a, b):
    return a.merge(
        b,
        on=["chr","start","end"],
        how="inner"
    )


def analyse(context, stress, time, resolution,
            df, base24, basemeth, d24, dmeth):

    cols = [base24, basemeth, d24, dmeth]

    d = df.copy()

    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=cols).reset_index(drop=True)

    b24 = d[base24].to_numpy(float)
    bm = d[basemeth].to_numpy(float)
    x = d[d24].to_numpy(float)
    y = d[dmeth].to_numpy(float)

    strata = (
        rank_bins(b24, NBINS) * NBINS +
        rank_bins(bm, NBINS)
    )

    tests = [
        (f"24_gain_to_{context}_gain",
         x, "gain", y, "gain"),

        (f"24_loss_to_{context}_loss",
         x, "loss", y, "loss"),

        (f"{context}_gain_to_24_gain",
         y, "gain", x, "gain"),

        (f"{context}_loss_to_24_loss",
         y, "loss", x, "loss"),
    ]

    rows = []

    for frac in TAILS:

        pct = int(round(frac * 100))

        for test, selector, select_dir, counterpart, counterpart_dir in tests:

            selected = exact_tail(
                selector,
                frac,
                select_dir
            )

            observed, null_mean, oe, pe, pdp, nsd = \
                empirical_test(
                    selected,
                    counterpart,
                    counterpart_dir,
                    strata
                )

            rows.append({
                "context": context,
                "stress": stress,
                "time": time,
                "resolution": resolution,
                "pct": pct,
                "test": test,
                "n_eligible": len(d),
                "n_selected": len(selected),
                "observed": observed,
                "null_mean": null_mean,
                "OE": oe,
                "empirical_p": pe,
                "p_deplete": pdp,
                "null_sd": nsd
            })

    return rows


rows = []


for context in ["CG", "CHG"]:

    print("Processing", context, flush=True)

    # --------------------------------------------------------
    # DROUGHT
    # --------------------------------------------------------

    for res in [100, 500]:

        srna = pd.read_csv(
            f"drought_consensus/results/drought_24nt_{res}bp_matrix.tsv",
            sep="\t",
            usecols=[
                "chr","start","end",
                "CTRL_CPM","delta24_CPM"
            ]
        )

        meth = pd.read_csv(
            f"drought_consensus/results/drought_{context}_{res}bp_matrix.tsv",
            sep="\t"
        )

        d = merge3(srna, meth)

        rows += analyse(
            context,
            "drought",
            "single",
            res,
            d,
            "CTRL_CPM",
            "CTRLmean",
            "delta24_CPM",
            f"delta{context}"
        )


    # --------------------------------------------------------
    # HEAT
    # Reuse the 24-nt columns from the established CHH joint
    # table; these are independent of methylation context.
    # --------------------------------------------------------

    for res in [100, 500]:

        s = pd.read_csv(
            f"heat_consensus/results/heat_joint_{res}bp.tsv",
            sep="\t",
            usecols=[
                "chr","start","end",
                "CTRL24","delta24"
            ]
        )

        m = pd.read_csv(
            f"heat_consensus/results/heat_{context}_{res}bp_matrix.tsv",
            sep="\t"
        )

        d = merge3(s, m)

        for t in ["6h","12h","24h"]:

            rows += analyse(
                context,
                "heat",
                t,
                res,
                d,
                "CTRL24",
                "CTRLmean",
                "delta24",
                f"delta{context}_{t}"
            )


    # --------------------------------------------------------
    # PATHOGEN
    # Reuse established sRNA values from CHH joint table.
    # --------------------------------------------------------

    for res in [100, 500]:

        s = pd.read_csv(
            f"pathogen_consensus/results/pathogen_joint_{res}bp.tsv",
            sep="\t",
            usecols=[
                "chr","start","end",
                "CTRL24_6h","delta24_6h",
                "CTRL24_14h","delta24_14h"
            ]
        )

        m = pd.read_csv(
            f"pathogen_consensus/results/pathogen_{context}_{res}bp_matrix.tsv",
            sep="\t"
        )

        d = merge3(s, m)

        for t in ["6h","14h"]:

            rows += analyse(
                context,
                "pathogen",
                t,
                res,
                d,
                f"CTRL24_{t}",
                "MOCK",
                f"delta24_{t}",
                f"delta{context}"
            )


    # --------------------------------------------------------
    # PHOSPHATE
    # Reuse established 24-nt matrix from CHH table.
    # --------------------------------------------------------

    for res in [100, 500]:

        s = pd.read_csv(
            f"results/root_24nt_CHH_{res}bp.tsv",
            sep="\t",
            usecols=[
                "chr","start","end",
                "plus24_CPM","delta24_CPM"
            ]
        )

        m = pd.read_csv(
            f"results/root_methylation/root_{context}_{res}bp_matrix.tsv",
            sep="\t"
        )

        d = merge3(s, m)

        # methylation sanity filtering
        d = d[
            pd.to_numeric(
                d[f"plus_{context}"],
                errors="coerce"
            ).between(0,1) &
            pd.to_numeric(
                d[f"minus_{context}"],
                errors="coerce"
            ).between(0,1) &
            pd.to_numeric(
                d[f"delta{context}"],
                errors="coerce"
            ).between(-1,1)
        ].copy()

        rows += analyse(
            context,
            "phosphate",
            "single",
            res,
            d,
            "plus24_CPM",
            f"plus_{context}",
            "delta24_CPM",
            f"delta{context}"
        )


r = pd.DataFrame(rows)

# One multiplicity family for the new CG + CHG extension.
r["FDR_global_CG_CHG"] = bh(
    r["empirical_p"].to_numpy()
)

r["significant_global_CG_CHG"] = (
    r["FDR_global_CG_CHG"] < 0.05
)

r.to_csv(
    OUT,
    sep="\t",
    index=False
)

print()
print("===== CG/CHG EXACT-RANK COMPLETE =====")
print("tests =", len(r))
print("expected =", 336)
print()

print("Significant by context:")
print(
    r.groupby("context")[
        "significant_global_CG_CHG"
    ].sum()
)

print()
print("Significant by context/stress:")
print(
    r.groupby(
        ["context","stress"]
    )["significant_global_CG_CHG"].sum()
)

print()
print("Direction summary:")

z = (
    r.groupby(["context","stress","test"])
     ["significant_global_CG_CHG"]
     .sum()
)

print(z.to_string())

print()
print("Output:", OUT)
