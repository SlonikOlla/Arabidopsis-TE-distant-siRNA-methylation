#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 20260829
N_PERM = 1000
NBINS = 10
TAILS = [0.01, 0.05, 0.10]

FINAL = "results/all_stresses_extreme_tail_EXACT_RANK_FINAL.tsv"
OUT = "results/all_stresses_extreme_tail_EXACT_RANK_RECONSTRUCTED.tsv"
CHECK = "results/exact_rank_CHH_reconstruction_validation.tsv"

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
    """
    Divide observations into exact rank-based bins.

    Stable sorting makes tie handling deterministic.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)

    z = np.floor(ranks * nbins / n).astype(int)
    z[z == nbins] = nbins - 1
    return z


def exact_tail(x, frac, direction):
    """
    Select exactly floor(n*frac) observations by rank.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    k = int(np.floor(n * frac))

    if k < 1:
        return np.array([], dtype=int)

    order = np.argsort(x, kind="mergesort")

    if direction == "loss":
        return order[:k]

    if direction == "gain":
        return order[-k:]

    raise ValueError(direction)


def empirical_test(selected, counterpart, counterpart_direction, strata):
    """
    Exact stratified permutation null, sampled efficiently.

    Within each baseline stratum:
      N = total eligible windows
      K = windows having requested counterpart sign
      n = selected windows

    Under within-stratum permutation, overlap follows
    Hypergeometric(N, K, n).

    Summing independent stratum-specific hypergeometrics is
    equivalent to shuffling counterpart labels within strata.
    """
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


def clean(df, cols):
    d = df.copy()

    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=cols).copy()

    return d.reset_index(drop=True)


def analyse(stress, time, resolution,
            df, base24, basechh, d24, dchh):

    d = clean(df, [base24, basechh, d24, dchh])

    b24 = d[base24].to_numpy(float)
    bchh = d[basechh].to_numpy(float)
    x = d[d24].to_numpy(float)
    y = d[dchh].to_numpy(float)

    strata = (
        rank_bins(b24, NBINS) * NBINS +
        rank_bins(bchh, NBINS)
    )

    rows = []

    tests = [
        ("24_gain_to_CHH_gain", x, "gain", y, "gain"),
        ("24_loss_to_CHH_loss", x, "loss", y, "loss"),
        ("CHH_gain_to_24_gain", y, "gain", x, "gain"),
        ("CHH_loss_to_24_loss", y, "loss", x, "loss"),
    ]

    for frac in TAILS:
        pct = int(round(frac * 100))

        for test, selector, select_dir, counterpart, counterpart_dir in tests:

            selected = exact_tail(selector, frac, select_dir)

            observed, null_mean, oe, p_enrich, p_deplete, null_sd = \
                empirical_test(
                    selected,
                    counterpart,
                    counterpart_dir,
                    strata
                )

            rows.append({
                "stress": stress,
                "time": time,
                "resolution": resolution,
                "pct": pct,
                "test": test,
                "n_selected": len(selected),
                "observed": observed,
                "null_mean": null_mean,
                "OE": oe,
                "empirical_p": p_enrich,
                "p_deplete": p_deplete,
                "null_sd": null_sd,
            })

    return rows


rows = []


# ------------------------------------------------------------
# DROUGHT
# ------------------------------------------------------------

for res in [100, 500]:

    f = f"drought_consensus/results/drought_joint_{res}bp.tsv"
    d = pd.read_csv(f, sep="\t")

    rows += analyse(
        "drought", "single", res, d,
        "CTRL_CPM",
        "CTRLmean_CHH",
        "delta24_CPM",
        "deltaCHH"
    )


# ------------------------------------------------------------
# HEAT
# One 1-h sRNA contrast paired separately with 6/12/24-h CHH.
# ------------------------------------------------------------

for res in [100, 500]:

    f = f"heat_consensus/results/heat_joint_{res}bp.tsv"
    d = pd.read_csv(f, sep="\t")

    for t in ["6h", "12h", "24h"]:

        rows += analyse(
            "heat", t, res, d,
            "CTRL24",
            "CTRL_CHH",
            "delta24",
            f"deltaCHH_{t}"
        )


# ------------------------------------------------------------
# PATHOGEN
# Same methylation contrast paired with 6-h and 14-h sRNA.
# ------------------------------------------------------------

for res in [100, 500]:

    f = f"pathogen_consensus/results/pathogen_joint_{res}bp.tsv"
    d = pd.read_csv(f, sep="\t")

    for t in ["6h", "14h"]:

        rows += analyse(
            "pathogen", t, res, d,
            f"CTRL24_{t}",
            "CTRL_CHH",
            f"delta24_{t}",
            "delta_CHH"
        )


# ------------------------------------------------------------
# PHOSPHATE
# Sentinel-safe filtering.
# Negative sentinel CHH values are invalid/missing.
# ------------------------------------------------------------

for res in [100, 500]:

    f = f"results/root_24nt_CHH_{res}bp.tsv"
    d = pd.read_csv(f, sep="\t")

    # Retain biologically valid methylation values only.
    d["plus_CHH"] = pd.to_numeric(d["plus_CHH"], errors="coerce")
    d["minus_CHH"] = pd.to_numeric(d["minus_CHH"], errors="coerce")
    d["deltaCHH"] = pd.to_numeric(d["deltaCHH"], errors="coerce")

    d = d[
        d["plus_CHH"].between(0, 1, inclusive="both") &
        d["minus_CHH"].between(0, 1, inclusive="both") &
        d["deltaCHH"].between(-1, 1, inclusive="both")
    ].copy()

    rows += analyse(
        "phosphate", "single", res, d,
        "plus24_CPM",
        "plus_CHH",
        "delta24_CPM",
        "deltaCHH"
    )


r = pd.DataFrame(rows)

r["FDR_global"] = bh(r["empirical_p"].to_numpy())
r["significant_global"] = r["FDR_global"] < 0.05

cols = [
    "stress", "time", "resolution", "pct", "test",
    "n_selected", "observed", "null_mean", "OE",
    "empirical_p", "p_deplete", "null_sd",
    "FDR_global", "significant_global"
]

r = r[cols]

r.to_csv(OUT, sep="\t", index=False)


# ------------------------------------------------------------
# VALIDATION AGAINST RETAINED FINAL CHH TABLE
# ------------------------------------------------------------

ref = pd.read_csv(FINAL, sep="\t")

keys = ["stress", "time", "resolution", "pct", "test"]

v = ref.merge(
    r,
    on=keys,
    how="outer",
    suffixes=("_reference", "_reconstructed"),
    indicator=True
)

for c in ["n_selected", "observed"]:
    v[f"{c}_match"] = (
        v[f"{c}_reference"] ==
        v[f"{c}_reconstructed"]
    )

for c in [
    "null_mean", "OE", "empirical_p",
    "p_deplete", "null_sd", "FDR_global"
]:
    v[f"{c}_absdiff"] = np.abs(
        v[f"{c}_reference"] -
        v[f"{c}_reconstructed"]
    )

v.to_csv(CHECK, sep="\t", index=False)


print()
print("===== CHH EXACT-RANK RECONSTRUCTION =====")
print("generated tests =", len(r))
print("reference tests =", len(ref))
print("merged rows =", len(v))
print()

print(
    "n_selected exact matches =",
    int(v["n_selected_match"].sum()),
    "/",
    len(v)
)

print(
    "observed exact matches =",
    int(v["observed_match"].sum()),
    "/",
    len(v)
)

print()

for c in [
    "null_mean", "OE", "empirical_p",
    "p_deplete", "null_sd", "FDR_global"
]:
    print(
        "max abs difference", c, "=",
        v[f"{c}_absdiff"].max()
    )

print()

print(
    "reference significant =",
    int(ref["significant_global"].sum())
)

print(
    "reconstructed significant =",
    int(r["significant_global"].sum())
)

sig_compare = ref[keys + ["significant_global"]].merge(
    r[keys + ["significant_global"]],
    on=keys,
    suffixes=("_reference", "_reconstructed")
)

print(
    "significance calls identical =",
    int(
        (
            sig_compare["significant_global_reference"] ==
            sig_compare["significant_global_reconstructed"]
        ).sum()
    ),
    "/",
    len(sig_compare)
)

print()
print("Wrote:")
print(OUT)
print(CHECK)

print()
print("First discrepancies in deterministic statistics:")

bad = v[
    (~v["n_selected_match"]) |
    (~v["observed_match"])
]

if len(bad) == 0:
    print("NONE")
else:
    print(
        bad[
            keys +
            [
                "n_selected_reference",
                "n_selected_reconstructed",
                "observed_reference",
                "observed_reconstructed"
            ]
        ].head(30).to_string(index=False)
    )
