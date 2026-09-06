#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import pandas as pd

ROOT = Path("/home/igor/arabidopsis_siRNA_methylation")
INFILE = ROOT / "results/nonTE_gene_analysis/heat_gene_body_window_opportunity_GO_10000perm_with_genes.tsv"
OUT = ROOT / "results/nonTE_gene_analysis"

d = pd.read_csv(INFILE, sep="\t")

# Keep only statistically validated enriched LOSS terms.
sig = d[
    (d["direction"] == "loss") &
    (d["FDR_BH"] < 0.05) &
    (d["fold_vs_window_null"] > 1)
].copy()

# ------------------------------------------------------------
# Define nonredundant functional modules
# ------------------------------------------------------------

photosynthesis_terms = {
    "GO:0015979",  # photosynthesis
    "GO:0009765",  # photosynthesis, light harvesting
    "GO:0019684",  # photosynthesis, light reaction
}

energy_terms = {
    "GO:0006091",  # generation precursor metabolites/energy
    "GO:0006096",  # glycolysis
    "GO:0006090",  # pyruvate metabolism
    "GO:0046034",  # ATP metabolism
    "GO:0046031",  # ADP metabolism
    "GO:0046032",  # ADP catabolism
    "GO:0009134",
    "GO:0009137",
    "GO:0009154",
    "GO:0009181",
    "GO:0009191",
    "GO:0009261",
    "GO:0006195",
    "GO:0019364",
    "GO:0072526",
    "GO:0009135",
    "GO:0009179",
    "GO:0009185",
    "GO:0009166",
    "GO:1901292",
    "GO:0009132",
    "GO:0072523",
    "GO:0046434",
}

def module_for(go):
    if go in photosynthesis_terms:
        return "Photosynthesis/light reaction"
    if go in energy_terms:
        return "Energy/carbon/nucleotide metabolism"
    return "Other"

sig["module"] = sig["GO_ID"].map(module_for)

# ------------------------------------------------------------
# Expand GO rows into gene rows
# ------------------------------------------------------------

rows = []

for _, r in sig.iterrows():

    genes = str(r["observed_genes"]).split(";")

    for gene in genes:
        gene = gene.strip()

        if not gene or gene == "nan":
            continue

        rows.append({
            "gene_id": gene,
            "context": r["context"],
            "module": r["module"],
            "GO_ID": r["GO_ID"],
            "GO_term": r["GO_term"],
            "GO_FDR": r["FDR_BH"],
            "GO_fold": r["fold_vs_window_null"]
        })

g = pd.DataFrame(rows).drop_duplicates()

# ------------------------------------------------------------
# Gene × context architecture
# ------------------------------------------------------------

gene_contexts = defaultdict(set)
gene_modules = defaultdict(set)
gene_terms = defaultdict(set)

for _, r in g.iterrows():
    gene_contexts[r["gene_id"]].add(r["context"])
    gene_modules[r["gene_id"]].add(r["module"])
    gene_terms[r["gene_id"]].add(r["GO_term"])

context_order = ["CG", "CHG", "CHH"]

gene_rows = []

for gene in sorted(gene_contexts):

    ctx = [
        x for x in context_order
        if x in gene_contexts[gene]
    ]

    gene_rows.append({
        "gene_id": gene,
        "n_contexts": len(ctx),
        "contexts": "+".join(ctx),
        "CG": int("CG" in ctx),
        "CHG": int("CHG" in ctx),
        "CHH": int("CHH" in ctx),
        "modules": "; ".join(sorted(gene_modules[gene])),
        "GO_terms": "; ".join(sorted(gene_terms[gene]))
    })

arch = pd.DataFrame(gene_rows)

arch = arch.sort_values(
    ["n_contexts", "contexts", "gene_id"],
    ascending=[False, True, True]
)

archfile = OUT / "heat_validated_gene_context_architecture.tsv"
arch.to_csv(archfile, sep="\t", index=False)

# ------------------------------------------------------------
# Context-sharing summary
# ------------------------------------------------------------

sharing = (
    arch.groupby(["contexts", "n_contexts"])
        .size()
        .reset_index(name="n_genes")
        .sort_values(
            ["n_contexts", "contexts"],
            ascending=[False, True]
        )
)

sharingfile = OUT / "heat_validated_context_sharing_summary.tsv"
sharing.to_csv(sharingfile, sep="\t", index=False)

# ------------------------------------------------------------
# Module × context summary
# ------------------------------------------------------------

module_rows = []

for module in sorted(g["module"].unique()):

    if module == "Other":
        continue

    gm = g[g["module"] == module]

    all_module_genes = set(gm["gene_id"])

    for context in context_order:

        genes = sorted(
            set(
                gm.loc[
                    gm["context"] == context,
                    "gene_id"
                ]
            )
        )

        module_rows.append({
            "module": module,
            "context": context,
            "n_genes": len(genes),
            "genes": ";".join(genes)
        })

    # Across-context union
    module_rows.append({
        "module": module,
        "context": "ANY",
        "n_genes": len(all_module_genes),
        "genes": ";".join(sorted(all_module_genes))
    })

modules = pd.DataFrame(module_rows)

modulefile = OUT / "heat_validated_nonredundant_modules.tsv"
modules.to_csv(modulefile, sep="\t", index=False)

# ------------------------------------------------------------
# Cross-context recurrence within each module
# ------------------------------------------------------------

rec_rows = []

for module in sorted(g["module"].unique()):

    if module == "Other":
        continue

    gm = g[g["module"] == module]

    for gene in sorted(set(gm["gene_id"])):

        ctx = sorted(
            set(
                gm.loc[
                    gm["gene_id"] == gene,
                    "context"
                ]
            ),
            key=context_order.index
        )

        rec_rows.append({
            "module": module,
            "gene_id": gene,
            "n_contexts": len(ctx),
            "contexts": "+".join(ctx)
        })

rec = pd.DataFrame(rec_rows).sort_values(
    ["module", "n_contexts", "gene_id"],
    ascending=[True, False, True]
)

recfile = OUT / "heat_validated_module_gene_recurrence.tsv"
rec.to_csv(recfile, sep="\t", index=False)

# ------------------------------------------------------------
# Console report
# ------------------------------------------------------------

print("\n========================================")
print("VALIDATED HEAT LOSS: CONTEXT SHARING")
print("========================================")
print(sharing.to_string(index=False))

print("\n========================================")
print("GENES PRESENT IN >=2 CONTEXTS")
print("========================================")

multi = arch[arch["n_contexts"] >= 2]

if len(multi):
    print(
        multi[
            ["gene_id", "contexts", "modules"]
        ].to_string(index=False)
    )
else:
    print("None")

print("\n========================================")
print("NONREDUNDANT FUNCTIONAL MODULES")
print("========================================")

for module in modules["module"].unique():

    print("\n" + module)

    q = modules[
        (modules["module"] == module) &
        (modules["context"] != "ANY")
    ]

    for _, r in q.iterrows():
        print(
            f'  {r["context"]}: '
            f'{r["n_genes"]} genes'
        )
        print("    " + r["genes"])

print("\n========================================")
print("MODULE GENES SHARED ACROSS CONTEXTS")
print("========================================")

for module in rec["module"].unique():

    print("\n" + module)

    q = rec[
        (rec["module"] == module) &
        (rec["n_contexts"] >= 2)
    ]

    if len(q):
        print(
            q[
                ["gene_id", "contexts"]
            ].to_string(index=False)
        )
    else:
        print("  None")

print("\nSaved:")
print(" ", archfile)
print(" ", sharingfile)
print(" ", modulefile)
print(" ", recfile)
print("\nDONE")
