# TE-distant 24-nt siRNA–DNA methylation correspondence in Arabidopsis

Reproducibility package for the manuscript:

**TE-distant gene networks show stress- and context-dependent 24-nt siRNA–DNA methylation correspondence in Arabidopsis thaliana**

This repository is intended to contain the exact analysis scripts, metadata, and processed result tables used for the manuscript. Raw public sequencing files are not redistributed; the source datasets are identified by accession in `metadata/datasets.tsv`.

## Analysis scope

The study integrates independent public small-RNA and whole-genome bisulfite-sequencing datasets for heat, drought, phosphate deficiency, and bacterial pathogen challenge. Analyses use 100-bp and 500-bp genomic windows, CG/CHG/CHH methylation contexts, TE exclusion (primary analysis: >=1 kb from the nearest annotated TE), joint-tail analyses, genomic-window opportunity permutations, cross-context recurrence, threshold sensitivity, threshold-free continuous permutation analysis, and descriptive gene-level ranking.

Because small-RNA and methylation measurements came from independent experiments, the analyses test cross-study spatial correspondence and do not establish within-sample molecular coupling, temporal ordering, or causality.

## Repository layout

- `metadata/datasets.tsv` — public source datasets and accessions
- `scripts/` — exact analysis and figure-generation scripts copied from the working project
- `results/` — compact processed tables required to reproduce manuscript values and figures
- `supplement/Supplementary_Table_S2.xlsx` — complete ranked and annotated gene-level results
- `docs/REPRODUCIBILITY.md` — reproducibility and release notes
- `collect_release_files.sh` — helper for assembling exact scripts/results from the working WSL project

## Important release rule

Do not replace the original analysis scripts with rewritten or reconstructed versions. The public repository should contain the exact scripts used for the reported analyses. Run `collect_release_files.sh` from the project root (`~/arabidopsis_siRNA_methylation`) to stage available manuscript-related scripts and compact outputs, then review the staged files before publication.

## Data availability

All source sequencing datasets are public through NCBI GEO. Large public source files should be retrieved from the original archives rather than committed to GitHub. A tagged GitHub release should be archived in Zenodo for a permanent DOI.

## License

No software license has been selected in this staging package. Choose the repository license before public release.
