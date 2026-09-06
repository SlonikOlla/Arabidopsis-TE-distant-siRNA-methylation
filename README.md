# TE-distant gene networks show stress- and context-dependent 24-nt siRNA–DNA methylation correspondence in Arabidopsis thaliana

Reproducibility repository for the expanded Arabidopsis analysis of stress-associated spatial correspondence between 24-nucleotide small interfering RNAs (24-nt siRNAs) and DNA methylation at TE-distant gene-associated regions.

## Scope

The study integrates independent public small-RNA and whole-genome bisulfite-sequencing datasets for heat, drought, phosphate deficiency, and bacterial pathogen challenge. Analyses use 100-bp and 500-bp windows and examine CG, CHG, and CHH methylation contexts.

The primary analysis excludes TE-overlapping windows and focuses on gene-associated windows located at least 1 kb from the nearest annotated transposable element. Because small-RNA and methylation measurements originate from independent public experiments, the study tests cross-study spatial correspondence rather than paired within-sample molecular coupling, temporal ordering, or causality.

## Public datasets

| Stress | Small-RNA dataset | Methylation dataset |
|---|---|---|
| Heat | GSE239836 | GSE139941 |
| Drought | GSE26356 | GSE94075 |
| Phosphate deficiency | GSE17741 | GSE72770 |
| Pathogen challenge | GSE19694 | GSE128768 |

No raw public sequencing files are redistributed here.

## Analysis framework

1. 100-bp and 500-bp genomic windows.
2. CG, CHG, and CHH methylation analyzed separately.
3. TE-overlap exclusion; primary stringent analysis >=1 kb from the nearest annotated TE.
4. Concordant gain and concordant loss analyzed separately.
5. Joint-tail discovery at 1%, with 5% and 10% sensitivity analyses.
6. Gene-level functional analysis with genomic-window opportunity permutation.
7. Cross-context recurrence at gene and exact physical-window levels.
8. Threshold-free, window-structure-preserving permutation analysis for targeted heat-response systems.
9. Descriptive gene-level ranking, functional annotation, and cross-stress recurrence.

## Repository organization

- `metadata/` — source dataset accessions and experimental-design notes.
- `scripts/` — exact analysis and figure-generation scripts used for the reported analyses.
- `results/` — compact processed result tables supporting reported statistics and figures.
- `supplement/` — supplementary data files.
- `docs/` — reproducibility and provenance documentation.

**The manuscript itself is intentionally not included in this repository.**

## Reproducibility policy

The release will contain exact retained analysis scripts where available. Reconstructed or approximate code will not be presented as historical executed code. Large public FASTQ, BAM, and per-cytosine methylation files are excluded and should be retrieved from their original repositories.

## Citation

Please cite the associated manuscript and the versioned Zenodo archive corresponding to the release used.

**Kovalchuk I. TE-distant gene networks show stress- and context-dependent 24-nt siRNA–DNA methylation correspondence in Arabidopsis thaliana.**

A Zenodo DOI will be added after the first release is archived.
