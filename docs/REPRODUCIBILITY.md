# Reproducibility notes

## Core design
1. Quantify stress-associated changes in 24-nt small-RNA abundance and DNA methylation in common 100-bp and 500-bp genomic windows.
2. Analyze CG, CHG, and CHH methylation separately.
3. Remove TE-overlapping windows; use >=1 kb from the nearest annotated TE for the primary TE-distant analysis.
4. Define concordant gain and loss from same-direction changes in the two independently measured molecular layers.
5. Evaluate 1%, 5%, and 10% joint tails.
6. Control gene-length/measurable-window opportunity with genomic-window permutation.
7. Evaluate cross-context gene and exact-window recurrence.
8. Validate selected heat functional systems with a threshold-free window-structure-preserving permutation.
9. Rank TE-distant gene-body loci descriptively after collapsing nested-tail duplication.

## Statistical parameters reported in the manuscript
- Definitive heat 1% gene-body opportunity analysis: 10,000 permutations.
- Threshold-free targeted heat validation: 1,000 permutations.
- Empirical P values in the continuous test: (k + 1)/(N + 1).
- Multiple-testing correction: Benjamini-Hochberg where specified.

## Release policy
The GitHub release should contain exact scripts and compact processed outputs, not public raw FASTQ/WGBS source files. Record software versions and random seeds from the working environment before tagging the release.
