#!/usr/bin/env bash
# Reproduce the IDO1/TDO2 pipeline end-to-end from the cached raw ChEMBL data.
# Each stage self-validates against the committed reference outputs and writes *_rebuilt.csv.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Stage 1: curation ============================================"; python3 s1_curate.py
echo "=== Stage 2: features ============================================"; python3 s2_features.py
echo "=== Stage 3: models + leakage-safe CV ============================"; python3 s3_models.py
echo "=== Stage 4: repurposing screen =================================="; python3 s4_screen.py
echo "=== Stage 5: paired M/D, split-hardness, LODO, conformal ========="; python3 s5_paired_conformal.py
echo "=== Stage 6: matched molecular pairs ============================="; python3 s6_mmp.py
echo "=== done. Rebuilt artifacts written as data/*_rebuilt.csv ========"
