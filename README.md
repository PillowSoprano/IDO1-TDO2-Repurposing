# IDO1/TDO2 dual-target repurposing — reproducibility package

Provenance-adaptive conformal ranking framework for honest dual-target drug repurposing, applied
to IDO1 and TDO2. This repository holds the manuscript, all data artifacts, trained models,
figures, a **reconstructed and self-validating analysis pipeline** (`src/`), and three
**extension analyses** (`extensions/`).

- **Manuscript:** `IDO1_TDO2_PACR_JCheminf_submission_ready.docx` (current Journal of Cheminformatics submission draft) + [`preprint.md`](preprint.md) (Markdown source for the longer preprint draft).
- **Headline result:** an honest, uncertainty-calibrated *negative* — no approved drug clears a
  confident dual-active conformal bound — plus a model-free measured IDO1/TDO2 inhibitor
  correlation *r* = 0.43.
- **Superseded / erroneous files** are quarantined in [`archive/`](archive/README.md) (e.g. the
  early PDB 6IC2 docking, which is carbonic anhydrase II, not IDO1; regenerated against 6E40).

## Quick start

```bash
pip install -r requirements.txt        # Python 3.11
bash src/run_all.sh                     # rebuild + validate every stage from raw ChEMBL data
```

Each stage prints a `VALIDATE vs <reference>` block comparing its freshly rebuilt output against
the committed artifact, and writes `data/*_rebuilt.csv` (references are never overwritten).

## Pipeline (`src/`) — raw ChEMBL → results, each stage self-validating

| Stage | Script | Input → Output | Validates against | Fidelity |
|------|--------|----------------|-------------------|----------|
| 1 Curate | `s1_curate.py` | `*_bioactivity_raw.csv` → curated potency | `ido1_curated.csv` | **exact** (3585/964 cpds, PAINS 362, median pActivity 6.5229) |
| 2 Features | `s2_features.py` | curated → 9 descriptors + Murcko scaffold | `*_features.csv` | descriptors Δ≈1e-13; scaffold strings 81–96% (RDKit-version canonicalisation) |
| 3 Models | `s3_models.py` | features → RF clf+reg, random/scaffold CV | `model_performance.csv` | AUROC Δ≤0.001, R² Δ≤0.015 |
| 4 Screen | `s4_screen.py` | full models → 2,387-drug deck scores + AD | `dual_scored.csv` | AD corr **1.000**, pred corr 0.999, prob corr 0.95–0.98 |
| 5 Paired | `s5_paired_conformal.py` | 684 paired → split-hardness ladder, LODO, conformal | `split_hardness.csv`, `lodo_results.csv`, `conformal_calibration.csv` | Butina clusters **171** exact; LODO Δ≤0.003; ladder Δ≤0.06 (chemspace stochastic); coverage near-nominal |
| 6 MMP | `s6_mmp.py` | 684 paired → matched-pair transformations | `mmp_transformations.csv` | 1882 vs 1780 (≥2 pairs), 430 vs 376 (≥3) — minor fragmentation-convention gap |

`chem.py` holds shared cheminformatics utilities (curation, Morgan fingerprints, descriptors,
scaffolds, PAINS, Tanimoto). Fingerprint regeneration reproduces the deployed models' stored
predictions to max |Δ| = 0.005.

### Reproducibility notes
- **Deterministic** given RDKit + fixed seeds, except: (a) the tSNE columns in `*_features.csv`
  and (b) the UMAP chemical-space split in stage 5, both stochastic/version-dependent (stage 5
  falls back to PCA if `umap-learn` is absent). These affect visualisation / one ladder rung only.
- **Not cached in-repo:** the upstream ChEMBL fetches — bioactivity (stages 1–6) is provided as
  `data/*_bioactivity_raw.csv`; the approved-drug deck SMILES are read from `data/dual_scored.csv`
  (the `max_phase=4` fetch itself is live-API and dated in the Methods). Stage 4 reproduces the
  *scoring*, not that fetch.
- **scikit-learn:** committed `.joblib` models were pickled under 1.9.0; this env pins 1.8.0.
  Predictions reproduce exactly, so the version warning is benign — pin 1.9.0 for byte-identical
  model archival.

## Extensions (`extensions/`) — three analyses beyond the preprint
See [`extensions/EXTENSIONS_RESULTS.md`](extensions/EXTENSIONS_RESULTS.md) for full results.
1. **PACR conformal ranking** — provenance-adaptive intervals add source-document risk to
   RF-normalized conformal ranking; sensitivity/ablation confirms the honest negative survives.
2. **Representation robustness** — the LODO collapse holds across ECFP4/RDKit-desc/MACCS/ChemBERTa
   × RF/XGB/Ridge (max LODO R² 0.11): it is the data, not the model.
3. **COX-1/COX-2 second pair** — paired-target validation workflow replication (measured *r* = 0.28; same scaffold→LODO decay).

## Layout
```
preprint.md, *.docx        manuscript source and current DOCX draft
data/                      raw ChEMBL, curated, features, paired set, screen outputs, *_rebuilt.csv
models/                    trained RF + conformal models (.joblib)
src/                       reconstructed self-validating pipeline (this package) + run_all.sh
extensions/                three extension analyses + cache
figures/                   all manuscript + extension figures
complexes/, 6E40.pdb       docking receptor (IDO1, PDB 6E40) and docked complexes
archive/                   superseded / erroneous 6IC2-era files (do not cite)
```
