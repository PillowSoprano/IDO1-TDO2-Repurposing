# Journal of Cheminformatics Supplementary Reproducibility Package

This supplementary package supports the manuscript:

**Provenance-adaptive conformal ranking for honest dual-target drug repurposing from public bioactivity data**

The package contains the analysis artifacts needed to reproduce the paired-target conformal ranking workflow without bundling large binary model files.

## Included contents

- `data/paired_MD_dataset.csv`
- `data/split_hardness.csv`
- `data/lodo_results.csv`
- `data/conformal_calibration.csv`
- `data/dual_conformal_ranked.csv`
- `data/mmp_transformations.csv`
- `extensions/`
- `requirements.txt`
- `README.md`
- `src/`

The self-validating pipeline entry point is:

```bash
bash src/run_all.sh
```

## Model artifacts

The trained `.joblib` model files are not included in this supplementary zip because they are approximately 285 MB total, exceeding normal supplementary-file size limits. They are tracked in the version-controlled GitHub repository via Git LFS:

https://github.com/PillowSoprano/IDO1-TDO2-Repurposing

See `MODEL_ARTIFACTS.md` for file names, sizes and SHA-256 checksums.

Before journal submission or peer review, the repository should be made publicly accessible or archived through Zenodo/GitHub Release with a citable DOI.
