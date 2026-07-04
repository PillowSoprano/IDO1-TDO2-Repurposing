# Extension analyses — results summary

Three computational extensions to the IDO1/TDO2 preprint, each self-contained, reproducible
from `extensions/`, and producing a manuscript-ready figure + data. All use public ChEMBL
data and the in-folder trained models (reproduction of stored `pred_M`/`pred_D` verified to
max |Δ| = 0.005). Run order: `common.py` (shared helpers) → `ext1` → `ext1_pacr_sensitivity`
→ `ext2` → `ext3`.

Environment note: models were pickled under scikit-learn 1.9.0; this run used 1.8.0. Predictions
reproduce exactly, so the version warning is benign, but pin sklearn for archival reproducibility.

---

## Extension 1 — Provenance-Adaptive Conformal Ranking (PACR)
**Script:** `ext1_adaptive_conformal.py` · **Figure:** `figures/figure_adaptive_conformal.png`
· **Data:** `ext1_conformal_comparison.csv`, `ext1_deck_normalized.csv`,
`ext1_deck_pacr.csv`, `ext1_pacr_scale_weights.csv`

**Sensitivity script:** `ext1_pacr_sensitivity.py` · **Figure:** `figures/figure_pacr_sensitivity.png`
· **Data:** `ext1_pacr_sensitivity.csv`, `ext1_pacr_sensitivity_summary.csv`

**Question.** The paper's split-conformal intervals have essentially *constant* width (~1.4 log
units), so coverage is marginal, not conditional — a stated Limitation. Can the source-document
shift revealed by LODO be used directly in the conformal ranking score, rather than treated only
as a diagnostic?

**Method.** Same scaffold-disjoint 3-way split (proper-train 185 / calibration 205 / test 294)
and same point-estimate M/D random forests for all methods. Standard = paper's paired-residual
intervals (fixed width). RF-normalized = nonconformity scaled by RF tree-to-tree variance. PACR =
RF-normalized conformal ranking plus a learned provenance multiplier:
`base_sigma(x) = tree_std(x) + median(tree_std_cal)`, multiplied by
`exp(theta0) * (1 + theta_chem * d_chem(x) + theta_doc * d_doc(x))`, where `d_chem` is
1 - max Tanimoto to proper-train compounds and `d_doc` is 1 - nearest source-document chemical
domain support. Document support uses `k = 5`; documents with fewer than five proper-train
compounds use all available compounds, and support is computed only from proper-train documents.
The PACR multiplier is learned only on proper-train leave-document-out residuals
(pinball loss, tau = 0.75), then held fixed; the final calibration fold still supplies the paired
M/D residual distribution. PACR is therefore a conservative source-shift ranking rule, not a new
QSAR model.

**Results.**
- Marginal 90% coverage: standard IDO1/TDO2 0.87/0.90; RF-normalized 0.89/0.89; **PACR 0.92/0.92**.
  PACR is intentionally conservative rather than tuned to sit exactly on 0.90.
- Standard interval width is **flat at 2.48 log units across every AD quartile**; it under-covers
  far compounds (Q4-far 0.84). RF-normalized width grows with AD distance (2.23 → 3.05), while
  PACR grows more strongly (2.37 → 3.58).
- By source-document domain distance, PACR raises far-quartile IDO1 coverage to **0.97** versus
  0.84 standard and 0.93 RF-normalized. Width-error correlation improves from −0.06 standard to
  +0.19 RF-normalized and **+0.22 PACR**.
- Learned scale weights show the provenance signal lands mainly on the selectivity axis: for *D*,
  `theta_doc = 0.254` and `theta_chem = 0.034`; for *M*, only a small global multiplier is selected.
- **Honest negative strengthens:** 0 / 2,387 drugs clear the confident dual bound (pActivity ≥ 6) under
  all methods. Max LCB_dual: standard 5.52, RF-normalized 5.44, **PACR 5.29**.
- **Sensitivity/ablation across 5 scaffold split seeds:** standard mean IDO1 coverage 0.887 and
  far source-domain coverage 0.778; RF-normalized 0.883 / 0.843; PACR chem+doc at tau = 0.75
  0.909 / 0.918. PACR ablations were similar (chem 0.910 / 0.918; doc 0.909 / 0.912), suggesting
  the gain comes from the conservative source-risk scale rather than one fragile distance term.
  The tau sweep behaves as an interpretable conservatism knob: PACR chem+doc coverage rises from
  0.890 / 0.866 at tau = 0.65 to 0.925 / 0.926 at tau = 0.80, while max deck LCB_dual falls from
  5.36 to 5.12. Across every seed, tau, and ablation, **0 drugs clear pActivity 6**.

**Takeaway.** Upgrades a Limitation into an algorithmic contribution: once document shift is shown
to be the relevant failure mode, PACR uses provenance to make conformal ranking source-shift-aware.
The result is not a claim that PACR discovers a repurposing hit; it makes the negative more robust
by penalising compounds unsupported by the training documents' chemical domains.
**Manuscript home:** §2.7 now includes Algorithm 1 (PACR); §3.12 reports the main result;
Figure S2/Table S1 carry the sensitivity and ablation details; the "constant-width" limitation
is reframed as a solved diagnostic rather than an unresolved weakness.

---

## Extension 2 — Representation robustness of the LODO collapse
**Script:** `ext2_representation_lodo.py` · **Figure:** `figures/figure_representation_lodo.png`
· **Data:** `ext2_representation_lodo.csv`

**Question.** The paper's central diagnostic — M skill collapses from R²≈0.64 (random) to ≈0
(leave-one-document-out) — invites the objection that this is a weakness of RF + Morgan, not of
the data. Would a richer representation or learner generalise across source documents?

**Method.** Same LODO protocol (LeaveOneGroupOut over the 44 ChEMBL documents) across a grid of
**4 representations** — ECFP4 (2048), RDKit 2D descriptors (217), MACCS (167), **pretrained
ChemBERTa embeddings (768)** — × **3 learners** (RandomForest, XGBoost, Ridge), each also under
random 5-fold and scaffold GroupKFold. Pooled global R² for shared potency M.

**Results (pooled global R² for M).**

| representation + learner | random | scaffold | LODO |
|---|---|---|---|
| ECFP4 + RF (paper baseline) | 0.638 | 0.551 | **0.032** |
| ECFP4 + XGB | 0.665 | 0.572 | 0.015 |
| ECFP4 + Ridge | 0.485 | 0.298 | −0.451 |
| RDKit-desc + RF | 0.640 | 0.475 | 0.103 |
| RDKit-desc + XGB | 0.652 | 0.506 | 0.024 |
| MACCS + RF | 0.609 | 0.465 | 0.112 |
| ChemBERTa + Ridge | 0.504 | 0.286 | −0.351 |
| ChemBERTa + RF | 0.506 | 0.364 | −0.031 |

- Mean R²: random 0.587 → scaffold 0.440 → **LODO −0.068**.
- **Max LODO R² across all 8 methods = 0.112.** The pretrained foundation-model embedding is among
  the *worst* under LODO, not a rescue. Baseline ECFP4+RF reproduces the paper's 0.03 exactly.

**Takeaway.** The cross-document collapse is a property of the **data** (cross-laboratory
distribution shift), not of any one model — pre-empts the primary reviewer objection.
**Manuscript home:** strengthen §3.9 / Fig. 7; add one panel.

---

## Extension 3 — Second target-pair validation (COX-1 / COX-2)
**Script:** `ext3_second_pair.py` · **Figure:** `figures/figure_second_pair_cox.png`
· **Data:** `ext3_cox_paired.csv`, `ext3_cox_splithardness.csv`, cache in `cache_cox/`

**Question.** The Discussion claims the paired-target validation workflow generalises beyond
IDO1/TDO2 but shows no second example. Does this workflow reproduce on an independent, canonical
selectivity pair?

**Method.** Fetched COX-1 (CHEMBL221) and COX-2 (CHEMBL230) bioactivity live from the ChEMBL REST
API (5,370 + 9,412 records), curated identically (→ 1,666 + 4,400 compounds), formed the co-tested
set, and ran the measured correlation → M/D reparametrisation → split-hardness ladder + LODO →
split-conformal coverage.

**Results.**
- **1,379 co-tested compounds** (214 documents, 427 scaffolds) — twice the IDO1/TDO2 set.
- **Measured correlation Pearson r = 0.28 (95% CI 0.23–0.33, p = 4×10⁻²⁶)** — a moderate positive
  correlation in the *same regime* as IDO1/TDO2's r = 0.43. Balanced-dual fraction (|D|≤1) = 0.44.
- **Same monotonic skill decay reappears:** M random 0.51 → scaffold 0.28 → LODO 0.16; D random
  0.60 → scaffold 0.42 → LODO 0.34. Scaffold splitting is again optimistic relative to LODO.
- Conformal 90% empirical coverage: M 0.88, D 0.81 — close to nominal for M but lower for D,
  indicating that selectivity imbalance remains harder to calibrate.

**Honest nuance.** COX LODO does **not** collapse all the way to ~0 (unlike IDO1/TDO2): with far
more co-tested compounds across many more documents, some cross-document signal survives. So the
*qualitative* pattern (monotonic random→scaffold→LODO degradation, scaffold over-optimism)
transfers robustly; the *magnitude* of collapse is dataset-dependent — a fair, defensible framing.

**Takeaway.** Gives the paired-target validation workflow real empirical backing on independent
data, without claiming a full COX-PACR comparison. **Manuscript home:** §3.14 / Fig. 13.

---

## Suggested next step
Fold these into `preprint.md` (priority 3, manuscript) as: one new Results subsection (Ext 1),
one strengthened subsection + panel (Ext 2), and one Discussion paragraph + supp figure (Ext 3);
convert two Limitations bullets (constant-width intervals; single target pair) into resolved
results. The reproducibility package (priority 2) is already half-built — `extensions/` +
`requirements.txt` would complete it.
