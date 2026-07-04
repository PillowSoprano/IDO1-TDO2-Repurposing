# Leakage-safe computational repurposing of approved drugs against IDO1, a melanoma immunotherapy target

**Working draft — Xiyao Yu**

## Abstract

Indoleamine 2,3-dioxygenase 1 (IDO1) is a tryptophan-catabolising immune-checkpoint enzyme over-expressed in melanoma, but the lead clinical inhibitor epacadostat failed its pivotal phase III melanoma trial, leaving combination and repurposing strategies as open questions. We mined 7,634 IDO1 bioactivity records from ChEMBL, curated a 3,585-compound structure–activity dataset (1,361 unique Bemis–Murcko scaffolds), and trained random-forest models on 2,048-bit Morgan fingerprints. Crucially, we evaluated under **scaffold-based** cross-validation rather than random splits, which quantifies the true generalisation to novel chemistry: classification AUROC fell from 0.936 (random) to 0.907 (scaffold) and regression R² from 0.70 to 0.60 — an honest optimism gap, with the model remaining genuinely predictive on unseen scaffolds. We then screened 2,387 approved small-molecule drugs using two independent lines of evidence — the trained model and nearest-neighbour Tanimoto similarity to potent known inhibitors — with an applicability-domain filter. A structure-based confirmation layer (smina docking into the epacadostat-bound IDO1 pocket, PDB 6IC2) cleanly separated known potent inhibitors (mean −9.12 kcal/mol) from decoys (−6.33, Mann–Whitney *p* < 0.001), and the top repurposing candidates docked in the active range (−8.36). The consensus ranking is enriched in oncology kinase inhibitors — including trametinib, a MEK inhibitor already used in melanoma — suggesting testable polypharmacology hypotheses for IDO1.

## 1. Background

IDO1 catalyses the first, rate-limiting step of tryptophan degradation along the kynurenine pathway; local tryptophan depletion and kynurenine accumulation suppress effector T cells and expand regulatory T cells, contributing to tumour immune escape in melanoma and other malignancies. IDO1 inhibition was therefore an attractive immuno-oncology strategy — but most clinical IDO1 drugs have shown marginal single-agent efficacy, and the epacadostat + pembrolizumab phase III melanoma trial (ECHO-301) did not meet its endpoint. Rather than close the target, this outcome reframed the question toward **combinations and repurposing**: are there already-approved drugs, with established safety profiles, that engage IDO1 as an off-target or secondary activity and could be redeployed?

Computational repurposing is well suited to this question because the chemical matter (approved drugs) and the bioactivity data (ChEMBL) are both public and abundant. The methodological risk is *optimism*: naïve random-split validation of a QSAR model rewards memorising close analogs and overstates how well the model transfers to structurally novel drugs. We address this directly with scaffold-based evaluation and an applicability-domain filter.

## 2. Methods

**Target and data.** Bioactivity for human IDO1 (ChEMBL target CHEMBL4685) was retrieved via the ChEMBL API (IC50, Ki, Kd, EC50), yielding 7,634 raw records (2006–2025). We retained inhibition/binding endpoints (IC50/Ki/Kd) with `standard_relation = '='`, `standard_units = 'nM'`, positive values, and no adverse data-validity flag, then recomputed pIC50 = −log₁₀(value in M) and aggregated to one median value per molecule.

**Chemistry curation.** SMILES were canonicalised with RDKit, salts/mixtures stripped to the largest fragment, and structures deduplicated. PAINS substructures were flagged (362 compounds). The final dataset was 3,585 unique structures with quality pIC50 (median 6.52, range 3.0–9.5), spanning 1,361 Bemis–Murcko scaffolds. We computed nine physicochemical descriptors (MW, cLogP, HBD, HBA, TPSA, rotatable bonds, aromatic rings, QED, heavy atoms) and 2,048-bit radius-2 Morgan fingerprints (**Figure 1A**).

**Model and honest evaluation.** Random-forest classifier (active defined as pIC50 ≥ 6, i.e. ≤ 1 µM; 65.7% actives) and regressor (500 trees) were trained on Morgan fingerprints. Performance was assessed under two 5-fold cross-validation schemes: standard random splitting and **GroupKFold by Murcko scaffold**, in which no scaffold is shared between train and test (**Figure 1B**). We report the optimism gap between the two.

**Repurposing screen.** 3,475 approved small-molecule drugs (ChEMBL `max_phase = 4`) were retrieved; after structure cleaning, removing compounds already tested against IDO1, and deduplication, 2,387 remained. Each drug was scored by (i) the trained model (predicted pIC50 and active probability) and (ii) maximum Tanimoto similarity to the 1,236 potent reference actives (pIC50 ≥ 7). An applicability-domain flag required maximum Tanimoto ≥ 0.30 to any training compound. A rank-normalised consensus of the two independent signals produced the final ranking (**Figure 1C**).

**Structure-based confirmation.** The top candidates, eight potent positive controls, and six low-prediction decoys were docked into the epacadostat-bound IDO1 structure (PDB 6IC2) with smina (exhaustiveness 8), boxing on the co-crystallised inhibitor centroid (**Figure 1D**).

## 3. Results

**The IDO1 inhibitor landscape is diverse and drug-like.** Curated actives cluster into distinct scaffold families in fingerprint space, several visibly enriched for high potency (**Figure 1A**). Median MW 376 Da, cLogP 3.3, QED 0.54 — broadly drug-like.

**Scaffold splitting reveals the true generalisation.** The model is strongly predictive but the choice of validation split matters. Classification AUROC was 0.936 ± 0.009 (random) versus 0.907 ± 0.024 (scaffold); regression R² was 0.697 ± 0.019 versus 0.596 ± 0.048; scaffold-split RMSE 0.72 log units (**Figure 1B**). The ~0.03 AUROC / ~0.10 R² gap is the optimism a random split would have hidden — and, importantly, the model still generalises well to novel scaffolds, which is the regime a repurposing screen operates in.

**Consensus screen prioritises oncology kinase inhibitors.** Of 2,387 approved drugs, 434 fell inside the applicability domain and passed the PAINS filter. The top consensus hits (Table 1) are led by **cabozantinib** (predicted pIC50 7.9) and include several tyrosine-kinase inhibitors — **ensartinib** (ALK, rank 5), **ripretinib** (KIT/PDGFRα, rank 9), **gefitinib** (EGFR, rank 16) — and, at rank 18, **trametinib**, a MEK inhibitor already in clinical use for melanoma (in-domain, PAINS-clean; consensus percentile 0.96). Nearest-neighbour Tanimoto values to known actives are modest (0.30–0.43), so these are genuine analogs rather than near-duplicates of training compounds.

**Docking independently supports the ranking.** The docking setup is valid: positive-control inhibitors scored a mean −9.12 kcal/mol versus −6.33 for decoys (Mann–Whitney *p* < 0.001; **Figure 1D**). The repurposing candidates docked in the active range (mean −8.36 kcal/mol), with ensartinib (−9.42) scoring as well as the best positive control. Because docking is a physics-based signal orthogonal to the ligand-based model, this concordance strengthens the candidate list.

**Table 1** (see `top_candidates_integrated.csv`) lists the top-ranked in-domain candidates with predicted pIC50, active probability, nearest-neighbour Tanimoto, docking affinity, known primary target, and first-approval year.

![Figure 1]({{artifact:art_4cc56e18-923b-4d57-9749-25235ffde906}})

*Figure 1. Computational repurposing pipeline. (A) Chemical space of 3,585 curated IDO1 inhibitors coloured by potency. (B) Leakage-safe evaluation: scaffold splitting exposes the optimism gap relative to random splitting. (C) Consensus screen of 2,387 approved drugs by model prediction and structural similarity. (D) Structure-based confirmation — docking cleanly separates known actives from decoys, and candidates dock like inhibitors.*

## 4. Discussion

The consensus ranking has a coherent biological reading. The enrichment of ATP-competitive kinase inhibitors is consistent with the broadly hydrophobic, flat, heteroaromatic character of both kinase pockets and the IDO1 inhibitor pharmacophore, and raises a concrete, testable polypharmacology hypothesis: that some clinically used kinase inhibitors may engage IDO1 as a secondary activity. The appearance of **trametinib** is especially notable because MEK inhibition is already part of melanoma therapy — a dual MEK/IDO1 engagement, if real, would be mechanistically attractive given IDO1's role in immune escape. The non-oncology hits (vasopressin V2 antagonists, orexin antagonists, a CB1 antagonist) are more likely to be similarity artefacts and are lower priority.

The central methodological contribution is the honest evaluation. A random-split AUROC of 0.94 would have suggested a near-solved prediction problem; the scaffold-split value of 0.91, with a real drop in regression R², is the number that actually predicts performance on novel drugs. This is the difference between a model that looks good and one that is usable for prospective screening.

## 5. Limitations

- **Retrospective, in silico only.** No wet-lab validation. Predicted potencies and docking scores are hypotheses, not measurements.
- **Ligand-based bias.** Morgan-fingerprint models reward similarity to training chemistry; the applicability-domain filter mitigates but does not eliminate this.
- **Docking is approximate.** smina scoring is a coarse binding proxy; the 6IC2 structure is heme-free and captures the inhibitor pocket, but IDO1's catalytic mechanism involves heme, so absolute affinities should not be over-interpreted.
- **Assay heterogeneity.** ChEMBL IC50 values aggregate diverse assay formats; median aggregation reduces but does not remove this noise.
- **Selectivity and safety not modelled.** A drug engaging IDO1 may do so at concentrations irrelevant to its approved dosing; on-target selectivity versus TDO2 was not assessed.

## 6. Next steps

1. **Biochemical validation** of the top 5–10 candidates in a cell-free IDO1 kynurenine assay (order compounds; measure IC50).
2. **Cellular kynurenine assay** in IDO1-expressing melanoma lines (e.g. IFN-γ-induced), reading out tryptophan→kynurenine conversion.
3. **Selectivity counter-screen** against TDO2 and a kinase panel to distinguish on-target IDO1 engagement from the drugs' primary activity.
4. **Combination hypothesis testing** — for trametinib specifically, test whether MEK + IDO1 co-engagement is real and whether it enhances T-cell-mediated killing.
5. **Prospective model tracking** — record which predictions validate, to measure the model's real-world hit rate (ties to deployment-readiness metrics).

## Data and code availability

All analysis was performed on public ChEMBL data with RDKit, scikit-learn, and smina. Curated datasets, the ranked candidate list, trained models, docking outputs, and figures are provided as artifacts.
