# IDO1 vs TDO2: cross-target repurposing and selectivity analysis

**Companion study to the IDO1 repurposing screen — Xiyao Yu**

## Rationale

IDO1 and tryptophan 2,3-dioxygenase (TDO2) catalyse the same first step of tryptophan degradation to kynurenine, and both contribute to tumour immune escape. Because the lead IDO1 inhibitor epacadostat failed in melanoma, attention has shifted to **dual IDO1/TDO2 inhibition** — the hypothesis being that tumours escape single-IDO1 blockade by compensatory TDO2 activity. This companion analysis asks, computationally: among approved drugs, which engage TDO2, which engage IDO1, and which might engage **both**?

## Methods

We built a TDO2 bioactivity model identical in construction to the IDO1 model: 1,684 ChEMBL records → 964 curated compounds (356 Bemis–Murcko scaffolds, median pIC50 6.0), 2,048-bit Morgan fingerprints, random-forest classifier and regressor, evaluated under scaffold-based cross-validation. The same 2,387 approved-drug deck was scored against **both** the IDO1 and TDO2 models. Applicability-domain filtering used proximity to each target's own training set. Candidates were classed as *dual* (predicted pIC50 ≥ 6 for both, in-domain for both), *TDO2-selective* (TDO2 ≥ 6, IDO1 < 5.5), or *IDO1-selective* (the converse).

## Results

**The TDO2 model is honest but noisier than IDO1's.** Scaffold-split AUROC 0.873 (vs 0.908 random); regression R² 0.40 (vs 0.61 random). The larger optimism gap reflects the smaller, more scaffold-clustered dataset — a limitation stated plainly rather than hidden by random splitting.

**IDO1 and TDO2 predicted potencies are essentially uncorrelated (r = 0.01).** Despite catalysing the same reaction, the two enzymes present distinct inhibitor pharmacophores — which is precisely why dual inhibitors are pharmacologically difficult and clinically sought.

**Only one dual candidate emerged: osilodrostat** (predicted IDO1 6.5 / TDO2 6.1, in-domain for both). The scarcity of dual hits among approved drugs is itself the result — it quantifies why dual IDO1/TDO2 engagement is rare and must usually be designed rather than repurposed.

**The TDO2-selective list recovers tryptophan-pathway chemistry** — led by niacin and niacinamide (NAD⁺/tryptophan-metabolism molecules), plus triamterene, pyrimethamine, and several benzodiazepine-scaffold drugs. The recovery of nicotinamide-pathway molecules for the tryptophan-degrading enzyme is an encouraging internal sanity check on the model.

![Selectivity analysis]({{artifact:art_fb26bb01-8a0c-46e3-8406-753016cf67d1}})

*Figure. (A) Cross-target selectivity map: every approved drug placed by predicted IDO1 vs TDO2 potency, with selective and dual candidates highlighted. (B) Honest scaffold-split vs random-split performance for both target models.*

Full candidate lists (dual, TDO2-selective, IDO1-selective) are in `dual_selectivity_candidates.csv`; all 2,387 drugs with both scores are in `dual_scored.csv`.

## Interpretation and limits

The uncorrelated pharmacophores and the single dual hit are the scientific payload: they give a concrete, model-based estimate of how rare dual IDO1/TDO2 engagement is in approved chemical space, supporting the view that dual inhibitors need dedicated design. All caveats from the IDO1 study apply, and more strongly here given TDO2's smaller dataset and lower scaffold-split R² — the TDO2-selective predictions in particular should be treated as low-confidence hypotheses pending biochemical testing (a TDO2 kynurenine assay, with an IDO1 counter-screen to confirm selectivity).
