# Archived / superseded files

These files are **outdated** and retained only for provenance. Do **not** cite numbers
or structures from them — the authoritative manuscript is `../preprint.md`.

## Superseded manuscript drafts

- **`repurposing_report.md`** — early IDO1-only draft. Docks into **PDB 6IC2**, later
  found to be carbonic anhydrase II (not IDO1). All structural results were regenerated
  against **PDB 6E40** in the current preprint.
- **`selectivity_report.md`** — early cross-target draft. States IDO1/TDO2 are
  "essentially uncorrelated (r = 0.01), distinct pharmacophores" and treats the niacin
  hits as a "sanity check." Both claims are **retracted** in the current preprint, which
  reports the model-free measured correlation **r = 0.43** (684 co-tested compounds) and
  explains the niacin hits as substrate-similarity artefacts.

## Stale 6IC2-era structural outputs

All contain **carbonic anhydrase II pocket residues** (e.g. Gln92, His64, His119, Val121,
Phe131), not the IDO1 inhibitor pocket (Phe163, Phe226, Arg231, Tyr126, Ser263, Leu234).
Superseded by `../data/candidate_pocket_overlap_6e40.csv` and `../complexes/`.

- **`6IC2.pdb`** — wrong target (carbonic anhydrase II).
- **`candidate_pocket_overlap.csv`** — pocket overlap computed against 6IC2.
- **`ensartinib_ido1_contacts.csv`** — ensartinib contacts computed against 6IC2.
- **`ido1_ensartinib_complex.pdb`** — ensartinib docked into 6IC2; superseded by
  `../complexes/ido1_ensartinib_complex.pdb` (6E40).
