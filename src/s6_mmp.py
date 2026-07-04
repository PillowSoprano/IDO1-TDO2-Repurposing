"""Stage 6 — matched molecular pair (MMP) analysis.

Single-cut MMP fragmentation of the 684 co-tested compounds; for every R-group
transformation recurring in >= 2 matched pairs, mean deltas in shared potency M,
selectivity D, and single-target pActivity values stored in historical `pIC50_*`
columns. Validates transformation counts against
data/mmp_transformations.csv.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, numpy as np, pandas as pd
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from rdkit import Chem
from rdkit.Chem import rdMMPA

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def single_cut_fragments(smi):
    """Single-cut MMP: rdMMPA returns ('', 'fragA[*:1].fragB[*:1]'). Each cut yields two
    (context, r-group) orientations so either fragment can act as the conserved core."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return []
    out = []
    for core, chains in rdMMPA.FragmentMol(m, maxCuts=1, resultsAsMols=False):
        parts = chains.split(".") if core == "" else [core] + chains.split(".")
        if len(parts) != 2:
            continue
        a, b = parts
        out.append((a, b))   # core=a, rgroup=b
        out.append((b, a))   # core=b, rgroup=a
    return out


def main():
    p = pd.read_csv(os.path.join(DATA, "paired_MD_dataset.csv")).dropna(
        subset=["clean_smiles", "M", "D", "pIC50_ido1", "pIC50_tdo2"]).reset_index(drop=True)
    # core -> list of (rgroup, molecule index, prop-vector); require core >= rgroup in heavy
    # atoms (the conserved-context convention) so each cut is counted in one orientation.
    def n_heavy(frag):
        m = Chem.MolFromSmiles(frag)
        return m.GetNumHeavyAtoms() if m else 0
    core_map = defaultdict(list)
    for idx, r in p.iterrows():
        props = np.array([r["M"], r["D"], r["pIC50_ido1"], r["pIC50_tdo2"]], float)
        for core, rgroup in single_cut_fragments(r["clean_smiles"]):
            if n_heavy(core) >= n_heavy(rgroup):
                core_map[core].append((rgroup, idx, props))
    # aggregate transformations; count each unordered MOLECULE pair once per transformation
    trans = defaultdict(list)
    seen = set()
    for core, members in core_map.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(len(members)):
                if i == j:
                    continue
                ra, ia, pa = members[i]; rb, ib, pb = members[j]
                if ra == rb:
                    continue
                key = (ra, rb, min(ia, ib), max(ia, ib))
                if key in seen:
                    continue
                seen.add(key)
                trans[(ra, rb)].append(pb - pa)
    rows = []
    for (ra, rb), deltas in trans.items():
        d = np.array(deltas)
        rows.append({"r_from": ra, "r_to": rb, "n_pairs": len(d),
                     "dM": d[:, 0].mean(), "dD": d[:, 1].mean(),
                     "dIDO1": d[:, 2].mean(), "dTDO2": d[:, 3].mean()})
    mmp = pd.DataFrame(rows)
    mmp = mmp[mmp.n_pairs >= 2].sort_values("n_pairs", ascending=False).reset_index(drop=True)
    mmp.to_csv(os.path.join(DATA, "mmp_transformations_rebuilt.csv"), index=False)
    n2, n3 = int((mmp.n_pairs >= 2).sum()), int((mmp.n_pairs >= 3).sum())
    print(f"MMP transformations: {n2} at >=2 pairs, {n3} at >=3 pairs  (paper: 1780 / 376)")
    ref = pd.read_csv(os.path.join(DATA, "mmp_transformations.csv"))
    print(f"  VALIDATE vs mmp_transformations.csv: reference rows={len(ref)}, "
          f">=3={int((ref.n_pairs>=3).sum())}")


if __name__ == "__main__":
    main()
