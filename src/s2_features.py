"""Stage 2 — featurisation.  curated compounds  ->  physicochemical descriptors + scaffold.

Adds the nine descriptors (MW, cLogP, HBD, HBA, TPSA, RotB, AromRings, QED, HeavyAtoms)
and the Bemis-Murcko scaffold used for leakage-safe splitting. (tSNE columns in the
committed features file are a stochastic 2-D embedding for visualisation only and are not
reproduced here.)

Validates descriptor values against data/{target}_features.csv.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from chem import descriptors, murcko_scaffold
from s1_curate import curate

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
DESC_COLS = ["MW", "LogP", "HBD", "HBA", "TPSA", "RotB", "AromRings", "QED", "HeavyAtoms"]


def featurise(cur):
    d = pd.DataFrame([descriptors(s) for s in cur["clean_smiles"]], index=cur.index)
    out = pd.concat([cur, d], axis=1)
    out["scaffold"] = out["clean_smiles"].map(murcko_scaffold)
    return out


def main():
    for tag in ("ido1", "tdo2"):
        cur = curate(os.path.join(DATA, f"{tag}_bioactivity_raw.csv"))
        feat = featurise(cur)
        out_path = os.path.join(DATA, f"{tag}_features_rebuilt.csv")
        feat.to_csv(out_path, index=False)
        print(f"{tag}: {len(feat)} rows, {len(DESC_COLS)} descriptors -> {os.path.basename(out_path)}")
        ref = pd.read_csv(os.path.join(DATA, f"{tag}_features.csv"))
        avail = [c for c in DESC_COLS if c in ref.columns]      # tdo2 file carries a reduced set
        m = feat.merge(ref[["clean_smiles"] + avail], on="clean_smiles", suffixes=("", "_ref"))
        worst = max((m[c] - m[c + "_ref"]).abs().max() for c in avail)
        # scaffold agreement
        ms = feat.merge(ref[["clean_smiles", "scaffold"]], on="clean_smiles", suffixes=("", "_ref"))
        scaf_match = (ms["scaffold"] == ms["scaffold_ref"]).mean()
        print(f"    VALIDATE vs {tag}_features.csv: matched {len(m)}/{len(ref)}  "
              f"worst descriptor max|Δ|={worst:.4g}  scaffold match={scaf_match:.3f}")


if __name__ == "__main__":
    main()
