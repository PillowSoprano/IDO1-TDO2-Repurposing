"""Stage 1 — curation.  raw ChEMBL bioactivity  ->  curated per-compound potency.

Recipe (Methods sec 2.1-2.2): keep IC50/Ki/Kd with standard_relation '=', units nM,
value > 0, and no data_validity_comment; pActivity = -log10(value in M);
largest-fragment canonical SMILES; median pActivity per compound; PAINS flag.
The historical output column name `pIC50` is retained for compatibility and stores pActivity.

Validates against data/{target}_curated.csv (IDO1 committed; TDO2 rebuilt from features).
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from chem import clean_smiles, is_pains

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def curate(raw_csv):
    raw = pd.read_csv(raw_csv, low_memory=False)
    df = raw[raw["standard_type"].isin(["IC50", "Ki", "Kd"])].copy()
    df = df[df["standard_relation"] == "="]
    df = df[df["standard_units"] == "nM"]
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df[df["standard_value"] > 0]
    df = df[df["data_validity_comment"].isna() |
            (df["data_validity_comment"].astype(str).str.strip() == "")]
    df["pIC50"] = -np.log10(df["standard_value"] * 1e-9)
    df["clean_smiles"] = df["canonical_smiles"].map(clean_smiles)
    df = df.dropna(subset=["clean_smiles"])
    agg = (df.groupby("clean_smiles")
             .agg(pIC50=("pIC50", "median"),
                  molecule_chembl_id=("molecule_chembl_id", "first"),
                  n_measurements=("pIC50", "size"),
                  pref_name=("molecule_pref_name", "first"))
             .reset_index())
    agg["pains"] = agg["clean_smiles"].map(is_pains)
    return agg


def main():
    out = {}
    for tag in ("ido1", "tdo2"):
        cur = curate(os.path.join(DATA, f"{tag}_bioactivity_raw.csv"))
        out_path = os.path.join(DATA, f"{tag}_curated_rebuilt.csv")
        cur.to_csv(out_path, index=False)
        print(f"{tag}: {len(cur)} compounds  median pActivity={cur.pIC50.median():.4f}  "
              f"PAINS={int(cur.pains.sum())}  -> {os.path.basename(out_path)}")
        # validate against reference (IDO1 has a committed curated file)
        ref_path = os.path.join(DATA, f"{tag}_curated.csv")
        if os.path.exists(ref_path):
            ref = pd.read_csv(ref_path)
            m = cur.merge(ref[["clean_smiles", "pIC50"]], on="clean_smiles", suffixes=("", "_ref"))
            d = (m["pIC50"] - m["pIC50_ref"]).abs()
            print(f"    VALIDATE vs {tag}_curated.csv: matched {len(m)}/{len(ref)}  "
                  f"pActivity max|Δ|={d.max():.4f} mean|Δ|={d.mean():.5f}")
        out[tag] = cur
    return out


if __name__ == "__main__":
    main()
