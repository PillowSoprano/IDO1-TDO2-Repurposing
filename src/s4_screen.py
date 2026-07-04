"""Stage 4 — repurposing screen.

Full-data IDO1 and TDO2 models score the approved-drug deck (predicted pActivity,
active
probability) plus an applicability-domain flag (max Tanimoto >= 0.30 to the training set).
Validates predicted potencies/probabilities against data/dual_scored.csv.

The deck SMILES are read from the committed data/dual_scored.csv (the upstream ChEMBL
max_phase=4 fetch is not cached in the repo); this stage reproduces the *scoring*.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from chem import morgan_matrix, bulk_tanimoto_max
from s1_curate import curate
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RNG = 0; ACT_THR = 6.0; AD_THR = 0.30


def fit_target(tag):
    cur = curate(os.path.join(DATA, f"{tag}_bioactivity_raw.csv"))
    X, keep = morgan_matrix(cur["clean_smiles"].tolist())
    cur = cur[keep].reset_index(drop=True)
    y = cur["pIC50"].to_numpy(float)
    reg = RandomForestRegressor(n_estimators=500, max_features=1.0, random_state=RNG,
                                n_jobs=-1).fit(X, y)
    clf = RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced",
                                 random_state=RNG, n_jobs=-1).fit(X, (y >= ACT_THR).astype(int))
    return reg, clf, X


def main():
    deck = pd.read_csv(os.path.join(DATA, "dual_scored.csv"))
    Xd, keepd = morgan_matrix(deck["clean_smiles"].tolist())
    deck = deck[keepd].reset_index(drop=True)
    got = {}
    for tag in ("ido1", "tdo2"):
        reg, clf, Xtr = fit_target(tag)
        got[f"{tag}_pred"] = reg.predict(Xd)
        got[f"{tag}_prob"] = clf.predict_proba(Xd)[:, 1]
        got[f"{tag}_ad"] = bulk_tanimoto_max(Xd, Xtr)   # max Tanimoto to training (AD proximity)
    rebuilt = deck[["molecule_chembl_id", "clean_smiles"]].copy()
    for k, v in got.items():
        rebuilt[k] = v
    rebuilt.to_csv(os.path.join(DATA, "dual_scored_rebuilt.csv"), index=False)
    print(f"scored {len(deck)} approved drugs -> dual_scored_rebuilt.csv")
    print("  VALIDATE vs dual_scored.csv (rebuilt vs reference):")
    for col in ("ido1_pred", "tdo2_pred", "ido1_prob", "tdo2_prob"):
        d = (rebuilt[col] - deck[col]).abs()
        print(f"    {col:10s}  max|Δ|={d.max():.4f}  mean|Δ|={d.mean():.5f}  corr={np.corrcoef(rebuilt[col], deck[col])[0,1]:.4f}")
    for col in ("ido1_ad", "tdo2_ad"):       # AD stored as max-Tanimoto value (float)
        d = (rebuilt[col] - deck[col]).abs()
        print(f"    {col:10s}  max|Δ|={d.max():.4f}  mean|Δ|={d.mean():.5f}  corr={np.corrcoef(rebuilt[col], deck[col])[0,1]:.4f}")


if __name__ == "__main__":
    main()
