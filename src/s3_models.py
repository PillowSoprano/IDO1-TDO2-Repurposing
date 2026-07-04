"""Stage 3 — models and leakage-safe evaluation.

Random-forest classifier (active = pActivity >= 6; historical column `pIC50`)
and regressor on 2048-bit Morgan
fingerprints, each cross-validated under random 5-fold and scaffold GroupKFold (Methods
sec 2.3). Reports AUROC / R2 / RMSE (mean +/- std over folds) and validates against
data/../model_performance.csv (IDO1). Refits full models for downstream use.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from chem import morgan_matrix
from s2_features import featurise
from s1_curate import curate
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
RNG = 0
ACT_THR = 6.0


def cv_metrics(X, y, groups):
    clf = lambda: RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                         class_weight="balanced", random_state=RNG, n_jobs=-1)
    reg = lambda: RandomForestRegressor(n_estimators=500, max_features=1.0,
                                        random_state=RNG, n_jobs=-1)
    yc = (y >= ACT_THR).astype(int)
    out = {}
    for split, cv, grp in [("random", KFold(5, shuffle=True, random_state=RNG), None),
                           ("scaffold", GroupKFold(5), groups)]:
        au, r2, rm = [], [], []
        for tr, te in (cv.split(X, yc, grp) if grp is not None else cv.split(X, yc)):
            c = clf().fit(X[tr], yc[tr])
            au.append(roc_auc_score(yc[te], c.predict_proba(X[te])[:, 1]))
            g = reg().fit(X[tr], y[tr])
            p = g.predict(X[te])
            r2.append(r2_score(y[te], p))
            rm.append(np.sqrt(mean_squared_error(y[te], p)))
        out[split] = {"AUROC": (np.mean(au), np.std(au)),
                      "R2": (np.mean(r2), np.std(r2)),
                      "RMSE": (np.mean(rm), np.std(rm))}
    return out


def main():
    cur = curate(os.path.join(DATA, "ido1_bioactivity_raw.csv"))
    feat = featurise(cur)
    X, keep = morgan_matrix(feat["clean_smiles"].tolist())
    feat = feat[keep].reset_index(drop=True)
    y = feat["pIC50"].to_numpy(float)
    groups = feat["scaffold"].to_numpy()
    print(f"IDO1: {len(feat)} compounds, {int((y>=ACT_THR).mean()*100)}% active")
    res = cv_metrics(X, y, groups)
    for split in ("random", "scaffold"):
        m = res[split]
        print(f"  {split:9s}  AUROC={m['AUROC'][0]:.3f}±{m['AUROC'][1]:.3f}  "
              f"R2={m['R2'][0]:.3f}±{m['R2'][1]:.3f}  RMSE={m['RMSE'][0]:.3f}±{m['RMSE'][1]:.3f}")
    # validate
    ref = pd.read_csv(os.path.join(ROOT, "model_performance.csv"))
    print("  VALIDATE vs model_performance.csv (rebuilt | reference):")
    for _, r in ref.iterrows():
        got = res[r["split"]][r["metric"]][0]
        print(f"    {r['split']:9s} {r['metric']:5s}  {got:.3f} | {r['mean']:.3f}  (Δ={abs(got-r['mean']):.3f})")


if __name__ == "__main__":
    main()
