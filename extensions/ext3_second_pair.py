"""Extension 3 — Does the framework transfer to a second target pair?

The paper claims (Discussion, "Generalizability") that the paired-target template applies
beyond IDO1/TDO2, but shows no second example. We instantiate the entire pipeline on the
canonical selectivity pair COX-1 (PTGS1, CHEMBL221) / COX-2 (PTGS2, CHEMBL230):

  fetch ChEMBL -> curate -> co-tested set -> measured cross-target correlation r
  -> M/D reparametrisation -> split-hardness ladder + LODO -> conformal coverage.

If the SAME phenomena reappear (a moderate measured r, a scaffold->LODO skill collapse,
calibrated conformal intervals), the template is shown to generalise on independent data.

Data are fetched live from the ChEMBL REST API and cached to extensions/cache_cox/.
"""
import warnings; warnings.filterwarnings("ignore")
import os, time, json, urllib.request, urllib.parse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (KFold, GroupKFold, LeaveOneGroupOut,
                                     GroupShuffleSplit, cross_val_predict)
from sklearn.metrics import r2_score
from common import fp, fps, murcko_scaffold
from rdkit import Chem
from rdkit.Chem import MolStandardize
from rdkit.Chem.MolStandardize import rdMolStandardize

RNG = 0
CACHE = "cache_cox"; os.makedirs(CACHE, exist_ok=True)
TARGETS = {"COX1": "CHEMBL221", "COX2": "CHEMBL230"}
BASE = "https://www.ebi.ac.uk/chembl/api/data/activity.json"


def fetch_target(chembl_id, tag):
    path = os.path.join(CACHE, f"{tag}_raw.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  {tag}: {len(df)} cached records")
        return df
    rows, offset, limit = [], 0, 1000
    while True:
        q = {"target_chembl_id": chembl_id, "standard_type__in": "IC50,Ki,Kd",
             "limit": limit, "offset": offset}
        url = BASE + "?" + urllib.parse.urlencode(q)
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    data = json.load(r)
                break
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(3)
        acts = data.get("activities", [])
        for a in acts:
            rows.append({k: a.get(k) for k in
                         ["molecule_chembl_id", "canonical_smiles", "standard_type",
                          "standard_relation", "standard_units", "standard_value",
                          "pchembl_value", "document_chembl_id", "data_validity_comment"]})
        offset += limit
        print(f"    {tag}: fetched {len(rows)} ...", end="\r")
        if len(acts) < limit:
            break
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"\n  {tag}: {len(df)} records fetched & cached")
    return df


_norm = rdMolStandardize.LargestFragmentChooser()
def clean_smiles(smi):
    if not isinstance(smi, str):
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        m = _norm.choose(m)
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def curate(df):
    """Replicate the paper's curation: valid '=' IC50/Ki/Kd in nM, positive, no adverse
    flag; pIC50 = pchembl_value (or -log10(M)); median per molecule; largest fragment."""
    df = df.copy()
    df = df[df["standard_relation"] == "="]
    df = df[df["standard_units"] == "nM"]
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df[df["standard_value"] > 0]
    df = df[df["data_validity_comment"].isna() | (df["data_validity_comment"].astype(str).str.strip() == "")]
    pchembl = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df["pIC50"] = np.where(pchembl.notna(), pchembl, -np.log10(df["standard_value"] * 1e-9))
    df["clean_smiles"] = df["canonical_smiles"].map(clean_smiles)
    df = df.dropna(subset=["clean_smiles", "pIC50"])
    # median per (molecule) potency; keep one representative document (mode) per molecule
    agg = (df.groupby("clean_smiles")
             .agg(pIC50=("pIC50", "median"),
                  src_doc=("document_chembl_id", lambda s: s.mode().iat[0] if len(s.mode()) else s.iat[0]),
                  molid=("molecule_chembl_id", "first"))
             .reset_index())
    return agg


def pooled_r2(est, X, y, cv, groups=None):
    yhat = cross_val_predict(est, X, y, cv=cv, groups=groups, n_jobs=-1)
    return r2_score(y, yhat)


def main():
    print("fetching ChEMBL bioactivity (COX-1, COX-2) ...")
    raw = {tag: fetch_target(cid, tag) for tag, cid in TARGETS.items()}
    cur = {tag: curate(df) for tag, df in raw.items()}
    for tag in TARGETS:
        print(f"  curated {tag}: {len(cur[tag])} compounds  median pActivity={cur[tag].pIC50.median():.2f}")

    # co-tested set
    m = cur["COX1"].merge(cur["COX2"], on="clean_smiles", suffixes=("_cox1", "_cox2"))
    m["M"] = (m["pIC50_cox1"] + m["pIC50_cox2"]) / 2
    m["D"] = m["pIC50_cox1"] - m["pIC50_cox2"]
    m["scaffold"] = m["clean_smiles"].map(murcko_scaffold)
    # provenance: use COX-2 source doc as the LODO grouping (analog to paper)
    m["src_doc"] = m["src_doc_cox2"]
    m = m.dropna(subset=["clean_smiles", "M", "D"]).reset_index(drop=True)
    X, keep = fps(m["clean_smiles"].tolist()); m = m[keep].reset_index(drop=True)
    print(f"\nco-tested COX-1/COX-2 compounds: {len(m)}  "
          f"documents={m['src_doc'].nunique()}  scaffolds={m['scaffold'].nunique()}")
    m.to_csv("ext3_cox_paired.csv", index=False)

    # measured cross-target correlation (analog to r=0.43)
    r, p = stats.pearsonr(m["pIC50_cox1"], m["pIC50_cox2"])
    n = len(m); z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    ci = np.tanh([z - 1.96 * se, z + 1.96 * se])
    print(f"\nMEASURED COX-1/COX-2 correlation: Pearson r={r:.3f} "
          f"95% CI [{ci[0]:.2f},{ci[1]:.2f}]  p={p:.1e}  (IDO1/TDO2 was r=0.43)")
    bal = (m["D"].abs() <= 1).mean()
    print(f"balanced-dual fraction (|D|<=1): {bal:.2f}")

    # split-hardness ladder + LODO for M and D
    yM, yD = m["M"].to_numpy(), m["D"].to_numpy()
    g_scaf, g_doc = m["scaffold"].to_numpy(), m["src_doc"].to_numpy()
    rf = lambda: RandomForestRegressor(n_estimators=300, random_state=RNG, n_jobs=-1)
    ladder = []
    for tgt, y in [("M", yM), ("D", yD)]:
        r_rand = pooled_r2(rf(), X, y, KFold(5, shuffle=True, random_state=RNG))
        r_scaf = pooled_r2(rf(), X, y, GroupKFold(5), groups=g_scaf)
        r_lodo = pooled_r2(rf(), X, y, LeaveOneGroupOut(), groups=g_doc)
        ladder.append({"target": tgt, "random": r_rand, "scaffold": r_scaf, "LODO": r_lodo})
        print(f"  {tgt}:  random R2={r_rand:+.3f}  scaffold={r_scaf:+.3f}  LODO={r_lodo:+.3f}")
    pd.DataFrame(ladder).to_csv("ext3_cox_splithardness.csv", index=False)

    # conformal coverage on a scaffold-disjoint 3-way split
    gss1 = GroupShuffleSplit(1, test_size=0.41, random_state=RNG)
    rest, test = next(gss1.split(X, yM, g_scaf))
    gss2 = GroupShuffleSplit(1, test_size=0.45, random_state=RNG)
    ptr_r, cal_r = next(gss2.split(X[rest], yM[rest], g_scaf[rest]))
    ptr, cal = rest[ptr_r], rest[cal_r]
    mM = rf().fit(X[ptr], yM[ptr]); mD = rf().fit(X[ptr], yD[ptr])
    eM, eD = yM[cal] - mM.predict(X[cal]), yD[cal] - mD.predict(X[cal])
    cov = {}
    for tgt, mdl, e, ytrue in [("M", mM, eM, yM[test]), ("D", mD, eD, yD[test])]:
        pred = mdl.predict(X[test])
        lo = pred + np.quantile(e, 0.05); hi = pred + np.quantile(e, 0.95)
        cov[tgt] = float(((ytrue >= lo) & (ytrue <= hi)).mean())
    print(f"\nconformal 90% empirical coverage:  M={cov['M']:.2f}  D={cov['D']:.2f}")

    # figure: (A) measured correlation scatter, (B) split-hardness ladder
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].scatter(m["pIC50_cox2"], m["pIC50_cox1"], s=10, alpha=0.4, color="#2166ac")
    b, a = np.polyfit(m["pIC50_cox2"], m["pIC50_cox1"], 1)
    xs = np.array([m["pIC50_cox2"].min(), m["pIC50_cox2"].max()])
    ax[0].plot(xs, a + b * xs, "r-", lw=2)
    for t in (6, 7):
        ax[0].axhline(t, ls=":", color="grey", lw=1); ax[0].axvline(t, ls=":", color="grey", lw=1)
    ax[0].set_xlabel("measured COX-2 pIC50"); ax[0].set_ylabel("measured COX-1 pIC50")
    ax[0].set_title(f"(A) COX-1/COX-2 co-tested (n={n})\nPearson r={r:.2f} [{ci[0]:.2f},{ci[1]:.2f}]")
    ld = pd.DataFrame(ladder).set_index("target")
    xx = np.arange(2); w = 0.26
    ax[1].bar(xx - w, ld["random"], w, label="random", color="#4575b4")
    ax[1].bar(xx,     ld["scaffold"], w, label="scaffold", color="#fdae61")
    ax[1].bar(xx + w, ld["LODO"], w, label="LODO", color="#d73027")
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set_xticks(xx); ax[1].set_xticklabels(["M", "D"])
    ax[1].set_ylabel("pooled global R²"); ax[1].legend(frameon=False)
    ax[1].set_title("(B) Same scaffold→LODO collapse reappears")
    fig.tight_layout(); fig.savefig("../figures/figure_second_pair_cox.png", dpi=150)
    print("\nwrote figures/figure_second_pair_cox.png, ext3_cox_paired.csv, ext3_cox_splithardness.csv")


if __name__ == "__main__":
    main()
