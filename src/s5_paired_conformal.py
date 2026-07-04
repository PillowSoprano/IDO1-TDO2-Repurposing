"""Stage 5 — paired M/D framework: split-hardness ladder, LODO, and conformal coverage.

Reparametrises the 684 co-tested compounds into shared potency M and selectivity D and
evaluates RF regressors under five split schemes (random, scaffold, Butina, chemical-space,
temporal) and leave-one-document-out; then fits split-conformal intervals and reports the
empirical-vs-nominal coverage curve.

Validates against data/split_hardness.csv, data/lodo_results.csv, conformal_calibration.csv.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from chem import morgan_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (KFold, GroupKFold, LeaveOneGroupOut,
                                     GroupShuffleSplit, cross_val_predict)
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
RNG = 0


def rf():
    return RandomForestRegressor(n_estimators=300, random_state=RNG, n_jobs=-1)


def pooled_r2(X, y, cv, groups=None):
    yhat = cross_val_predict(rf(), X, y, cv=cv, groups=groups, n_jobs=-1)
    return r2_score(y, yhat), np.sqrt(mean_squared_error(y, yhat))


def butina_groups(X, cutoff=0.4):
    from rdkit.DataStructs import CreateFromBitString
    bvs = [CreateFromBitString("".join(map(str, row))) for row in X.astype(int)]
    n = len(bvs)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(bvs[i], bvs[:i])
        dists.extend(1 - s for s in sims)
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    lab = np.empty(n, dtype=int)
    for cid, members in enumerate(clusters):
        for m in members:
            lab[m] = cid
    return lab


def chemspace_groups(X, k=10):
    try:
        import umap
        emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="jaccard",
                        random_state=RNG).fit_transform(X)
    except Exception as e:
        print(f"    (UMAP unavailable: {type(e).__name__}; using PCA fallback for chemspace)")
        from sklearn.decomposition import PCA
        emb = PCA(n_components=10, random_state=RNG).fit_transform(X.astype(float))
    return KMeans(n_clusters=k, random_state=RNG, n_init=10).fit_predict(emb)


def main():
    p = pd.read_csv(os.path.join(DATA, "paired_MD_dataset.csv")).dropna(
        subset=["clean_smiles", "M", "D", "scaffold", "src_doc", "year"]).reset_index(drop=True)
    X, keep = morgan_matrix(p["clean_smiles"].tolist())
    p = p[keep].reset_index(drop=True)
    yM, yD = p["M"].to_numpy(float), p["D"].to_numpy(float)
    g_scaf, g_doc = p["scaffold"].to_numpy(), p["src_doc"].to_numpy()
    year = p["year"].to_numpy()
    print(f"paired compounds: {len(p)}")

    g_but = butina_groups(X)
    g_chem = chemspace_groups(X)
    print(f"Butina clusters={len(set(g_but))}  chemspace clusters={len(set(g_chem))}")

    ladder = []
    for tgt, y in [("M", yM), ("D", yD)]:
        r_rand = pooled_r2(X, y, KFold(5, shuffle=True, random_state=RNG))[0]
        r_scaf = pooled_r2(X, y, GroupKFold(5), groups=g_scaf)[0]
        r_but = pooled_r2(X, y, GroupKFold(5), groups=g_but)[0]
        r_chem = pooled_r2(X, y, GroupKFold(5), groups=g_chem)[0]
        # temporal: train<=2021 / test>=2022 (single pooled split)
        def temporal(cut):
            tr, te = year <= cut, year > cut
            g = rf().fit(X[tr], y[tr]); pr = g.predict(X[te])
            return r2_score(y[te], pr), int(te.sum())
        r_t22, n22 = temporal(2021)
        r_t21, n21 = temporal(2020)
        r_lodo = pooled_r2(X, y, LeaveOneGroupOut(), groups=g_doc)[0]
        ladder += [
            {"target": tgt, "split": "random", "R2": r_rand},
            {"target": tgt, "split": "scaffold", "R2": r_scaf},
            {"target": tgt, "split": "butina", "R2": r_but},
            {"target": tgt, "split": "chemspace", "R2": r_chem},
            {"target": tgt, "split": "temporal_2022+", "R2": r_t22},
            {"target": tgt, "split": "temporal_2021+", "R2": r_t21},
            {"target": tgt, "split": "leave-document-out", "R2": r_lodo},
        ]
        print(f"  {tgt}: random={r_rand:+.3f} scaffold={r_scaf:+.3f} butina={r_but:+.3f} "
              f"chemspace={r_chem:+.3f} temporal22={r_t22:+.3f} LODO={r_lodo:+.3f}")
    lad = pd.DataFrame(ladder)
    lad.to_csv(os.path.join(DATA, "split_hardness_rebuilt.csv"), index=False)

    # validate against references
    ref_sh = pd.read_csv(os.path.join(DATA, "split_hardness.csv"))
    ref_lodo = pd.read_csv(os.path.join(DATA, "lodo_results.csv"))
    print("  VALIDATE vs split_hardness.csv (|Δ R2| per row):")
    for _, r in ref_sh.iterrows():
        got = lad[(lad.target == r.target) & (lad.split == r.split)]
        if len(got):
            print(f"    {r.target} {r.split:15s} rebuilt={got.R2.iat[0]:+.3f} ref={r.R2:+.3f} "
                  f"Δ={abs(got.R2.iat[0]-r.R2):.3f}")
    print("  VALIDATE vs lodo_results.csv:")
    for _, r in ref_lodo.iterrows():
        got = lad[(lad.target == r.target) & (lad.split == "leave-document-out")]
        print(f"    {r.target} LODO  rebuilt={got.R2.iat[0]:+.3f} ref={r.R2:+.3f} "
              f"Δ={abs(got.R2.iat[0]-r.R2):.3f}")

    # conformal coverage curve on scaffold-disjoint 3-way split
    gss1 = GroupShuffleSplit(1, test_size=0.41, random_state=RNG)
    rest, test = next(gss1.split(X, yM, g_scaf))
    gss2 = GroupShuffleSplit(1, test_size=0.45, random_state=RNG)
    pr, ca = next(gss2.split(X[rest], yM[rest], g_scaf[rest]))
    ptr, cal = rest[pr], rest[ca]
    mM = rf().fit(X[ptr], yM[ptr]); mD = rf().fit(X[ptr], yD[ptr])
    eM, eD = yM[cal] - mM.predict(X[cal]), yD[cal] - mD.predict(X[cal])
    MhT, DhT = mM.predict(X[test]), mD.predict(X[test])
    ido1_obs, tdo2_obs = yM[test] + yD[test] / 2, yM[test] - yD[test] / 2
    rows = []
    for nominal in np.round(np.arange(0.5, 0.951, 0.05), 2):
        a = (1 - nominal)
        def cov(hat, e, obs):
            lo = hat[:, None] + np.quantile(e, a / 2)
            hi = hat[:, None] + np.quantile(e, 1 - a / 2)
            return ((obs >= lo[:, 0]) & (obs <= hi[:, 0])).mean()
        ido1_s = MhT[:, None] + eM[None, :] + (DhT[:, None] + eD[None, :]) / 2
        tdo2_s = MhT[:, None] + eM[None, :] - (DhT[:, None] + eD[None, :]) / 2
        def cov_pair(samp, obs):
            lo = np.quantile(samp, a / 2, axis=1); hi = np.quantile(samp, 1 - a / 2, axis=1)
            return ((obs >= lo) & (obs <= hi)).mean()
        rows.append({"nominal": nominal,
                     "IDO1": cov_pair(ido1_s, ido1_obs), "TDO2": cov_pair(tdo2_s, tdo2_obs),
                     "M": cov(MhT, eM, yM[test]), "D": cov(DhT, eD, yD[test])})
    cal_df = pd.DataFrame(rows)
    cal_df.to_csv(os.path.join(DATA, "conformal_calibration_rebuilt.csv"), index=False)
    print(f"  conformal coverage @0.90: IDO1={cal_df.iloc[-2].IDO1:.2f} TDO2={cal_df.iloc[-2].TDO2:.2f} "
          f"M={cal_df.iloc[-2].M:.2f} D={cal_df.iloc[-2].D:.2f}")


if __name__ == "__main__":
    main()
