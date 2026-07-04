"""Extension 1 - Provenance-Adaptive Conformal Ranking (PACR).

The paper's split-conformal intervals have essentially constant width, so coverage is
marginal rather than conditional. The first extension compared that baseline with a
normalized conformal predictor using only RF tree-to-tree variance as sigma(x).

This upgraded analysis adds PACR: a source-shift-aware difficulty scale learned from
proper-train leave-document-out residuals, then held fixed while the calibration fold supplies the
conformal residual distribution. The scale uses:

  chem_dist(x)  = 1 - max Tanimoto similarity to proper-train compounds
  model_unc(x)  = RF tree-to-tree prediction standard deviation, used as the base scale
  doc_dist(x)   = 1 - nearest source-document chemical-domain support

The M/D residuals remain paired throughout interval construction, preserving their
correlation when reconstructing IDO1 and TDO2.
"""
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit

from common import fps, bulk_tanimoto_max


HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data")
FIGURES = os.path.join(ROOT, "figures")

RNG = 0
ALPHA = 0.10                      # 90% intervals
THR = 6.0                         # pActivity threshold (pIC50-equivalent <=1 uM)
EPS = 1e-6
PACR_TAU = 0.75                   # fit an upper residual quantile, not mean error
DOC_TOP_K = 5                      # DSS averages top min(k, n_doc) similarities per document
FEATURE_NAMES = ["chem_dist", "doc_dist"]
METHODS = ["standard", "normalized", "pacr"]
COLORS = {"standard": "#888888", "normalized": "#1b7837", "pacr": "#2166ac"}

np.random.seed(RNG)


def rf():
    return RandomForestRegressor(
        n_estimators=300,
        random_state=RNG,
        n_jobs=-1,
        oob_score=True,
        bootstrap=True,
    )


def tree_std(model, Xq):
    """Per-sample std of predictions across trees = model-internal uncertainty."""
    preds = np.stack([t.predict(Xq) for t in model.estimators_], axis=0)
    return preds.std(0)


def tanimoto_to_train(Xq, ref_X, leave_one_out=False):
    """Return full Tanimoto matrix to the reference set, optionally zeroing self hits."""
    _, tan = bulk_tanimoto_max(Xq, ref_X)
    if leave_one_out:
        n = min(tan.shape)
        tan[np.arange(n), np.arange(n)] = 0.0
    return tan


def document_domain_distance(tan, ref_docs, k=DOC_TOP_K):
    """Distance to the nearest source-document chemical domain.

    For each source document, average the top min(k, n_doc) compound similarities inside
    that document; documents with fewer than k compounds use all available compounds.
    The query's document support is the best such document-level average. The caller
    supplies the reference documents, which are proper-train documents in PACR.
    """
    ref_docs = np.asarray(ref_docs)
    best_support = np.zeros(tan.shape[0], dtype=float)
    for doc in np.unique(ref_docs):
        cols = np.flatnonzero(ref_docs == doc)
        if len(cols) == 0:
            continue
        sims = tan[:, cols]
        kk = min(k, sims.shape[1])
        if kk == sims.shape[1]:
            support = sims.mean(axis=1)
        else:
            support = np.partition(sims, -kk, axis=1)[:, -kk:].mean(axis=1)
        best_support = np.maximum(best_support, support)
    return 1.0 - best_support


def domain_features(Xq, ref_X, ref_docs, leave_one_out=False):
    tan = tanimoto_to_train(Xq, ref_X, leave_one_out=leave_one_out)
    chem_dist = 1.0 - tan.max(axis=1)
    doc_dist = document_domain_distance(tan, ref_docs)
    return chem_dist, doc_dist


def source_feature_matrix(Xq, ref_X, ref_docs, leave_one_out=False):
    chem_dist, doc_dist = domain_features(Xq, ref_X, ref_docs, leave_one_out)
    return np.column_stack([chem_dist, doc_dist])


def fit_feature_scaler(Z):
    lo = np.nanpercentile(Z, 5, axis=0)
    hi = np.nanpercentile(Z, 95, axis=0)
    span = np.where((hi - lo) > EPS, hi - lo, 1.0)
    return {"lo": lo, "span": span}


def transform_features(Z, scaler):
    return np.clip((Z - scaler["lo"]) / scaler["span"], 0.0, 1.5)


def fit_source_scale(residual, Zraw, base_scale, target):
    """Learn positive sigma(x) = base(x) * exp(theta0) * (1 + theta*z).

    The base is the RF-normalized conformal scale. The global multiplier is constrained
    to be at least 1, and slopes are constrained non-negative, so PACR can only widen
    the RF-normalized interval when source-shift risk is high. The final conformal
    calibration fold is not used to fit theta.
    """
    residual = np.asarray(residual, dtype=float)
    Zraw = np.asarray(Zraw, dtype=float)
    base_scale = np.asarray(base_scale, dtype=float)
    ok = np.isfinite(residual) & np.isfinite(base_scale) & np.all(np.isfinite(Zraw), axis=1)
    y = np.abs(residual[ok]) + EPS
    base = base_scale[ok] + EPS
    scaler = fit_feature_scaler(Zraw[ok])
    Z = transform_features(Zraw[ok], scaler)

    init = np.r_[max(0.0, np.log(np.median(y / base))), np.zeros(Z.shape[1])]
    bounds = [(0.0, 4.0)] + [(0.0, 10.0)] * Z.shape[1]

    ratio = y / base

    def objective(theta):
        pred = np.exp(theta[0]) * (1.0 + Z @ theta[1:]) + EPS
        diff = ratio - pred
        pinball = np.where(diff >= 0, PACR_TAU * diff, (PACR_TAU - 1.0) * diff)
        return float(np.mean(pinball) + 0.01 * np.sum(theta[1:] ** 2))

    opt = minimize(objective, init, method="L-BFGS-B", bounds=bounds)
    theta = opt.x if opt.success else init
    fitted = {
        "target": target,
        "theta": theta,
        "scaler": scaler,
        "success": bool(opt.success),
        "objective": float(opt.fun if opt.success else objective(theta)),
    }
    return fitted


def apply_source_scale(model_info, Zraw, base_scale):
    Z = transform_features(np.asarray(Zraw, dtype=float), model_info["scaler"])
    return np.asarray(base_scale, dtype=float) * np.exp(model_info["theta"][0]) * (1.0 + Z @ model_info["theta"][1:]) + EPS


def source_document_residual_features(y, label):
    """Proper-train internal leave-document-out residuals for scale fitting."""
    residual = np.full(len(Xtr), np.nan, dtype=float)
    Zraw = np.full((len(Xtr), len(FEATURE_NAMES)), np.nan, dtype=float)
    base = np.full(len(Xtr), np.nan, dtype=float)
    for doc in np.unique(docs_tr):
        held = docs_tr == doc
        train = ~held
        if train.sum() < 20:
            continue
        fold_model = rf().fit(Xtr[train], y[train])
        residual[held] = y[held] - fold_model.predict(Xtr[held])
        fold_ridge = np.median(tree_std(fold_model, Xtr[train]))
        base[held] = tree_std(fold_model, Xtr[held]) + fold_ridge
        Zraw[held] = source_feature_matrix(Xtr[held], Xtr[train], docs_tr[train])
    print(f"PACR scale-fit {label}: leave-document-out residuals for {np.isfinite(residual).sum()} compounds")
    return residual, Zraw, base


def qbin(values):
    labels = ["Q1_near", "Q2", "Q3", "Q4_far"]
    ranks = pd.Series(values).rank(method="first")
    return np.asarray(pd.qcut(ranks, 4, labels=labels))


def coverage_mad(res, method, axis):
    sub = res[(res.method == method) & (res.target == "IDO1") & (res.bin_type == axis)]
    return float(np.mean(np.abs(sub.coverage - (1 - ALPHA))))


# ---------------------------------------------------------------- data
paired = pd.read_csv(os.path.join(DATA, "paired_MD_dataset.csv"))
paired = paired.dropna(subset=["clean_smiles", "M", "D", "scaffold", "src_doc"]).reset_index(drop=True)
X, keep = fps(paired["clean_smiles"].tolist())
paired = paired[keep].reset_index(drop=True)
M = paired["M"].to_numpy(float)
D = paired["D"].to_numpy(float)
groups = paired["scaffold"].to_numpy()
docs = paired["src_doc"].to_numpy()
print(f"paired compounds with FP: {len(paired)}")

# ---------------------------------------------------------------- scaffold-disjoint 3-way split
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.41, random_state=RNG)
rest_idx, test_idx = next(gss1.split(X, M, groups))
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.45, random_state=RNG)
ptr_rel, cal_rel = next(gss2.split(X[rest_idx], M[rest_idx], groups[rest_idx]))
ptr_idx, cal_idx = rest_idx[ptr_rel], rest_idx[cal_rel]
assert set(groups[ptr_idx]) & set(groups[test_idx]) == set()
assert set(groups[cal_idx]) & set(groups[test_idx]) == set()
print(f"split  proper-train={len(ptr_idx)}  calibration={len(cal_idx)}  test={len(test_idx)}")

Xtr, Xcal, Xte = X[ptr_idx], X[cal_idx], X[test_idx]
Mtr, Mcal, Mte = M[ptr_idx], M[cal_idx], M[test_idx]
Dtr, Dcal, Dte = D[ptr_idx], D[cal_idx], D[test_idx]
docs_tr = docs[ptr_idx]

# ---------------------------------------------------------------- point-estimate models (proper-train only)
mM = rf().fit(Xtr, Mtr)
mD = rf().fit(Xtr, Dtr)

eM = Mcal - mM.predict(Xcal)
eD = Dcal - mD.predict(Xcal)
print(f"calib resid: M mean={eM.mean():.3f} sd={eM.std():.3f} | D mean={eD.mean():.3f} sd={eD.std():.3f}")

# normalized baseline: RF tree variance only, with a median ridge for stability
sMcal = tree_std(mM, Xcal)
sDcal = tree_std(mD, Xcal)
bM, bD = np.median(sMcal), np.median(sDcal)
nM = eM / (sMcal + bM)
nD = eD / (sDcal + bD)
print(f"normalized ridge beta_M={bM:.3f} beta_D={bD:.3f}")

# PACR: learn provenance-aware difficulty from proper-train source-document residuals only
doc_eM, Zdoc_M, base_doc_M = source_document_residual_features(Mtr, "M")
doc_eD, Zdoc_D, base_doc_D = source_document_residual_features(Dtr, "D")
scale_M = fit_source_scale(doc_eM, Zdoc_M, base_doc_M, "M")
scale_D = fit_source_scale(doc_eD, Zdoc_D, base_doc_D, "D")

Zcal = source_feature_matrix(Xcal, Xtr, docs_tr)
# PACR uses the same RF-normalized calibration residual pairs as the normalized
# baseline, then applies the learned provenance multiplier on the query side.
# This makes PACR a conservative re-ranking rule: it cannot shrink intervals
# relative to the RF-normalized conformal baseline for high-risk compounds.
pM = nM
pD = nD

coef_rows = []
for info in [scale_M, scale_D]:
    coef_rows.append({
        "target": info["target"],
        "feature": "model_unc_base_fixed",
        "coefficient": 1.0,
        "optimizer_success": info["success"],
        "objective": info["objective"],
    })
    coef_rows.append({
        "target": info["target"],
        "feature": "intercept_log_scale",
        "coefficient": info["theta"][0],
        "optimizer_success": info["success"],
        "objective": info["objective"],
    })
    for name, coef in zip(FEATURE_NAMES, info["theta"][1:]):
        coef_rows.append({
            "target": info["target"],
            "feature": name,
            "coefficient": coef,
            "optimizer_success": info["success"],
            "objective": info["objective"],
        })
coef_df = pd.DataFrame(coef_rows)
coef_df.to_csv(os.path.join(HERE, "ext1_pacr_scale_weights.csv"), index=False)
print("\n=== PACR learned scale weights (non-negative slopes; proper-train LODO fit) ===")
print(coef_df.pivot(index="feature", columns="target", values="coefficient").round(3))


def predict_intervals(Xq, method):
    """Return lower/upper 90% bounds for M, D, IDO1, TDO2 at query points.

    IDO1 = M + D/2 and TDO2 = M - D/2 are reconstructed by joint resampling of
    calibration residual pairs, preserving M/D residual correlation.
    """
    Mh, Dh = mM.predict(Xq), mD.predict(Xq)
    if method == "standard":
        addM = eM[None, :]
        addD = eD[None, :]
    elif method == "normalized":
        sM = (tree_std(mM, Xq) + bM)[:, None]
        sD = (tree_std(mD, Xq) + bD)[:, None]
        addM = nM[None, :] * sM
        addD = nD[None, :] * sD
    elif method == "pacr":
        Zq = source_feature_matrix(Xq, Xtr, docs_tr)
        gM = apply_source_scale(scale_M, Zq, tree_std(mM, Xq) + bM)[:, None]
        gD = apply_source_scale(scale_D, Zq, tree_std(mD, Xq) + bD)[:, None]
        addM = pM[None, :] * gM
        addD = pD[None, :] * gD
    else:
        raise ValueError(method)

    Msamp = Mh[:, None] + addM
    Dsamp = Dh[:, None] + addD
    ido1 = Msamp + Dsamp / 2.0
    tdo2 = Msamp - Dsamp / 2.0

    def qlo(a):
        return np.quantile(a, ALPHA / 2, axis=1)

    def qhi(a):
        return np.quantile(a, 1 - ALPHA / 2, axis=1)

    return {
        "M": (qlo(Msamp), qhi(Msamp)),
        "D": (qlo(Dsamp), qhi(Dsamp)),
        "IDO1": (qlo(ido1), qhi(ido1)),
        "TDO2": (qlo(tdo2), qhi(tdo2)),
        "Mh": Mh,
        "Dh": Dh,
        "ido1_lo": np.quantile(ido1, ALPHA / 2, axis=1),
        "tdo2_lo": np.quantile(tdo2, ALPHA / 2, axis=1),
    }


# ---------------------------------------------------------------- evaluate on TEST fold
te_ido1_obs = Mte + Dte / 2.0
te_tdo2_obs = Mte - Dte / 2.0
adist, docdist = domain_features(Xte, Xtr, docs_tr)

rows = []
width_by_method = {}
for method in METHODS:
    iv = predict_intervals(Xte, method)
    for tgt, obs in [("IDO1", te_ido1_obs), ("TDO2", te_tdo2_obs), ("M", Mte), ("D", Dte)]:
        lo, hi = iv[tgt]
        rows.append({
            "method": method,
            "target": tgt,
            "bin_type": "ALL",
            "bin": "ALL",
            "coverage": ((obs >= lo) & (obs <= hi)).mean(),
            "mean_width": (hi - lo).mean(),
            "n": len(obs),
        })

    lo, hi = iv["IDO1"]
    width = hi - lo
    err = np.abs(te_ido1_obs - (iv["Mh"] + iv["Dh"] / 2.0))
    width_by_method[method] = (adist, docdist, width, err)

    for axis, values in [("AD", adist), ("DOC", docdist)]:
        bins = qbin(values)
        for b in ["Q1_near", "Q2", "Q3", "Q4_far"]:
            msk = bins == b
            rows.append({
                "method": method,
                "target": "IDO1",
                "bin_type": axis,
                "bin": b,
                "coverage": ((te_ido1_obs[msk] >= lo[msk]) & (te_ido1_obs[msk] <= hi[msk])).mean(),
                "mean_width": width[msk].mean(),
                "n": int(msk.sum()),
            })

res = pd.DataFrame(rows)
res.to_csv(os.path.join(HERE, "ext1_conformal_comparison.csv"), index=False)

print("\n=== marginal coverage (test fold, 90% nominal) ===")
print(res[res.bin_type == "ALL"].pivot(index="target", columns="method", values="coverage").round(3))
for axis, title in [("AD", "AD-distance"), ("DOC", "document-domain distance")]:
    piv = res[(res.target == "IDO1") & (res.bin_type == axis)].pivot(
        index="bin", columns="method", values="coverage"
    ).round(3)
    pivw = res[(res.target == "IDO1") & (res.bin_type == axis)].pivot(
        index="bin", columns="method", values="mean_width"
    ).round(3)
    print(f"\n=== IDO1 conditional coverage by {title} quartile ===")
    print(piv)
    print(f"\n=== IDO1 mean interval width by {title} quartile ===")
    print(pivw)
    print(
        "conditional-coverage MAD (lower=better): "
        + "  ".join(f"{m}={coverage_mad(res, m, axis):.3f}" for m in METHODS)
    )

for method in METHODS:
    _, _, w, err = width_by_method[method]
    print(f"adaptivity corr(width, |error|)  {method:10s}= {np.corrcoef(w, err)[0, 1]:+.3f}")


# ---------------------------------------------------------------- deck re-ranking
deck = pd.read_csv(os.path.join(DATA, "dual_conformal_ranked.csv")).dropna(subset=["clean_smiles"]).reset_index(drop=True)
Xd, keepd = fps(deck["clean_smiles"].tolist())
deck = deck[keepd].reset_index(drop=True)

for method in METHODS:
    iv = predict_intervals(Xd, method)
    deck[f"LCB_dual_{method}"] = np.minimum(iv["ido1_lo"], iv["tdo2_lo"])

deck["AD_dist_to_paired"] = 1.0 - bulk_tanimoto_max(Xd, X)[0]
deck["PACR_chem_dist_to_ptr"], deck["PACR_doc_dist_to_ptr"] = domain_features(Xd, Xtr, docs_tr)

deck.sort_values("LCB_dual_normalized", ascending=False).to_csv(
    os.path.join(HERE, "ext1_deck_normalized.csv"), index=False
)
deck.sort_values("LCB_dual_pacr", ascending=False).to_csv(
    os.path.join(HERE, "ext1_deck_pacr.csv"), index=False
)

print(f"\n=== deck (n={len(deck)}) confident-dual bound at pActivity>={THR} ===")
for method in METHODS:
    n_clear = int((deck[f"LCB_dual_{method}"] >= THR).sum())
    max_lcb = deck[f"LCB_dual_{method}"].max()
    print(f"drugs clearing LCB_dual_{method:10s}: {n_clear:4d}  max={max_lcb:.2f}")


# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
ax = ax.ravel()

for method in METHODS:
    d, _, w, _ = width_by_method[method]
    ax[0].scatter(d, w, s=10, alpha=0.35, color=COLORS[method], label=method)
ax[0].set_xlabel("AD distance to proper-train (1 - max Tanimoto)")
ax[0].set_ylabel("90% IDO1 interval width (log units)")
ax[0].set_title("(A) Interval width vs chemical domain distance")
ax[0].legend(frameon=False, fontsize=8)

xb = np.arange(4)
binlab = ["Q1\nnear", "Q2", "Q3", "Q4\nfar"]
for method in METHODS:
    sub = res[(res.target == "IDO1") & (res.bin_type == "AD")].pivot(
        index="bin", columns="method", values="coverage"
    )
    yv = [sub.loc[b, method] for b in ["Q1_near", "Q2", "Q3", "Q4_far"]]
    ax[1].plot(xb, yv, "o-", color=COLORS[method], label=method)
ax[1].axhline(1 - ALPHA, ls="--", color="k", lw=1, label="nominal 0.90")
ax[1].set_xticks(xb)
ax[1].set_xticklabels(binlab)
ax[1].set_ylim(0.5, 1.02)
ax[1].set_ylabel("empirical coverage")
ax[1].set_xlabel("test compounds by AD distance")
ax[1].set_title("(B) Coverage vs chemical distance")
ax[1].legend(frameon=False, fontsize=8)

for method in METHODS:
    sub = res[(res.target == "IDO1") & (res.bin_type == "DOC")].pivot(
        index="bin", columns="method", values="coverage"
    )
    yv = [sub.loc[b, method] for b in ["Q1_near", "Q2", "Q3", "Q4_far"]]
    ax[2].plot(xb, yv, "o-", color=COLORS[method], label=method)
ax[2].axhline(1 - ALPHA, ls="--", color="k", lw=1)
ax[2].set_xticks(xb)
ax[2].set_xticklabels(binlab)
ax[2].set_ylim(0.5, 1.02)
ax[2].set_ylabel("empirical coverage")
ax[2].set_xlabel("test compounds by document-domain distance")
ax[2].set_title("(C) Coverage vs source-document domain")

sc = ax[3].scatter(
    deck["LCB_dual_normalized"],
    deck["LCB_dual_pacr"],
    c=deck["PACR_doc_dist_to_ptr"],
    s=8,
    cmap="viridis",
    alpha=0.6,
)
lim = [
    min(deck["LCB_dual_normalized"].min(), deck["LCB_dual_pacr"].min()) - 0.2,
    max(THR + 0.15, deck["LCB_dual_normalized"].max(), deck["LCB_dual_pacr"].max()) + 0.2,
]
ax[3].plot(lim, lim, "k-", lw=0.8)
ax[3].axhline(THR, ls="--", color="r", lw=1)
ax[3].axvline(THR, ls="--", color="r", lw=1)
ax[3].set_xlim(lim)
ax[3].set_ylim(lim)
ax[3].set_xlabel("LCB_dual (RF-normalized)")
ax[3].set_ylabel("LCB_dual (PACR)")
ax[3].set_title("(D) Deck re-ranking under provenance-aware intervals")
fig.colorbar(sc, ax=ax[3], label="document-domain distance")

fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "figure_adaptive_conformal.png"), dpi=150)
print(
    "\nwrote figures/figure_adaptive_conformal.png, "
    "extensions/ext1_conformal_comparison.csv, "
    "extensions/ext1_deck_normalized.csv, extensions/ext1_deck_pacr.csv, "
    "extensions/ext1_pacr_scale_weights.csv"
)
