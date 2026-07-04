"""PACR sensitivity and ablation analysis.

This script stress-tests Extension 1 by varying scaffold-disjoint split seeds,
the PACR tail quantile, and the provenance features used in the learned scale.

Outputs:
  ext1_pacr_sensitivity.csv
  ext1_pacr_sensitivity_summary.csv
  ../figures/figure_pacr_sensitivity.png
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

ALPHA = 0.10
THR = 6.0  # pActivity threshold (pIC50-equivalent <=1 uM)
EPS = 1e-6
RNG_MODEL = 0
N_TREES = 300
SPLIT_SEEDS = [0, 1, 2, 3, 4]
TAUS = [0.65, 0.70, 0.75, 0.80]
DOC_TOP_K = 5
FEATURE_NAMES = ["chem_dist", "doc_dist"]
FEATURE_SETS = {
    "chem": [0],
    "doc": [1],
    "chem_doc": [0, 1],
}
PLOT_METHODS = ["standard", "normalized", "pacr_chem", "pacr_doc", "pacr_chem_doc"]
COLORS = {
    "standard": "#888888",
    "normalized": "#1b7837",
    "pacr_chem": "#d95f02",
    "pacr_doc": "#7570b3",
    "pacr_chem_doc": "#2166ac",
}


def rf():
    return RandomForestRegressor(
        n_estimators=N_TREES,
        random_state=RNG_MODEL,
        n_jobs=-1,
        bootstrap=True,
    )


def tree_std(model, Xq):
    preds = np.stack([t.predict(Xq) for t in model.estimators_], axis=0)
    return preds.std(0)


def tanimoto_matrix(Xq, ref_X):
    _, tan = bulk_tanimoto_max(Xq, ref_X)
    return tan


def document_domain_distance(tan, ref_docs, k=DOC_TOP_K):
    """Best per-document mean of top min(k, n_doc) Tanimoto similarities."""
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


def domain_features(Xq, ref_X, ref_docs):
    tan = tanimoto_matrix(Xq, ref_X)
    chem_dist = 1.0 - tan.max(axis=1)
    doc_dist = document_domain_distance(tan, ref_docs)
    return np.column_stack([chem_dist, doc_dist])


def fit_feature_scaler(Z):
    lo = np.nanpercentile(Z, 5, axis=0)
    hi = np.nanpercentile(Z, 95, axis=0)
    span = np.where((hi - lo) > EPS, hi - lo, 1.0)
    return {"lo": lo, "span": span}


def transform_features(Z, scaler):
    return np.clip((Z - scaler["lo"]) / scaler["span"], 0.0, 1.5)


def fit_source_scale(residual, Zraw, base_scale, tau, cols):
    residual = np.asarray(residual, dtype=float)
    Zraw = np.asarray(Zraw, dtype=float)[:, cols]
    base_scale = np.asarray(base_scale, dtype=float)
    ok = np.isfinite(residual) & np.isfinite(base_scale) & np.all(np.isfinite(Zraw), axis=1)
    y = np.abs(residual[ok]) + EPS
    base = base_scale[ok] + EPS
    scaler = fit_feature_scaler(Zraw[ok])
    Z = transform_features(Zraw[ok], scaler)
    ratio = y / base

    init = np.r_[max(0.0, np.log(np.median(ratio))), np.zeros(Z.shape[1])]
    bounds = [(0.0, 4.0)] + [(0.0, 10.0)] * Z.shape[1]

    def objective(theta):
        pred = np.exp(theta[0]) * (1.0 + Z @ theta[1:]) + EPS
        diff = ratio - pred
        pinball = np.where(diff >= 0, tau * diff, (tau - 1.0) * diff)
        return float(np.mean(pinball) + 0.01 * np.sum(theta[1:] ** 2))

    opt = minimize(objective, init, method="L-BFGS-B", bounds=bounds)
    theta = opt.x if opt.success else init
    return {
        "theta": theta,
        "scaler": scaler,
        "cols": cols,
        "success": bool(opt.success),
        "objective": float(opt.fun if opt.success else objective(theta)),
    }


def apply_source_scale(info, Zraw, base_scale):
    Z = transform_features(np.asarray(Zraw, dtype=float)[:, info["cols"]], info["scaler"])
    multiplier = np.exp(info["theta"][0]) * (1.0 + Z @ info["theta"][1:])
    return np.asarray(base_scale, dtype=float) * multiplier + EPS


def source_document_residual_features(Xtr, y, docs_tr):
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
        Zraw[held] = domain_features(Xtr[held], Xtr[train], docs_tr[train])
    return residual, Zraw, base


def qbin(values):
    labels = ["Q1_near", "Q2", "Q3", "Q4_far"]
    ranks = pd.Series(values).rank(method="first")
    return np.asarray(pd.qcut(ranks, 4, labels=labels))


def build_intervals(Mh, Dh, addM, addD):
    Msamp = Mh[:, None] + addM
    Dsamp = Dh[:, None] + addD
    ido1 = Msamp + Dsamp / 2.0
    tdo2 = Msamp - Dsamp / 2.0
    return {
        "M": (np.quantile(Msamp, ALPHA / 2, axis=1), np.quantile(Msamp, 1 - ALPHA / 2, axis=1)),
        "D": (np.quantile(Dsamp, ALPHA / 2, axis=1), np.quantile(Dsamp, 1 - ALPHA / 2, axis=1)),
        "IDO1": (np.quantile(ido1, ALPHA / 2, axis=1), np.quantile(ido1, 1 - ALPHA / 2, axis=1)),
        "TDO2": (np.quantile(tdo2, ALPHA / 2, axis=1), np.quantile(tdo2, 1 - ALPHA / 2, axis=1)),
        "ido1_lo": np.quantile(ido1, ALPHA / 2, axis=1),
        "tdo2_lo": np.quantile(tdo2, ALPHA / 2, axis=1),
    }


def conditional_coverage(lo, hi, obs, values):
    bins = qbin(values)
    coverages, widths = [], []
    for b in ["Q1_near", "Q2", "Q3", "Q4_far"]:
        msk = bins == b
        coverages.append(float(((obs[msk] >= lo[msk]) & (obs[msk] <= hi[msk])).mean()))
        widths.append(float((hi[msk] - lo[msk]).mean()))
    return {
        "q1_coverage": coverages[0],
        "q4_coverage": coverages[-1],
        "coverage_mad": float(np.mean(np.abs(np.asarray(coverages) - (1 - ALPHA)))),
        "q1_width": widths[0],
        "q4_width": widths[-1],
    }


def summarize_method(row_meta, iv_test, obs, pred_ido1, adist, docdist, iv_deck, deck):
    out = dict(row_meta)
    for target, y in [("IDO1", obs["ido1"]), ("TDO2", obs["tdo2"]), ("M", obs["M"]), ("D", obs["D"])]:
        lo, hi = iv_test[target]
        out[f"coverage_{target}"] = float(((y >= lo) & (y <= hi)).mean())
        out[f"width_{target}"] = float((hi - lo).mean())
    lo, hi = iv_test["IDO1"]
    ad = conditional_coverage(lo, hi, obs["ido1"], adist)
    doc = conditional_coverage(lo, hi, obs["ido1"], docdist)
    out.update({
        "ad_q1_coverage_IDO1": ad["q1_coverage"],
        "ad_q4_coverage_IDO1": ad["q4_coverage"],
        "ad_coverage_mad_IDO1": ad["coverage_mad"],
        "ad_q1_width_IDO1": ad["q1_width"],
        "ad_q4_width_IDO1": ad["q4_width"],
        "doc_q1_coverage_IDO1": doc["q1_coverage"],
        "doc_q4_coverage_IDO1": doc["q4_coverage"],
        "doc_coverage_mad_IDO1": doc["coverage_mad"],
        "doc_q1_width_IDO1": doc["q1_width"],
        "doc_q4_width_IDO1": doc["q4_width"],
        "width_error_corr_IDO1": float(np.corrcoef(hi - lo, np.abs(obs["ido1"] - pred_ido1))[0, 1]),
    })
    lcb_dual = np.minimum(iv_deck["ido1_lo"], iv_deck["tdo2_lo"])
    out["deck_max_LCB_dual"] = float(lcb_dual.max())
    out["deck_n_clear_dual"] = int((lcb_dual >= THR).sum())
    top_idx = int(np.argmax(lcb_dual))
    out["deck_top_pref_name"] = deck["pref_name"].iat[top_idx] if "pref_name" in deck else ""
    return out


def split_indices(X, M, groups, seed):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.41, random_state=seed)
    rest_idx, test_idx = next(gss1.split(X, M, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.45, random_state=seed)
    ptr_rel, cal_rel = next(gss2.split(X[rest_idx], M[rest_idx], groups[rest_idx]))
    return rest_idx[ptr_rel], rest_idx[cal_rel], test_idx


def evaluate_seed(seed, X, M, D, groups, docs, deck, Xd):
    ptr_idx, cal_idx, test_idx = split_indices(X, M, groups, seed)
    Xtr, Xcal, Xte = X[ptr_idx], X[cal_idx], X[test_idx]
    Mtr, Mcal, Mte = M[ptr_idx], M[cal_idx], M[test_idx]
    Dtr, Dcal, Dte = D[ptr_idx], D[cal_idx], D[test_idx]
    docs_tr = docs[ptr_idx]

    mM = rf().fit(Xtr, Mtr)
    mD = rf().fit(Xtr, Dtr)
    eM = Mcal - mM.predict(Xcal)
    eD = Dcal - mD.predict(Xcal)
    sMcal = tree_std(mM, Xcal)
    sDcal = tree_std(mD, Xcal)
    bM, bD = np.median(sMcal), np.median(sDcal)
    nM = eM / (sMcal + bM)
    nD = eD / (sDcal + bD)

    doc_eM, Zdoc_M, base_doc_M = source_document_residual_features(Xtr, Mtr, docs_tr)
    doc_eD, Zdoc_D, base_doc_D = source_document_residual_features(Xtr, Dtr, docs_tr)

    Mh_te, Dh_te = mM.predict(Xte), mD.predict(Xte)
    Mh_d, Dh_d = mM.predict(Xd), mD.predict(Xd)
    baseM_te, baseD_te = tree_std(mM, Xte) + bM, tree_std(mD, Xte) + bD
    baseM_d, baseD_d = tree_std(mM, Xd) + bM, tree_std(mD, Xd) + bD
    Zte = domain_features(Xte, Xtr, docs_tr)
    Zd = domain_features(Xd, Xtr, docs_tr)
    adist, docdist = Zte[:, 0], Zte[:, 1]

    obs = {
        "M": Mte,
        "D": Dte,
        "ido1": Mte + Dte / 2.0,
        "tdo2": Mte - Dte / 2.0,
    }
    pred_ido1 = Mh_te + Dh_te / 2.0
    rows = []

    def add_row(method, tau, feature_set, addM_te, addD_te, addM_d, addD_d, theta_M="", theta_D=""):
        iv_test = build_intervals(Mh_te, Dh_te, addM_te, addD_te)
        iv_deck = build_intervals(Mh_d, Dh_d, addM_d, addD_d)
        meta = {
            "split_seed": seed,
            "method": method,
            "tau": tau,
            "feature_set": feature_set,
            "n_proper_train": len(ptr_idx),
            "n_calibration": len(cal_idx),
            "n_test": len(test_idx),
            "theta_M": theta_M,
            "theta_D": theta_D,
        }
        rows.append(summarize_method(meta, iv_test, obs, pred_ido1, adist, docdist, iv_deck, deck))

    add_row("standard", np.nan, "none", eM[None, :], eD[None, :], eM[None, :], eD[None, :])
    add_row(
        "normalized",
        np.nan,
        "model_unc",
        nM[None, :] * baseM_te[:, None],
        nD[None, :] * baseD_te[:, None],
        nM[None, :] * baseM_d[:, None],
        nD[None, :] * baseD_d[:, None],
    )

    for tau in TAUS:
        for feature_set, cols in FEATURE_SETS.items():
            scale_M = fit_source_scale(doc_eM, Zdoc_M, base_doc_M, tau, cols)
            scale_D = fit_source_scale(doc_eD, Zdoc_D, base_doc_D, tau, cols)
            gM_te = apply_source_scale(scale_M, Zte, baseM_te)
            gD_te = apply_source_scale(scale_D, Zte, baseD_te)
            gM_d = apply_source_scale(scale_M, Zd, baseM_d)
            gD_d = apply_source_scale(scale_D, Zd, baseD_d)
            theta_M = ";".join(f"{x:.4g}" for x in scale_M["theta"])
            theta_D = ";".join(f"{x:.4g}" for x in scale_D["theta"])
            add_row(
                f"pacr_{feature_set}",
                tau,
                feature_set,
                nM[None, :] * gM_te[:, None],
                nD[None, :] * gD_te[:, None],
                nM[None, :] * gM_d[:, None],
                nD[None, :] * gD_d[:, None],
                theta_M,
                theta_D,
            )

    print(f"seed={seed} done: train/cal/test={len(ptr_idx)}/{len(cal_idx)}/{len(test_idx)}")
    return rows


def plot_summary(summary):
    tau75 = summary[(summary["tau_label"] == "0.75") | (summary["method"].isin(["standard", "normalized"]))].copy()
    tau75["plot_label"] = tau75["method"].map({
        "standard": "standard",
        "normalized": "RF-norm",
        "pacr_chem": "PACR chem",
        "pacr_doc": "PACR doc",
        "pacr_chem_doc": "PACR chem+doc",
    })
    tau75 = tau75.set_index("method").loc[PLOT_METHODS].reset_index()

    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    ax = ax.ravel()
    x = np.arange(len(tau75))
    colors = [COLORS[m] for m in tau75["method"]]

    ax[0].bar(x, tau75["coverage_IDO1_mean"], yerr=tau75["coverage_IDO1_std"], color=colors, alpha=0.85)
    ax[0].axhline(0.90, color="k", ls="--", lw=1)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(tau75["plot_label"], rotation=25, ha="right")
    ax[0].set_ylim(0.72, 1.02)
    ax[0].set_ylabel("mean IDO1 coverage")
    ax[0].set_title("(A) Marginal coverage across split seeds")

    ax[1].bar(x, tau75["doc_q4_coverage_IDO1_mean"], yerr=tau75["doc_q4_coverage_IDO1_std"], color=colors, alpha=0.85)
    ax[1].axhline(0.90, color="k", ls="--", lw=1)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(tau75["plot_label"], rotation=25, ha="right")
    ax[1].set_ylim(0.55, 1.02)
    ax[1].set_ylabel("Q4 document-domain coverage")
    ax[1].set_title("(B) Far source-domain coverage")

    ax[2].bar(x, tau75["deck_max_LCB_dual_mean"], yerr=tau75["deck_max_LCB_dual_std"], color=colors, alpha=0.85)
    ax[2].axhline(THR, color="r", ls="--", lw=1)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(tau75["plot_label"], rotation=25, ha="right")
    ax[2].set_ylabel("max deck LCB_dual")
    ax[2].set_title("(C) Honest-negative bound")

    full = summary[summary["method"] == "pacr_chem_doc"].sort_values("tau")
    ax[3].errorbar(full["tau"], full["coverage_IDO1_mean"], yerr=full["coverage_IDO1_std"],
                   marker="o", color=COLORS["pacr_chem_doc"], label="IDO1 coverage")
    ax[3].errorbar(full["tau"], full["doc_q4_coverage_IDO1_mean"], yerr=full["doc_q4_coverage_IDO1_std"],
                   marker="s", color="#542788", label="Q4 doc coverage")
    ax[3].axhline(0.90, color="k", ls="--", lw=1)
    ax[3].set_ylim(0.70, 1.02)
    ax[3].set_xlabel("PACR tail quantile tau")
    ax[3].set_ylabel("coverage")
    ax[3].set_title("(D) Tau sensitivity for PACR chem+doc")
    ax[3].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "figure_pacr_sensitivity.png"), dpi=150)


def main():
    paired = pd.read_csv(os.path.join(DATA, "paired_MD_dataset.csv"))
    paired = paired.dropna(subset=["clean_smiles", "M", "D", "scaffold", "src_doc"]).reset_index(drop=True)
    X, keep = fps(paired["clean_smiles"].tolist())
    paired = paired[keep].reset_index(drop=True)
    M = paired["M"].to_numpy(float)
    D = paired["D"].to_numpy(float)
    groups = paired["scaffold"].to_numpy()
    docs = paired["src_doc"].to_numpy()

    deck = pd.read_csv(os.path.join(DATA, "dual_conformal_ranked.csv")).dropna(subset=["clean_smiles"]).reset_index(drop=True)
    Xd, keepd = fps(deck["clean_smiles"].tolist())
    deck = deck[keepd].reset_index(drop=True)

    print(f"paired={len(paired)} deck={len(deck)} trees={N_TREES} seeds={SPLIT_SEEDS}")
    rows = []
    for seed in SPLIT_SEEDS:
        rows.extend(evaluate_seed(seed, X, M, D, groups, docs, deck, Xd))

    out = pd.DataFrame(rows)
    out["tau_label"] = out["tau"].map(lambda x: "baseline" if pd.isna(x) else f"{x:.2f}")
    out.to_csv(os.path.join(HERE, "ext1_pacr_sensitivity.csv"), index=False)

    metric_cols = [
        c for c in out.columns
        if c.startswith(("coverage_", "width_", "ad_", "doc_", "deck_", "width_error_"))
        and c not in {"deck_top_pref_name"}
    ]
    summary = (
        out.groupby(["method", "feature_set", "tau_label", "tau"], dropna=False)[metric_cols]
        .agg(["mean", "std", "min", "max"])
    )
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(HERE, "ext1_pacr_sensitivity_summary.csv"), index=False)
    plot_summary(summary)

    view = summary[(summary["tau_label"] == "0.75") | (summary["method"].isin(["standard", "normalized"]))]
    cols = [
        "method", "tau_label", "coverage_IDO1_mean", "doc_q4_coverage_IDO1_mean",
        "width_error_corr_IDO1_mean", "deck_max_LCB_dual_mean", "deck_n_clear_dual_max",
    ]
    print("\n=== sensitivity summary (tau=0.75 for PACR; mean over seeds) ===")
    print(view[cols].sort_values(["method", "tau_label"]).round(3).to_string(index=False))
    print(
        "\nwrote ext1_pacr_sensitivity.csv, ext1_pacr_sensitivity_summary.csv, "
        "../figures/figure_pacr_sensitivity.png"
    )


if __name__ == "__main__":
    main()
