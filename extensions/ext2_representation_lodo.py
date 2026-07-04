"""Extension 2 — Is the leave-one-document-out (LODO) collapse the DATA or the MODEL?

The paper's central diagnostic is that predictive skill for shared potency M, healthy
under random (R2~0.64) and scaffold (~0.50) splitting, collapses to ~0 under LODO. The
obvious reviewer objection: maybe that is a weakness of RF + Morgan, and a richer
representation or learner would generalise across source documents.

We test that head-on: the SAME LODO protocol (LeaveOneGroupOut over the 44 ChEMBL source
documents) applied to a grid of representations x learners, each also evaluated under
random 5-fold and scaffold GroupKFold for reference. If every diverse method collapses
under LODO while succeeding under random/scaffold, the collapse is a property of the data
(cross-laboratory distribution shift), not of any one model.

Representations: ECFP4 (2048 bit), RDKit 2D descriptors (217), MACCS keys (167),
pretrained ChemBERTa embeddings (768, seyonec/ChemBERTa-zinc-base-v1).
Learners: RandomForest, XGBoost, Ridge.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from common import fp
from rdkit import Chem
from rdkit.Chem import MACCSkeys, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

RNG = 0
TARGET = "M"           # headline analysis on shared potency M

df = pd.read_csv("../data/paired_MD_dataset.csv").dropna(subset=["clean_smiles", "M", "D", "scaffold", "src_doc"]).reset_index(drop=True)
mols = [Chem.MolFromSmiles(s) for s in df["clean_smiles"]]
ok = np.array([m is not None for m in mols])
df, mols = df[ok].reset_index(drop=True), [m for m, k in zip(mols, ok) if k]
y = df[TARGET].to_numpy(float)
g_scaf = df["scaffold"].to_numpy()
g_doc = df["src_doc"].to_numpy()
print(f"compounds={len(df)}  documents={df['src_doc'].nunique()}  scaffolds={df['scaffold'].nunique()}")

# ---------------------------------------------------------------- representations
print("building representations ...")
X_ecfp = np.array([fp(s) for s in df["clean_smiles"]], dtype=np.float32)

_descnames = [d[0] for d in Descriptors._descList]
_calc = MoleculeDescriptors.MolecularDescriptorCalculator(_descnames)
X_desc = np.array([_calc.CalcDescriptors(m) for m in mols], dtype=np.float64)
X_desc[~np.isfinite(X_desc)] = np.nan

X_maccs = np.array([np.frombuffer(MACCSkeys.GenMACCSKeys(m).ToBitString().encode(), "u1") - ord("0")
                    for m in mols], dtype=np.float32)

def chemberta_embed(smiles, model_name="seyonec/ChemBERTa-zinc-base-v1", batch=64):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(smiles), batch):
            b = list(smiles[i:i + batch])
            enc = tok(b, padding=True, truncation=True, max_length=256, return_tensors="pt")
            h = mdl(**enc).last_hidden_state           # (B, L, 768)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1)   # mean-pool
            outs.append(emb.cpu().numpy())
    return np.vstack(outs).astype(np.float32)

reps = {"ECFP4": X_ecfp, "RDKit-desc": X_desc, "MACCS": X_maccs}
try:
    print("embedding with ChemBERTa (downloads ~150MB on first run) ...")
    reps["ChemBERTa"] = chemberta_embed(df["clean_smiles"].tolist())
    print("  ChemBERTa embedding OK:", reps["ChemBERTa"].shape)
except Exception as e:
    print("  ChemBERTa unavailable, skipping:", type(e).__name__, e)

# ---------------------------------------------------------------- learners
def make_learner(kind, sparse_ok):
    if kind == "RF":
        return RandomForestRegressor(n_estimators=300, random_state=RNG, n_jobs=-1)
    if kind == "XGB":
        return XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.6, random_state=RNG, n_jobs=-1)
    if kind == "Ridge":
        # impute + scale for descriptor/embedding representations
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(with_mean=True), Ridge(alpha=10.0))
    raise ValueError(kind)

# curated (representation, learner) combos
COMBOS = [
    ("ECFP4", "RF"), ("ECFP4", "XGB"), ("ECFP4", "Ridge"),
    ("RDKit-desc", "RF"), ("RDKit-desc", "XGB"),
    ("MACCS", "RF"),
]
if "ChemBERTa" in reps:
    COMBOS += [("ChemBERTa", "Ridge"), ("ChemBERTa", "RF")]

def pooled_r2(X, y, cv, groups=None):
    """Global R2 over pooled out-of-fold predictions (never averaged per fold)."""
    yhat = cross_val_predict(est, X, y, cv=cv, groups=groups, n_jobs=-1)
    return r2_score(y, yhat)

rows = []
for rep, learner in COMBOS:
    X = reps[rep]
    # RF/XGB tolerate NaN? RF does not -> impute for descriptor reps
    if rep == "RDKit-desc" and learner in ("RF",):
        X = SimpleImputer(strategy="median").fit_transform(X)
    est = make_learner(learner, sparse_ok=(rep in ("ECFP4", "MACCS")))
    r_random = pooled_r2(X, y, KFold(5, shuffle=True, random_state=RNG))
    r_scaf   = pooled_r2(X, y, GroupKFold(5), groups=g_scaf)
    r_lodo   = pooled_r2(X, y, LeaveOneGroupOut(), groups=g_doc)
    rows.append({"representation": rep, "learner": learner,
                 "random": r_random, "scaffold": r_scaf, "LODO": r_lodo})
    print(f"  {rep:11s}+{learner:5s}  random={r_random:+.3f}  scaffold={r_scaf:+.3f}  LODO={r_lodo:+.3f}")

res = pd.DataFrame(rows)
res.to_csv("ext2_representation_lodo.csv", index=False)
print("\n=== summary (pooled global R2 for M) ===")
print(res.round(3).to_string(index=False))
print(f"\nmax LODO R2 across ALL {len(res)} method combinations: {res['LODO'].max():.3f}")
print(f"mean random R2={res['random'].mean():.3f}  mean scaffold={res['scaffold'].mean():.3f}  mean LODO={res['LODO'].mean():.3f}")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(11, 5))
labels = [f"{r}+{l}" for r, l in zip(res.representation, res.learner)]
x = np.arange(len(labels)); w = 0.26
ax.bar(x - w, res["random"], w, label="random 5-fold", color="#4575b4")
ax.bar(x,     res["scaffold"], w, label="scaffold GroupKFold", color="#fdae61")
ax.bar(x + w, res["LODO"], w, label="leave-one-document-out", color="#d73027")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel("pooled global R²  (shared potency M)")
ax.set_title("LODO collapse is model-independent: every representation × learner fails "
             "across source documents")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("../figures/figure_representation_lodo.png", dpi=150)
print("\nwrote figures/figure_representation_lodo.png, ext2_representation_lodo.csv")
