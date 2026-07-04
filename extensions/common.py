"""Shared helpers for the extension analyses (adaptive conformal, LODO representation
robustness, second target-pair validation).

Regenerates the exact 2048-bit radius-2 Morgan fingerprints the deployed models expect
(verified to reproduce stored pred_M/pred_D to max|Δ|=0.005). Uses the modern
MorganGenerator API to avoid RDKit deprecation spam.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

_MGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fp(smiles):
    """SMILES -> 2048-bit ECFP4 numpy int8 array, or None if unparseable."""
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return None
    bv = _MGEN.GetFingerprint(m)
    a = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, a)
    return a


def fps(smiles_list):
    """List of SMILES -> (X matrix, keep_mask) dropping unparseable entries."""
    out, keep = [], []
    for s in smiles_list:
        f = fp(s)
        keep.append(f is not None)
        if f is not None:
            out.append(f)
    return np.asarray(out), np.asarray(keep)


def bulk_tanimoto_max(query_fps, ref_fps):
    """For each query fingerprint (n_q, 2048), max Tanimoto similarity to any ref fp.
    Fast bit-count implementation on int8 arrays."""
    q = query_fps.astype(np.float32)
    r = ref_fps.astype(np.float32)
    inter = q @ r.T                      # (n_q, n_r) shared bits
    qsum = q.sum(1, keepdims=True)       # popcount per query
    rsum = r.sum(1, keepdims=True).T     # popcount per ref
    union = qsum + rsum - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        tan = np.where(union > 0, inter / union, 0.0)
    return tan.max(1), tan


def murcko_scaffold(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        return ""
