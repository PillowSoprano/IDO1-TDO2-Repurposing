"""Shared cheminformatics utilities for the reconstructed pipeline.

Deterministic given RDKit; every downstream stage validates its output against the
committed reference CSVs, so these definitions are pinned by that agreement.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdFingerprintGenerator, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import FilterCatalog

_LFC = rdMolStandardize.LargestFragmentChooser()
_MGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# PAINS A/B/C catalogue
_pp = FilterCatalog.FilterCatalogParams()
for _c in (FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A,
           FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B,
           FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C):
    _pp.AddCatalog(_c)
_PAINS = FilterCatalog.FilterCatalog(_pp)


def clean_smiles(smi):
    """Canonical SMILES of the largest organic fragment, or None."""
    if not isinstance(smi, str):
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(_LFC.choose(m))
    except Exception:
        return None


def morgan_fp(smiles):
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return None
    a = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(_MGEN.GetFingerprint(m), a)
    return a


def morgan_matrix(smiles_list):
    out, keep = [], []
    for s in smiles_list:
        f = morgan_fp(s)
        keep.append(f is not None)
        if f is not None:
            out.append(f)
    return np.asarray(out), np.asarray(keep)


def murcko_scaffold(smiles):
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        return ""


def is_pains(smiles):
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return bool(m is not None and _PAINS.HasMatch(m))


def descriptors(smiles):
    """Nine physicochemical descriptors matching *_features.csv columns."""
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return None
    return {
        "MW": Descriptors.MolWt(m),
        "LogP": Descriptors.MolLogP(m),
        "HBD": Descriptors.NumHDonors(m),
        "HBA": Descriptors.NumHAcceptors(m),
        "TPSA": Descriptors.TPSA(m),
        "RotB": Descriptors.NumRotatableBonds(m),
        "AromRings": Descriptors.NumAromaticRings(m),
        "QED": QED.qed(m),
        "HeavyAtoms": float(m.GetNumHeavyAtoms()),
    }


def bulk_tanimoto_max(query_fps, ref_fps):
    q = query_fps.astype(np.float32); r = ref_fps.astype(np.float32)
    inter = q @ r.T
    union = q.sum(1, keepdims=True) + r.sum(1, keepdims=True).T - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        tan = np.where(union > 0, inter / union, 0.0)
    return tan.max(1)
