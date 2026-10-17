"""Extras cliniques sans redondance avec le preprocess de base.

Pipeline retenu
---------------
Base (54 brutes → preprocess → ~103 cols)
+ NO_DUP_EXTRAS (7 cols) :
  - stroke, insulin : info absente de selected_features
  - hta_untreated, diab_x_female, risk_x_care, high_risk_low_care,
    silent_highrisk_phenotype : interactions / phénotypes
    (ascvd et care sont calculés en interne, pas ajoutés comme colonnes)

Ne pas empiler ascvd_points / care_access_score / metabolic_* sur la base :
ils résument des variables déjà présentes.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# Colonnes brutes nécessaires pour construire les 7 extras
REQUIRED_COLS = [
    "_AGE80",
    "SEX",
    "_BMI5",
    "_SMOKER3",
    "_RFHYPE5",
    "BPHIGH4",
    "BPMEDS",
    "_RFCHOL",
    "TOLDHI2",
    "CHOLCHK",
    "DIABETE3",
    "INSULIN",
    "_TOTINDA",
    "CVDSTRK3",
    "GENHLTH",
    "DIFFWALK",
    "HLTHPLN1",
    "PERSDOC2",
    "MEDCOST",
    "CHECKUP1",
]

# Version figée (celle qui a marché en CV ~0.424)
NO_DUP_EXTRAS = [
    "stroke",
    "insulin",
    "hta_untreated",
    "diab_x_female",
    "risk_x_care",
    "high_risk_low_care",
    "silent_highrisk_phenotype",
]


def _parse(s: str) -> float:
    s = s.strip()
    if not s:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_raw_matrix(x_path: Path, colnames: list[str]) -> tuple[np.ndarray, list[str]]:
    with x_path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = [header.index(c) for c in colnames if c in header]
        used = [header[j] for j in idx]
        rows = [[_parse(row[j]) for j in idx] for row in reader]
    return np.asarray(rows, dtype=np.float64), used


def _yesno(v: np.ndarray, yes: float = 1.0, no: float = 2.0) -> np.ndarray:
    out = np.full(v.shape, np.nan, dtype=np.float64)
    out[v == yes] = 1.0
    out[v == no] = 0.0
    return out


def build_nodup_extras(X: np.ndarray, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """Construit uniquement les 7 extras NO_DUP (peut contenir NaN)."""
    ix = {n: i for i, n in enumerate(cols)}

    def c(n: str) -> np.ndarray:
        return X[:, ix[n]]

    n = len(X)
    age = c("_AGE80")
    sex_female = (c("SEX") == 2).astype(np.float64)
    sex_male = 1.0 - sex_female

    bmi_raw = c("_BMI5")
    bmi = np.where(np.nanmedian(bmi_raw) > 100, bmi_raw / 100.0, bmi_raw)
    obese = (bmi >= 30).astype(np.float64)

    sm = c("_SMOKER3")
    smoke_current = np.isin(sm, [1, 2]).astype(np.float64)
    smoke_former = (sm == 3).astype(np.float64)

    rfh = _yesno(c("_RFHYPE5"))
    bph = c("BPHIGH4")
    hta = np.full(n, np.nan)
    hta[bph == 1] = 1.0
    hta[bph == 2] = 0.5
    hta[bph == 3] = 0.0
    hta[bph == 4] = 0.0
    hta = np.where(np.isnan(rfh), hta, rfh)
    bpmeds = _yesno(c("BPMEDS"))
    hta_untreated = ((np.nan_to_num(hta) >= 1) & (bpmeds == 0)).astype(np.float64)

    chol = _yesno(c("_RFCHOL"))
    told = _yesno(c("TOLDHI2"))
    high_chol = np.where(np.isnan(chol), told, chol)

    di = c("DIABETE3")
    diab = (di == 1).astype(np.float64)
    prediab = (di == 4).astype(np.float64)
    diab_x_female = diab * sex_female
    insulin = _yesno(c("INSULIN"))

    totinda = _yesno(c("_TOTINDA"))
    inactive = 1.0 - np.nan_to_num(totinda, nan=0.5)

    stroke = _yesno(c("CVDSTRK3"))
    gh = c("GENHLTH")
    diffwalk = _yesno(c("DIFFWALK"))

    # Scores internes (non exportés) pour risk_x_care / high_risk_low_care
    ascvd = (
        0.07 * np.nan_to_num(age, nan=55.0)
        + 2.0 * sex_male
        + 2.5 * smoke_current
        + 1.0 * smoke_former
        + 2.2 * np.nan_to_num(hta)
        + 1.5 * np.nan_to_num(high_chol)
        + 3.0 * diab
        + 1.0 * prediab
        + 1.2 * obese
        + 0.8 * inactive
    )
    has_doc = np.isin(c("PERSDOC2"), [1, 2]).astype(np.float64)
    recent_checkup = np.isin(c("CHECKUP1"), [1, 2]).astype(np.float64)
    insured = _yesno(c("HLTHPLN1"))
    cost_barrier = _yesno(c("MEDCOST"))
    cholchk = c("CHOLCHK")
    chol_checked_recent = np.isin(cholchk, [1, 2]).astype(np.float64)
    care = (
        np.nan_to_num(insured)
        + has_doc
        + recent_checkup
        + chol_checked_recent
        - np.nan_to_num(cost_barrier)
    )

    risk_x_care = ascvd * np.maximum(care, 0.0)
    high_risk_low_care = (
        (ascvd >= np.nanpercentile(ascvd, 75)) & (care <= 1.0)
    ).astype(np.float64)
    silent_highrisk_phenotype = (
        (age >= 60).astype(np.float64)
        * sex_male
        * np.maximum(smoke_current, np.nan_to_num(hta))
        * (gh <= 2).astype(np.float64)
        * (1.0 - np.nan_to_num(diffwalk))
    )

    feats = {
        "stroke": stroke,
        "insulin": insulin,
        "hta_untreated": hta_untreated,
        "diab_x_female": diab_x_female,
        "risk_x_care": risk_x_care,
        "high_risk_low_care": high_risk_low_care,
        "silent_highrisk_phenotype": silent_highrisk_phenotype,
    }
    names = list(NO_DUP_EXTRAS)
    mat = np.column_stack([feats[n] for n in names])
    return mat, names


class ClinicalFE:
    """Fit train (médiane / mean / std) puis transform — extras NO_DUP uniquement."""

    def __init__(self, add_bias: bool = False):
        self.add_bias = add_bias
        self.colnames_: list[str] | None = None
        self.feat_names_: list[str] = list(NO_DUP_EXTRAS)
        self.median_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x_path: Path) -> "ClinicalFE":
        raw, used = load_raw_matrix(x_path, REQUIRED_COLS)
        self.colnames_ = used
        Z, names = build_nodup_extras(raw, used)
        self.feat_names_ = names
        self.median_ = np.nanmedian(Z, axis=0)
        Z = Z.copy()
        inds = np.where(np.isnan(Z))
        Z[inds] = np.take(self.median_, inds[1])
        self.mean_ = Z.mean(axis=0)
        self.std_ = Z.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, x_path: Path) -> np.ndarray:
        assert self.colnames_ is not None and self.mean_ is not None
        raw, used = load_raw_matrix(x_path, self.colnames_)
        if used != self.colnames_:
            ix = {n: i for i, n in enumerate(used)}
            raw = np.column_stack([raw[:, ix[n]] for n in self.colnames_])
        Z, _ = build_nodup_extras(raw, self.colnames_)
        Z = Z.copy()
        inds = np.where(np.isnan(Z))
        Z[inds] = np.take(self.median_, inds[1])
        Z = (Z - self.mean_) / self.std_
        if self.add_bias:
            Z = np.c_[np.ones(len(Z)), Z]
        return Z

    def fit_transform(self, x_path: Path) -> np.ndarray:
        self.fit(x_path)
        return self.transform(x_path)
