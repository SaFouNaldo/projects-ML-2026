"""Preprocessing v1 / v2, NumPy only. Fit on train, apply the same statistics to test."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
DATA = _ROOT / "data" / "dataset_unzipped" / "dataset"
FEATURES_PATH = _ROOT / "notebooks" / "feature_selection" / "selected_features.txt"

NONE_TO_ZERO = {"PHYSHLTH", "MENTHLTH", "CHILDREN", "POORHLTH"}
LOG1P = {"_FRUTSUM", "_VEGESUM"}

# Missing indicators (0/1): only where a missing answer is informative (skip / module).
# No indicator for age, SEX, etc.: a missing value there is noise, not a profile.
MISSING_FLAG_COLS = {
    "BPMEDS",  # only asked if hypertensive
    "PREDIAB1",  # module / non-diabetics only
    "POORHLTH",  # often linked to PHYSHLTH/MENTHLTH
    "CIMEMLOS",  # cognitive module
    "SCNTMNY1",  # money module (often skipped)
    "SCNTMEL1",
    "SXORIENT",  # often skipped depending on questionnaire / state
    "TRNSGNDR",
}

# 1 = yes, 2 = no -> encoded as 1 / 0
BINARY_YESNO = {
    "_DRDXAR1",
    "_RFBING5",
    "_RFHLTH",
    "HLTHPLN1",
    "_RFHYPE5",
    "_RFCHOL",
    "MEDCOST",
    "CHCSCNCR",
    "CHCOCNCR",
    "CHCCOPD1",
    "ADDEPEV2",
    "CHCKIDNY",
    "VETERAN3",
    "INTERNET",
    "QLACTLM2",
    "USEEQUIP",
    "BLIND",
    "DECIDE",
    "DIFFWALK",
    "DIFFDRES",
    "DIFFALON",
    "_RFDRHV5",
    "CIMEMLOS",
    "BPMEDS",
}

# 7 and 9 = don't know / refused
DK_79 = BINARY_YESNO | {
    "_PACAT1",
    "_ASTHMS1",
    "_SMOKER3",
    "_CHOLCHK",
    "PERSDOC2",
    "CHECKUP1",
    "DIABETE3",
    "GENHLTH",
    "PREDIAB1",
}

# 7 is a real answer here (retired, etc.)
DK_9 = {
    "_EDUCAG",
    "_INCOMG",
    "EMPLOY1",
    "MARITAL",
    "RENTHOM1",
    "USENOW3",
    "SXORIENT",
    "TRNSGNDR",
    "SCNTMNY1",
    "SCNTMEL1",
    "_RACE",
}

DK_7799 = {"PHYSHLTH", "MENTHLTH", "CHILDREN", "POORHLTH"}

CONTINUOUS = {"_AGE80", "_BMI5", "_FRUTSUM", "_VEGESUM", "ALCDAY5"}
COUNT = {"PHYSHLTH", "MENTHLTH", "CHILDREN", "POORHLTH"}
ORDINAL = {
    "_PACAT1",
    "_ASTHMS1",
    "_SMOKER3",
    "_CHOLCHK",
    "_EDUCAG",
    "_INCOMG",
    "CHECKUP1",
    "SCNTMNY1",
    "SCNTMEL1",
    "GENHLTH",
}
NOMINAL = {
    "_RACE",
    "MARITAL",
    "EMPLOY1",
    "PERSDOC2",
    "DIABETE3",
    "RENTHOM1",
    "USENOW3",
    "SXORIENT",
    "TRNSGNDR",
    "PREDIAB1",
}


def load_feature_names(path: Path = FEATURES_PATH) -> list[str]:
    """Names of the selected raw columns, one per line in `path` (selected_features.txt)."""
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def load_selected(x_path: Path, names: list[str]) -> np.ndarray:
    """CSV -> array of strings, only the columns in `names`."""
    with x_path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = [header.index(n) for n in names]
        rows = [[row[j] for j in idx] for row in reader]
    return np.array(rows, dtype=object)


def load_y(y_path: Path) -> np.ndarray:
    """Labels of y_train.csv (second column) as floats in {-1, 1}."""
    with y_path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader)
        return np.array([float(row[1]) for row in reader])


def encode_y(y: np.ndarray) -> np.ndarray:
    """Competition labels -1/1 -> logistic 0/1."""
    return (y == 1).astype(np.float64)


def decode_y(y01: np.ndarray) -> np.ndarray:
    """0/1 -> submission labels -1/1."""
    return np.where(y01 >= 0.5, 1.0, -1.0)


def _is_dk(name: str, x: float) -> bool:
    """True if `x` is a "don't know" / "refused" code for column `name`.

    The code depends on the question: 7 / 9, 77 / 99, or 9 / 77 / 99 (see DK_79,
    DK_7799 and DK_9 above).
    """
    if name in CONTINUOUS:
        return False
    if name == "_AGE80":
        return False
    if name in DK_7799 and x in (77, 99):
        return True
    if name in DK_79 and x in (7, 9):
        return True
    if name in DK_9 and x in (9, 77, 99):
        return True
    return False


def recode_alcday5(x: float) -> float:
    """101-199 = days per week, 201-299 = days per month, 888 = 0, 777/999 = missing."""
    if x == 888:
        return 0.0
    if x in (777, 999):
        return np.nan
    if 101 <= x <= 199:
        return (x - 100) * (30.0 / 7.0)
    if 201 <= x <= 299:
        return float(x - 200)
    return np.nan


def recode_cell(name: str, raw: str) -> float:
    """One raw CSV cell -> number.

    Blank or "don't know" / "refused" -> NaN, "none" (88) -> 0 for the day counts,
    ALCDAY5 -> drinking days per month.
    """
    if raw is None or str(raw).strip() == "":
        return np.nan
    try:
        x = float(raw)
    except ValueError:
        return np.nan
    if name == "ALCDAY5":
        return recode_alcday5(x)
    if name in NONE_TO_ZERO and x == 88:
        return 0.0
    if _is_dk(name, x):
        return np.nan
    return x


def recode_matrix(raw: np.ndarray, names: list[str]) -> np.ndarray:
    """Applies recode_cell to every cell of the string matrix `raw` (columns `names`)."""
    n, d = raw.shape
    out = np.empty((n, d), dtype=np.float64)
    for j, name in enumerate(names):
        col = raw[:, j]
        out[:, j] = np.array([recode_cell(name, v) for v in col], dtype=np.float64)
    return out


class Preprocessor:
    """Cleaning and encoding of the selected columns, fitted on the training rows.

    fit_transform learns the imputation values, the one-hot levels and the
    standardisation on the training rows; transform applies exactly the same
    operations to new rows (validation or test).
    """

    def __init__(self, names: list[str], add_bias: bool = True):
        """names: raw columns to encode; add_bias: prepend a column of ones."""
        self.names = names
        self.add_bias = add_bias
        self.medians_: dict[str, float] = {}
        self.modes_: dict[str, float] = {}
        self.onehot_levels_: dict[str, np.ndarray] = {}
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.out_names_: list[str] = []

    def _impute_col(self, name: str, col: np.ndarray, fit: bool) -> np.ndarray:
        """Fills the missing values of one column.

        Mode for nominal, yes/no and SEX columns, median otherwise; the value is learned
        when `fit` is True and reused otherwise.
        """
        miss = ~np.isfinite(col)
        if name in NOMINAL or name in BINARY_YESNO or name == "SEX":
            if fit:
                finite = col[~miss]
                self.modes_[name] = (
                    float(np.bincount(finite.astype(int)).argmax())
                    if len(finite)
                    else 0.0
                )
            fill = self.modes_[name]
        else:
            if fit:
                finite = col[~miss]
                self.medians_[name] = float(np.median(finite)) if len(finite) else 0.0
            fill = self.medians_[name]
        col = col.copy()
        col[miss] = fill
        return col

    def _missing_flags(self, X: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """0/1 before imputation, only for the MISSING_FLAG_COLS present in `names`."""
        flags = []
        names_out = []
        for j, name in enumerate(self.names):
            if name not in MISSING_FLAG_COLS:
                continue
            flags.append((~np.isfinite(X[:, j])).astype(np.float64))
            names_out.append(f"{name}__isna")
        if not flags:
            return np.zeros((X.shape[0], 0), dtype=np.float64), []
        return np.column_stack(flags), names_out

    def _to_numeric_cols(
        self, X: np.ndarray, fit: bool
    ) -> tuple[np.ndarray, list[str]]:
        """Imputed columns -> numeric features, with their names.

        Yes/no columns become 0 / 1, SEX a female indicator, nominal columns one-hot
        (levels seen in the training rows), fruit and vegetable sums log(1 + x); the
        other columns are kept as they are.
        """
        parts = []
        names_out = []
        n = X.shape[0]
        for j, name in enumerate(self.names):
            col = self._impute_col(name, X[:, j], fit=fit)
            if name in BINARY_YESNO:
                parts.append((col == 1).astype(np.float64))
                names_out.append(name)
            elif name == "SEX":
                parts.append((col == 2).astype(np.float64))  # 1 = female
                names_out.append("SEX_female")
            elif name in NOMINAL:
                if fit:
                    levels = np.unique(col)
                    self.onehot_levels_[name] = levels
                levels = self.onehot_levels_[name]
                oh = np.zeros((n, len(levels)), dtype=np.float64)
                for k, lv in enumerate(levels):
                    oh[:, k] = (col == lv).astype(np.float64)
                parts.append(oh)
                names_out.extend(f"{name}={lv:g}" for lv in levels)
            else:
                if name in LOG1P:
                    col = np.log1p(np.clip(col, 0.0, None))
                parts.append(col)
                names_out.append(name)
        return np.column_stack(parts), names_out

    def fit_transform(self, raw: np.ndarray) -> np.ndarray:
        """Fits on the training rows and returns their encoded, standardised matrix.

        Missing indicators are added for MISSING_FLAG_COLS, and the bias column comes
        first when add_bias is True.
        """
        X = recode_matrix(raw, self.names)
        miss_Z, miss_names = self._missing_flags(X)  # before imputation
        Z, names_out = self._to_numeric_cols(X, fit=True)
        if miss_names:
            Z = np.c_[Z, miss_Z]
            names_out = names_out + miss_names
        self.mean_ = Z.mean(axis=0)
        self.std_ = Z.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Z = (Z - self.mean_) / self.std_
        if self.add_bias:
            Z = np.c_[np.ones(Z.shape[0]), Z]
            names_out = ["bias"] + names_out
        self.out_names_ = names_out
        return Z

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Encodes new rows with the statistics learned by fit_transform."""
        X = recode_matrix(raw, self.names)
        miss_Z, miss_names = self._missing_flags(X)
        Z, _ = self._to_numeric_cols(X, fit=False)
        if miss_names:
            Z = np.c_[Z, miss_Z]
        Z = (Z - self.mean_) / self.std_
        if self.add_bias:
            Z = np.c_[np.ones(Z.shape[0]), Z]
        return Z


def build_train_test(
    x_train: Path = DATA / "x_train.csv",
    y_train: Path = DATA / "y_train.csv",
    x_test: Path = DATA / "x_test.csv",
    features_path: Path = FEATURES_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Preprocessor]:
    """Reads the CSV files, keeps the selected columns and fits the Preprocessor on train.

    Returns (tx_train, y in {0, 1}, tx_test, preprocessor).
    """
    names = load_feature_names(features_path)
    raw_tr = load_selected(x_train, names)
    raw_te = load_selected(x_test, names)
    y = encode_y(load_y(y_train))
    pp = Preprocessor(names)
    tx = pp.fit_transform(raw_tr)
    tx_te = pp.transform(raw_te)
    return tx, y, tx_te, pp
