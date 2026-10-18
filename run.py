"""run.py -- reproduces the final AIcrowd submission of team F1-Cardiaque.

    python run.py                    # inference: load models/gbm_final.npz, predict (minutes)
    python run.py --mode train       # train the 4 models from scratch (hours), then predict
    python run.py --mode train --configs d5 v2ctrl   # train only some members
    python run.py --mode train --retrain             # ignore the cached folds

Both modes write submissions/final_submission.csv, and give the same file.

Model
-----
Equal-weight average of four gradient-boosted tree ensembles. The boosting code is
written in NumPy (research/gbm.py): histogram trees, log-loss, Newton gain, missing
values sent to the better side. Each member is trained with stratified 5-fold
cross-validation (seed 42, research/common.py):

* the out-of-fold (OOF) scores of the 5 fold models give a score for every training row;
* the test score is the mean of the 5 fold models.

The logits of the four members are averaged. The decision threshold is the one that
maximises F1 on the averaged OOF scores (one sort over all rows, research/common.py),
and it is applied to the averaged test scores.

Training saves the 20 fold models (4 members x 5 folds), the binning fitted on the
training rows and the threshold in models/gbm_final.npz. Inference only reads
x_test.csv and that file.

Data preparation
----------------
The trees work on the raw BRFSS columns, cut into 63 quantile bins (bin 0 = missing),
see fit_binning in research/common.py. Identifiers, dates and survey-design columns
are removed (research/wide.py, DROP). The members marked "v2" also recode the "none"
answers (88 / 888 / 8888) to 0 and add ALCDAY5 in days per month.

The codebook rules (don't know / refused codes, "none" answers, unit conversions,
imputation, one-hot encoding) used by our linear models, including the six methods
of implementations.py, are in preprocess.py, completed by clinical_fe.py and
interactions.py and assembled by pipeline.py. The cross-validated comparison of all
the models we tried is in research/ (see README.md).

Runtime of the training
-----------------------
About 15 000 trees in total: a few hours on a laptop CPU. The scores and trees of
each fold are cached in data/cache/run/ as soon as the fold is trained, so an
interrupted run resumes where it stopped.
"""

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "research"))

from common import (  # noqa: E402
    DATA,
    apply_binning,
    best_thr,
    fit_binning,
    folds,
    load_bins,
    load_raw,
    report,
)
from gbm import GBM  # noqa: E402
from helpers import create_csv_submission  # noqa: E402

# The four members of the final average (the same as in research/README.md).
CONFIGS = [
    {"tag": "d5", "bins": "v1", "trees": 500, "lr": 0.05, "seed": 0},
    {"tag": "v2ctrl", "bins": "v2", "trees": 500, "lr": 0.05, "seed": 0},
    {"tag": "v2s1", "bins": "v2", "trees": 1000, "lr": 0.03, "seed": 1},
    {"tag": "v2s2", "bins": "v2", "trees": 1000, "lr": 0.03, "seed": 2},
]
# Shared by all members: tree depth, L2 on the leaves, minimum hessian per child,
# and the share of rows / columns drawn at random for each tree.
TREE = {"depth": 5, "lam": 1.0, "min_hess": 20.0, "row_sub": 0.5, "col_sub": 0.5}
K_FOLDS = 5

ZIP = ROOT / "data" / "dataset_to_release.zip"
SCORES = ROOT / "data" / "cache" / "run"
MODELS = ROOT / "models" / "gbm_final.npz"
OUT = ROOT / "submissions" / "final_submission.csv"


def ensure_data():
    """Unzip data/dataset_to_release.zip into data/dataset_unzipped/ if needed."""
    if (DATA / "x_test.csv").exists():
        return
    if not ZIP.exists():
        sys.exit(f"put dataset_to_release.zip in {ZIP.parent} (or the CSVs in {DATA})")
    print(f"unzipping {ZIP.name} ...", flush=True)
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(DATA.parent)


def write_submission(ids, test, thr):
    """Predicts 1 when the averaged test logit reaches the threshold, else -1."""
    pred = np.where(test >= thr, 1, -1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    create_csv_submission(ids, pred, OUT)
    print(
        f"wrote {OUT.relative_to(ROOT)}: threshold {thr:.4f}, "
        f"{100 * (pred == 1).mean():.2f} % predicted positive"
    )


# ===================================================================== training
def member_name(cfg):
    return f"{cfg['tag']}_{cfg['bins']}_t{cfg['trees']}_s{cfg['seed']}"


def cross_validated_gbm(cfg, fl, retrain):
    """Trains the 5 fold models of one member; each fold is cached as soon as it is done.

    Returns the OOF logits, the test logits (mean of the 5 fold models) and the
    fold models as arrays (GBM.to_arrays).
    """
    t0 = time.time()
    SCORES.mkdir(parents=True, exist_ok=True)
    B, BT, y, _ = load_bins(cfg["bins"])
    oof, test, models = np.zeros(len(y)), np.zeros(len(BT)), []
    for i, (tr, va) in enumerate(fl):
        path = SCORES / f"{member_name(cfg)}_fold{i}.npz"
        if path.exists() and not retrain:
            arrays = dict(np.load(path))
            scores = arrays.pop("scores")
        else:
            model = GBM(
                cfg["trees"],
                TREE["depth"],
                cfg["lr"],
                lam=TREE["lam"],
                min_hess=TREE["min_hess"],
                row_sub=TREE["row_sub"],
                col_sub=TREE["col_sub"],
                seed=42 + i + 1000 * cfg["seed"],
            )
            # the held-out fold and the test set are scored while the trees are grown
            model.fit(
                B[tr], y[tr], B_eval=np.vstack([B[va], BT]), checkpoints={cfg["trees"]}
            )
            scores = model.snapshots[cfg["trees"]]
            arrays = model.to_arrays()
            np.savez(path, scores=scores, **arrays)
        oof[va] = scores[: len(va)]
        test += scores[len(va) :] / len(fl)
        models.append(arrays)
        print(f"  {cfg['tag']}: fold {i + 1}/{len(fl)}  {time.time() - t0:.0f}s")
    return oof, test, models


def train(args):
    """Cross-validated training of the four members, then threshold and saved models."""
    _, _, y, ids = load_bins("v2")
    fl = folds(y, k=K_FOLDS)
    print(
        f"train {len(y)} rows ({100 * y.mean():.2f} % positive), test {len(ids)} rows"
    )

    chosen = [c for c in CONFIGS if not args.configs or c["tag"] in args.configs]
    members = {}
    for cfg in chosen:
        print(f"member {cfg}", flush=True)
        members[cfg["tag"]] = cross_validated_gbm(cfg, fl, args.retrain)
        report(cfg["tag"], y, members[cfg["tag"]][0], fl)
    if len(members) < len(CONFIGS):
        print("not all members trained: no model file and no submission written")
        return

    # equal-weight average of the four members' logits
    oof = np.mean([members[c["tag"]][0] for c in CONFIGS], axis=0)
    test = np.mean([members[c["tag"]][1] for c in CONFIGS], axis=0)
    report("average of 4 GBM", y, oof, fl)
    thr, _ = best_thr(y, oof)

    # everything inference needs: binning fitted on the training rows, trees, threshold
    X, _, _, _, hdr = load_raw()
    saved = {"configs": np.array(json.dumps(CONFIGS)), "threshold": np.array(thr)}
    for version in sorted({c["bins"] for c in CONFIGS}):
        spec = fit_binning(X, hdr, version)
        saved[f"{version}__columns"] = np.array(spec["columns"])
        saved[f"{version}__edges"] = np.concatenate(spec["edges"])
        saved[f"{version}__edge_counts"] = np.array([len(e) for e in spec["edges"]])
        if "none_codes" in spec:
            saved[f"{version}__none_codes"] = spec["none_codes"]
    for cfg in CONFIGS:
        for i, arrays in enumerate(members[cfg["tag"]][2]):
            for key, value in arrays.items():
                saved[f"{cfg['tag']}__fold{i}__{key}"] = value
    MODELS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MODELS, **saved)
    print(f"saved the models in {MODELS.relative_to(ROOT)}")

    write_submission(ids, test, thr)


# ==================================================================== inference
def load_test():
    """x_test.csv as a float matrix (NaN for blanks), its column names and the ids."""
    hdr = (DATA / "x_test.csv").open().readline().strip().split(",")
    XT = np.genfromtxt(DATA / "x_test.csv", delimiter=",", skip_header=1)
    return XT[:, 1:], hdr[1:], XT[:, 0].astype(np.int64)


def inference():
    """Predicts the test set with the saved fold models (no training data needed)."""
    if not MODELS.exists():
        sys.exit(
            f"{MODELS.relative_to(ROOT)} not found: run  python run.py --mode train"
        )
    saved = np.load(MODELS)
    configs = json.loads(str(saved["configs"]))
    XT, hdr, ids = load_test()
    print(f"test {len(ids)} rows, models from {MODELS.relative_to(ROOT)}", flush=True)

    binned = {}
    for version in sorted({c["bins"] for c in configs}):
        counts = saved[f"{version}__edge_counts"]
        spec = {
            "version": version,
            "columns": [str(c) for c in saved[f"{version}__columns"]],
            "edges": np.split(saved[f"{version}__edges"], np.cumsum(counts)[:-1]),
        }
        if f"{version}__none_codes" in saved.files:
            spec["none_codes"] = saved[f"{version}__none_codes"]
        binned[version] = apply_binning(XT, hdr, spec)

    # the same operations, in the same order, as during training: identical scores
    per_member = []
    for cfg in configs:
        test = np.zeros(len(ids))
        for i in range(K_FOLDS):
            prefix = f"{cfg['tag']}__fold{i}__"
            arrays = {
                k[len(prefix) :]: saved[k] for k in saved.files if k.startswith(prefix)
            }
            test += GBM.from_arrays(arrays).decision(binned[cfg["bins"]]) / K_FOLDS
        per_member.append(test)
        print(f"  {cfg['tag']}: predicted", flush=True)
    write_submission(ids, np.mean(per_member, axis=0), float(saved["threshold"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["inference", "train"], default="inference")
    ap.add_argument("--configs", nargs="*", help="train mode: members to train")
    ap.add_argument("--retrain", action="store_true", help="train mode: ignore cache")
    args = ap.parse_args()

    t0 = time.time()
    ensure_data()
    if args.mode == "train":
        train(args)
    else:
        inference()
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
