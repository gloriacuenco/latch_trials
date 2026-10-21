"""Does harmonization keep the biology? Age and sex predicted from latents, before and after.

A harmonizer should remove the site but keep what differs between people. Age and sex are the
biology ABIDE records for everyone. Linear models (ridge for age, logistic for sex) on the same
latent features as the site probes (hdev/probes.py, coarse + fine) are trained on the training scans
and scored on the validation scans, for the original latents and for each harmonized set.

Age is confounded with site in ABIDE (some sites scanned only children), so a model can predict age
partly by recognizing the site. ``age_r_within_site`` removes that route: predictions and true ages
are centred per site before correlating, so only differences between people of the same site count.
Reference: the same models on FastSurfer anatomy (regional volumes, brain size) alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hdev.flow import ANATOMY_REGIONS
from hdev.latents import load_latents
from hdev.probes import coarse, fine


def within_site_r(y: np.ndarray, pred: np.ndarray, site: np.ndarray) -> float:
    yc, pc = y.astype(float).copy(), pred.astype(float).copy()
    for s in np.unique(site):
        m = site == s
        yc[m] -= yc[m].mean()
        pc[m] -= pc[m].mean()
    return float(np.corrcoef(yc, pc)[0, 1])


def score(x: np.ndarray, age: np.ndarray, sex: np.ndarray, site: np.ndarray, tr: np.ndarray, va: np.ndarray) -> dict:
    reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))).fit(x[tr], age[tr])
    pa = reg.predict(x[va])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced"))
    ps = clf.fit(x[tr], sex[tr]).predict(x[va])
    return {"age_mae": float(mean_absolute_error(age[va], pa)), "age_r": float(np.corrcoef(age[va], pa)[0, 1]),
            "age_r_within_site": within_site_r(age[va], pa, site[va]),
            "sex_ba": float(balanced_accuracy_score(sex[va], ps))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--latent-dir", required=True)
    p.add_argument("--brain-index", required=True)
    p.add_argument("--harmonized", nargs="*", default=[], help="name=latents.npz from e3_slice_export.py")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    z, meta = load_latents(args.latent_dir, splits=("train", "val"))
    brain = pd.read_csv(args.brain_index).set_index("subject_id").loc[meta["subject_id"]]
    keep = brain["age"].notna().to_numpy() & brain["sex"].notna().to_numpy()
    age, sex = brain["age"].to_numpy(float), brain["sex"].to_numpy()
    site = meta["site_id"].to_numpy()
    tr, va = (meta["split"] == "train").to_numpy() & keep, (meta["split"] == "val").to_numpy() & keep

    def feats(lat: np.ndarray) -> np.ndarray:
        return np.concatenate([coarse(lat), fine(lat)], 1)

    total = brain["vol_brain"].to_numpy(float)
    anatomy = np.stack([np.log(total)] + [brain[f"vol_{r}"].to_numpy(float) / total for r in ANATOMY_REGIONS], 1)
    res = {"n_train": int(tr.sum()), "n_val": int(va.sum()),
           "fastsurfer_anatomy": score(anatomy, age, sex, site, tr, va),
           "original": score(feats(z), age, sex, site, tr, va)}
    for spec in args.harmonized:
        name, path = spec.split("=", 1)
        with np.load(path, allow_pickle=False) as d:
            order = {s: i for i, s in enumerate(d["subject_ids"].astype(str))}
            lat = d["latents"][[order[s] for s in meta["subject_id"]]]
        res[name] = score(feats(lat), age, sex, site, tr, va)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(pd.DataFrame({k: v for k, v in res.items() if isinstance(v, dict)}).T.round(3).to_string())


if __name__ == "__main__":
    main()
