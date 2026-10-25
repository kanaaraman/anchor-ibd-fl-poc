"""OMOP CDM -> model features. The same code runs unchanged at every site.

Normalisation uses fixed clinical reference values rather than data
statistics, so no site has to share means or variances to align features.
"""
import csv
import os
from collections import defaultdict

import numpy as np

from . import vocab as V

FEATURES = [
    "age", "female", "crohns", "duration", "ileal", "colonic", "uc_extensive",
    "smoker", "prior_anti_tnf", "immunomod", "steroid",
    "log_crp", "log_calpro", "albumin", "hb",
    "drug_infliximab", "drug_adalimumab", "drug_vedolizumab", "drug_ustekinumab",
]
N_FEATURES = len(FEATURES)

_ANTI_TNF_IDS = {d["concept_id"] for d in V.BIOLOGICS.values() if d["anti_tnf"]}
_DRUG_IDX = {V.BIOLOGICS[n]["concept_id"]: i for i, n in enumerate(V.DRUG_ORDER)}


def _read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_site(site_dir):
    """Return X (n, N_FEATURES) float32, y (n,) float32, female (n,) bool, drug (n,) int."""
    persons = _read(os.path.join(site_dir, "person.csv"))
    rec = defaultdict(dict)
    for c in _read(os.path.join(site_dir, "condition_occurrence.csv")):
        r = rec[c["person_id"]]
        r["crohns"] = int(c["condition_concept_id"]) == V.CROHNS["concept_id"]
        r["dx_year"] = int(c["condition_start_date"][:4]) + int(c["condition_start_date"][5:7]) / 12
    for d in _read(os.path.join(site_dir, "drug_exposure.csv")):
        r = rec[d["person_id"]]
        cid = int(d["drug_concept_id"])
        if d["drug_type"] == "index_biologic":
            r["drug"] = _DRUG_IDX[cid]
            r["index_year"] = int(d["drug_exposure_start_date"][:4]) + int(d["drug_exposure_start_date"][5:7]) / 12
        elif d["drug_type"] == "prior" and cid in _ANTI_TNF_IDS:
            r["prior_anti_tnf"] = 1
        elif cid == V.AZATHIOPRINE["concept_id"]:
            r["immunomod"] = 1
        elif cid == V.PREDNISOLONE["concept_id"]:
            r["steroid"] = 1
    meas = {V.CRP["concept_id"]: "crp", V.CALPROTECTIN["concept_id"]: "calpro",
            V.ALBUMIN["concept_id"]: "albumin", V.HEMOGLOBIN["concept_id"]: "hb"}
    for m in _read(os.path.join(site_dir, "measurement.csv")):
        rec[m["person_id"]][meas[int(m["measurement_concept_id"])]] = float(m["value_as_number"])
    for o in _read(os.path.join(site_dir, "observation.csv")):
        r = rec[o["person_id"]]
        cid = int(o["observation_concept_id"])
        if cid == V.MONTREAL["concept_id"]:
            r["loc"] = o["value_as_string"]
        elif cid == V.SMOKING_CURRENT["concept_id"]:
            r["smoker"] = int(o["value_as_number"])
        elif cid == V.OUTCOME_SFR52["concept_id"]:
            r["y"] = int(o["value_as_number"])

    X, y, female, drug = [], [], [], []
    for p in persons:
        r = rec[p["person_id"]]
        if "y" not in r or "drug" not in r:
            continue
        is_f = int(p["gender_concept_id"]) == V.GENDER_FEMALE
        age = r["index_year"] - int(p["year_of_birth"])
        onehot = [0.0] * 4
        onehot[r["drug"]] = 1.0
        X.append([
            (age - 40) / 15, float(is_f), float(r["crohns"]), (r["index_year"] - r["dx_year"]) / 5,
            float(r.get("loc") in ("L1", "L3")), float(r.get("loc") in ("L2", "L3")), float(r.get("loc") == "E3"),
            float(r.get("smoker", 0)), float(r.get("prior_anti_tnf", 0)), float(r.get("immunomod", 0)),
            float(r.get("steroid", 0)),
            (np.log(r["crp"]) - np.log(8)), (np.log(r["calpro"]) - np.log(600)),
            (r["albumin"] - 38) / 4, (r["hb"] - 130) / 15,
            *onehot,
        ])
        y.append(r["y"])
        female.append(is_f)
        drug.append(r["drug"])
    return (np.asarray(X, np.float32), np.asarray(y, np.float32),
            np.asarray(female, bool), np.asarray(drug, np.int64))


def split3(n, val_frac=0.15, test_frac=0.25, seed=0):
    """Deterministic train/validation/test index split, done locally at each site.

    Validation picks the stopping point; test is only used for the reported numbers.
    """
    idx = np.random.default_rng(seed).permutation(n)
    k_te, k_va = int(n * test_frac), int(n * val_frac)
    return idx[k_te + k_va:], idx[k_te:k_te + k_va], idx[:k_te]
