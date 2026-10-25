"""Synthetic IBD registry cohorts written as OMOP CDM tables.

No real patient data. Each site gets its own cohort with a deliberately
different (non-IID) case mix, mirroring the registries in ANCHOR-IBD:

  rost  Region Östergötland (SWIBREG)  established cohort, anti-TNF heavy
  oru   Örebro University Hospital (SWIBREG)  similar, slightly more vedolizumab
  ous   Oslo University Hospital (NorIBD / IBSEN III-like)  newly diagnosed,
        few anti-TNF-experienced patients, ustekinumab used far more often

The outcome (steroid-free remission at week 52 on the index biologic) comes
from one shared ground-truth model with drug x phenotype and drug x sex
interactions. Some of those interactions are rare at a single site, which is
why pooling signal across sites helps.
"""
import csv
import os
from datetime import date, timedelta

import numpy as np

from . import vocab as V

SITES = {
    "rost": dict(name="Region Östergötland", registry="SWIBREG", country="SE", n=600,
                 p_cd=0.55, dur_mean=8.0, p_prior_tnf=0.40,
                 drugs=[0.35, 0.35, 0.25, 0.05], care_site_id=101),
    "oru": dict(name="Örebro University Hospital", registry="SWIBREG", country="SE", n=450,
                p_cd=0.50, dur_mean=7.0, p_prior_tnf=0.35,
                drugs=[0.30, 0.30, 0.30, 0.10], care_site_id=102),
    "ous": dict(name="Oslo University Hospital", registry="NorIBD", country="NO", n=350,
                p_cd=0.45, dur_mean=1.5, p_prior_tnf=0.15,
                drugs=[0.25, 0.15, 0.15, 0.45], care_site_id=201),
}


def true_logit(p):
    """Ground-truth response model shared by all sites (dict of arrays)."""
    anti_tnf = p["drug"] <= 1
    vdz = p["drug"] == 2
    ust = p["drug"] == 3
    cd = p["cd"]
    z = (0.35
         - 1.1 * p["prior_tnf"] * anti_tnf
         - 0.3 * p["prior_tnf"] * ~anti_tnf
         + 1.0 * vdz * ~cd - 0.7 * vdz * cd
         + 0.2 * ust + 1.2 * ust * cd * p["prior_tnf"]
         - 0.7 * p["female"] * anti_tnf          # lower adherence to biologics in women (Salem 2024)
         - 0.6 * p["smoker"] * cd
         - 0.45 * (np.log(p["crp"]) - np.log(8))
         - 0.35 * (np.log(p["calpro"]) - np.log(600))
         + 0.07 * (p["albumin"] - 38)
         - 0.15 * p["duration"] / 5
         + 0.4 * p["immunomod"] * (p["drug"] == 0)  # infliximab + azathioprine combination
         - 0.3 * p["steroid"])
    return z


def sample_patients(site, rng, n=None):
    s = SITES[site]
    n = n or s["n"]
    p = {}
    p["female"] = rng.random(n) < 0.5
    p["cd"] = rng.random(n) < s["p_cd"]
    p["age_dx"] = np.clip(rng.normal(30, 12, n), 8, 80)
    p["duration"] = np.clip(rng.exponential(s["dur_mean"], n), 0.1, 40)
    p["age"] = p["age_dx"] + p["duration"]
    p["loc"] = np.where(p["cd"], rng.choice(["L1", "L2", "L3"], n, p=[0.35, 0.25, 0.40]),
                        rng.choice(["E1", "E2", "E3"], n, p=[0.20, 0.40, 0.40]))
    p["smoker"] = rng.random(n) < np.where(p["cd"], 0.25, 0.10)
    p["prior_tnf"] = rng.random(n) < s["p_prior_tnf"]
    p["immunomod"] = rng.random(n) < 0.40
    p["steroid"] = rng.random(n) < 0.45
    # drug choice: anti-TNF-experienced patients rarely get another anti-TNF; UC leans vedolizumab
    drug = rng.choice(4, n, p=s["drugs"])
    switch = p["prior_tnf"] & (drug <= 1) & (rng.random(n) < 0.6)
    drug[switch] = np.where(p["cd"][switch], 3, 2)
    p["drug"] = drug
    infl = rng.normal(0, 1, n) + 0.3 * p["cd"]
    p["crp"] = np.exp(np.log(8) + 0.9 * infl + rng.normal(0, 0.4, n))
    p["calpro"] = np.exp(np.log(600) + 0.7 * infl + rng.normal(0, 0.5, n))
    p["albumin"] = rng.normal(38, 3, n) - 1.5 * infl
    p["hb"] = rng.normal(135, 12, n) - 6 * infl - 10 * p["female"]
    site_shift = {"rost": 0.0, "oru": -0.1, "ous": 0.15}[site]
    p["response"] = rng.random(n) < 1 / (1 + np.exp(-(true_logit(p) + site_shift)))
    return p


def write_omop(site, out_dir, seed=0, n=None):
    """Write person, condition_occurrence, drug_exposure, measurement, observation CSVs.

    n overrides the site's cohort size (used for large held-out evaluation cohorts).
    """
    s = SITES[site]
    rng = np.random.default_rng(seed * 1000 + s["care_site_id"])
    p = sample_patients(site, rng, n)
    n = len(p["female"])
    os.makedirs(out_dir, exist_ok=True)
    pid0 = s["care_site_id"] * 100000
    rows = {k: [] for k in ["person", "condition_occurrence", "drug_exposure", "measurement", "observation"]}
    for i in range(n):
        pid = pid0 + i
        index = date(2020, 1, 1) + timedelta(days=int(rng.integers(0, 1460)))
        dx = index - timedelta(days=int(p["duration"][i] * 365.25))
        yob = index.year - int(p["age"][i])
        female = bool(p["female"][i])
        rows["person"].append(dict(person_id=pid, gender_concept_id=V.GENDER_FEMALE if female else V.GENDER_MALE,
                                   year_of_birth=yob, care_site_id=s["care_site_id"],
                                   person_source_value=f"{site}-{i:05d}", gender_source_value="F" if female else "M"))
        cond = V.CROHNS if p["cd"][i] else V.ULCERATIVE_COLITIS
        rows["condition_occurrence"].append(dict(person_id=pid, condition_concept_id=cond["concept_id"],
                                                 condition_start_date=dx, condition_source_value=cond["source"]))
        drug = V.BIOLOGICS[V.DRUG_ORDER[p["drug"][i]]]
        rows["drug_exposure"].append(dict(person_id=pid, drug_concept_id=drug["concept_id"],
                                          drug_exposure_start_date=index, drug_source_value=drug["source"],
                                          drug_type="index_biologic"))
        if p["prior_tnf"][i]:
            prior = V.BIOLOGICS["infliximab" if rng.random() < 0.5 else "adalimumab"]
            rows["drug_exposure"].append(dict(person_id=pid, drug_concept_id=prior["concept_id"],
                                              drug_exposure_start_date=index - timedelta(days=int(rng.integers(200, 1500))),
                                              drug_source_value=prior["source"], drug_type="prior"))
        for flag, d in [("immunomod", V.AZATHIOPRINE), ("steroid", V.PREDNISOLONE)]:
            if p[flag][i]:
                rows["drug_exposure"].append(dict(person_id=pid, drug_concept_id=d["concept_id"],
                                                  drug_exposure_start_date=index - timedelta(days=int(rng.integers(0, 60))),
                                                  drug_source_value=d["source"], drug_type="concomitant"))
        for key, m in [("crp", V.CRP), ("calpro", V.CALPROTECTIN), ("albumin", V.ALBUMIN), ("hb", V.HEMOGLOBIN)]:
            rows["measurement"].append(dict(person_id=pid, measurement_concept_id=m["concept_id"],
                                            measurement_date=index - timedelta(days=int(rng.integers(0, 14))),
                                            value_as_number=round(float(p[key][i]), 1),
                                            unit_source_value=m["unit"], measurement_source_value=m["source"]))
        rows["observation"] += [
            dict(person_id=pid, observation_concept_id=V.MONTREAL["concept_id"], observation_date=index,
                 value_as_number="", value_as_string=p["loc"][i], observation_source_value=V.MONTREAL["source"]),
            dict(person_id=pid, observation_concept_id=V.SMOKING_CURRENT["concept_id"], observation_date=index,
                 value_as_number=int(p["smoker"][i]), value_as_string="", observation_source_value=V.SMOKING_CURRENT["source"]),
            dict(person_id=pid, observation_concept_id=V.OUTCOME_SFR52["concept_id"],
                 observation_date=index + timedelta(days=364), value_as_number=int(p["response"][i]),
                 value_as_string="", observation_source_value=V.OUTCOME_SFR52["source"]),
        ]
    for table, recs in rows.items():
        with open(os.path.join(out_dir, f"{table}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0]))
            w.writeheader()
            w.writerows(recs)
    return {k: len(v) for k, v in rows.items()}
