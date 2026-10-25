# The data: where it comes from and how it is validated

**Every patient in this proof of concept is synthetic.** No SWIBREG, NorIBD, hospital or trial records were used. That follows the ANCHOR-IBD project description, which limits WP2 to regulatory frameworks, data model specifications and synthetic examples, with no ethics review needed. The synthetic data exists to exercise the pipeline realistically: OMOP CDM tables, three sites with different case mixes, federated training. It does **not** show how well biologic response can be predicted in real patients.

## 1. How the synthetic patients are made

[`ibdfl/synth.py`](ibdfl/synth.py) generates each site's cohort and writes it as OMOP CDM tables.

| Step | What happens |
| --- | --- |
| Case mix per site | Crohn's disease (CD) or ulcerative colitis (UC), sex, age at diagnosis, disease duration, Montreal location or extent, smoking, prior anti-TNF, azathioprine, prednisolone |
| Inflammation | A hidden inflammation level per patient drives CRP, faecal calprotectin, albumin and haemoglobin, so the labs are correlated as in real patients |
| Biologic | Drawn from each site's mix; anti-TNF-experienced patients are usually switched to vedolizumab (UC) or ustekinumab (CD) |
| Outcome | Clinical remission at week 52 on that biologic, drawn from a ground-truth model (below) |
| Output | `person`, `condition_occurrence`, `drug_exposure`, `measurement` and `observation` CSV tables with real ICD-10, ATC and LOINC source codes |

**Sites**

| Site | Registry | Training cohort | Case mix |
| --- | --- | --- | --- |
| `rost` Region Östergötland | SWIBREG | 600 | Established cohort: 55% Crohn's, 40% prior anti-TNF, mostly anti-TNF |
| `oru` Örebro University Hospital | SWIBREG | 450 | Similar, more vedolizumab |
| `ous` Oslo University Hospital | NorIBD | 350 | Newly diagnosed, like the IBSEN III inception cohort: 37% Crohn's, 15% prior anti-TNF, 46% ustekinumab |

A further 3,000 synthetic patients per site, never used in training, measure every result (`generate_data.py --eval_n 3000`).

**The ground-truth outcome model.** Each patient's chance of remission comes from one formula shared by all sites. It encodes patterns reported in the IBD literature; the effect sizes are set by the PoC team, not estimated from data:

- Prior anti-TNF lowers the chance that another anti-TNF works.
- Vedolizumab does better in UC than in Crohn's.
- Ustekinumab helps anti-TNF-experienced Crohn's patients.
- Women do worse on anti-TNF, reflecting lower adherence (Salem et al., BMC Gastroenterology 2024, cited in the project description).
- Smoking worsens Crohn's outcomes. Higher CRP and calprotectin, lower albumin and longer disease duration lower the chance of remission; infliximab with azathioprine does better.

**Calibration.** Each diagnosis × drug combination has an offset, fitted by [`validation/calibrate.py`](validation/calibrate.py) and stored in [`ibdfl/calibration.json`](ibdfl/calibration.json). The offset moves that combination's week-52 remission rate onto the published figure in [`validation/targets.json`](validation/targets.json). The Oslo cohort's composition follows IBSEN III directly.

## 2. Validation against published figures

[`validation/validate.py`](validation/validate.py) reads the generated OMOP tables through the same feature code the model uses, computes each target and writes [`validation/report.md`](validation/report.md). It fails if a verified target is out of tolerance.

```bash
python generate_data.py --out data && python generate_data.py --out data_eval --eval_n 3000
python validation/validate.py
```

| Check | Published figure | Before calibration | After calibration | Status of the figure |
| --- | --- | --- | --- | --- |
| Oslo: share of Crohn's (vs UC) | 36.7%, IBSEN III (626 CD, 1,082 UC) | 45.0% ✗ | 36.0% ✓ | verified |
| Oslo UC: share of women | 45.9%, IBSEN III UC subset | 49.5% ✗ | 46.9% ✓ | provisional |
| Oslo UC: current smokers | 6.2%, IBSEN III UC subset | 10.3% ✗ | 7.1% ✓ | provisional |
| UC, vedolizumab, anti-TNF-naive: remission week 52 | 31.3%, VARSITY | 70.2% ✗ | 32.6% ✓ | verified |
| UC, adalimumab, anti-TNF-naive: remission week 52 | 22.5%, VARSITY | 43.4% ✗ | 23.9% ✓ | verified |
| UC, infliximab, anti-TNF-naive: remission week 54 | 34.7%, ACT 1 | 47.2% ✗ | 34.7% ✓ | provisional |
| Crohn's, ustekinumab, biologic-naive: remission week 52 | 65%, SEAVUE | 50.9% ✗ | 66.6% ✓ | verified |
| Crohn's, adalimumab, biologic-naive: remission week 52 | 61%, SEAVUE | 36.7% ✗ | 59.3% ✓ | verified |
| UC, ustekinumab, anti-TNF-exposed: remission | 32%, SWIBREG real-world | 52.6% ✗ | 31.1% ✓ | verified |
| Crohn's, infliximab, anti-TNF-naive | 61% (set equal to adalimumab) | 39.4% | 59.4% | assumption |
| Crohn's, vedolizumab, anti-TNF-naive | 45% | 28.3% | 43.1% | assumption |
| Swedish sites: share of Crohn's | 52% | 52.7% | 52.4% | assumption |

Before calibration every scored check failed: the first version roughly doubled the trial remission rates. After calibration all 9 scored checks pass within their tolerance (3 to 5 percentage points). The relative patterns stayed the same.

**Status of the figures**

- **Verified:** checked against the publication or its abstract.
- **Provisional:** seen only in a secondary source; verify before relying on it. The two IBSEN III UC figures come from a search-engine extract of an ECCO congress abstract whose page now returns 404. The ACT 1 week-54 remission figure comes from a review table, because the NEJM abstract reports response, not remission.
- **Assumption:** no comparable published figure found; chosen by the PoC team and reported but not scored.

**Sources**

- IBSEN III baseline: Scand J Gastroenterol 2021;56:899-905, [PMID 34154494](https://pubmed.ncbi.nlm.nih.gov/34154494/)
- VARSITY: Sands et al., N Engl J Med 2019;381:1215-1226 ([abstract](https://science.rsu.lv/en/publications/vedolizumab-versus-adalimumab-for-moderate-to-severe-ulcerative-c/))
- SEAVUE: Sands et al., Lancet 2022;399:2200-2211 ([record](https://cuk.elsevierpure.com/en/publications/ustekinumab-versus-adalimumab-for-induction-and-maintenance-thera/))
- ACT 1: Rutgeerts et al., N Engl J Med 2005, week-54 remission as tabulated in a [2015 review](https://wjgnet.com/2150-5330/tables/v6/i4/219.htm)
- SWIBREG ustekinumab in UC: United European Gastroenterol J 2022, [PMC9486503](https://pmc.ncbi.nlm.nih.gov/articles/PMC9486503)

**Limits of this validation**

- Trial populations are not registry populations. Trials select moderate-to-severe active disease and define remission by Mayo or CDAI scores; registries record routine care. The SWIBREG figure is the only real-world, Swedish target, and it is measured at a median of 32 weeks, not 52.
- The targets cover single variables and remission rates per drug. They do not check correlations between variables, which need real data or published cross-tabulations.
- Matching published averages does not make individual synthetic patients realistic.

## 3. How to validate further

| Dimension | Without real data (now) | With real data (follow-on, under governance) |
| --- | --- | --- |
| **Fidelity** (does it look real?) | Published figures, as above; clinical review of value ranges and effect directions by the project's clinicians; OHDSI Data Quality Dashboard on the OMOP tables | Distribution distances per variable, correlation differences, and a classifier that tries to tell real from synthetic |
| **Utility** (do analyses agree?) | Known effect directions reappear (they do, by construction) | Train on synthetic, test on real; compare model conclusions |
| **Privacy** (could it reveal patients?) | Not applicable: generated from rules, not from records | Distance to the closest real record, membership-inference tests, differential privacy |

## 4. Sources of real data, for later

| Source | Data | Access |
| --- | --- | --- |
| SWIBREG and NorIBD | The real registries this project targets | Through ANCHOR-IBD governance: ethics approval, DPIA, data agreements |
| [YODA Project](https://yoda.yale.edu/wp-content/uploads/2025/06/YODA-Project-Due-Diligence-Assessment-2025-0444.pdf) | Patient-level Janssen trials of infliximab and ustekinumab in UC and Crohn's | Application; analysis in a secure environment |
| Vivli, Takeda | Patient-level vedolizumab studies | Application |
| AbbVie trial data sharing | Adalimumab and other trials | Ask the AbbVie partners |
| [IBD BioResource](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6709775/), [1000IBD](https://bmcgastroenterol.biomedcentral.com/articles/10.1186/s12876-018-0917-5) | Real IBD cohorts with treatment history | Collaboration or application |
| [MIMIC-IV demo in OMOP](https://www.physionet.org/content/mimic-iv-demo-omop/) | 100 real hospital patients in OMOP (not IBD-specific) | Open download; useful for testing the OMOP loader |

No openly downloadable patient-level dataset pairs IBD biologics with one-year response.

## 5. The models trained on it

| | Proof of concept (Python, NVIDIA FLARE) | Website live demo (JavaScript) |
| --- | --- | --- |
| Model | Neural network 19 → 32 → 16 → 1 (1,185 weights), [`ibdfl/model.py`](ibdfl/model.py) | Logistic regression with 30 inputs, including drug × patient interactions |
| Inputs | Age, sex, CD or UC, disease duration, Montreal location and extent, smoking, prior anti-TNF, azathioprine, prednisolone, CRP, calprotectin, albumin, haemoglobin, biologic | The same, without age, haemoglobin and Montreal classification |
| Training | Adam, 3 local epochs per round, 30 rounds, best round chosen on validation loss | Gradient descent, 2 epochs per round, 30 rounds |
| Combining sites | Swarm learning or FedAvg, weighted by training size; optional clipping and Gaussian noise | The same idea in a few lines |

## 6. Results on the calibrated data

AUROC on 3,000 new synthetic patients per hospital, mean of 5 training runs:

| | Östergötland | Örebro | Oslo | All |
| --- | --- | --- | --- | --- |
| Each hospital alone | 0.762 | 0.741 | 0.736 | 0.743 |
| Swarm / FedAvg | 0.768 | 0.757 | 0.742 | 0.758 |
| All records copied to one place (ceiling) | 0.770 | 0.758 | 0.744 | 0.760 |

Training together recovers about 88% of the gain from pooling, and beats training alone at every hospital. For women and men the swarm model scores 0.756 and 0.756. Earlier, uncalibrated data gave 0.714, 0.728 and 0.734; those numbers are superseded.
