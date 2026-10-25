# ANCHOR-IBD federated learning proof of concept

Three hospitals train one model together without sharing patient data and without a central party that holds the model. Each hospital runs its own NVIDIA DGX Spark and keeps its registry extract there in OMOP CDM format. The hospitals pass model weights among themselves, and a different hospital averages them each round (NVIDIA FLARE swarm learning). A coordinator only admits participants and starts jobs. A classic FedAvg mode, where the server aggregates, is included for comparison.

**[SPARKS.md](SPARKS.md)** explains how the Sparks run it, what is trained, what you need, and how to install and run it.

The PoC follows the ANCHOR-IBD project description (Vinnova 2026-00990) and the `ibd-fl-dgx-spark` deck:

| Project description | In this PoC |
| --- | --- |
| SWIBREG and NorIBD, Sweden and Norway | Sites `rost` (Region Östergötland, SWIBREG), `oru` (Örebro University Hospital, SWIBREG), `ous` (Oslo University Hospital, NorIBD) |
| No patient data; synthetic examples only (WP2) | Synthetic cohorts with a deliberately different case mix per site |
| OMOP CDM, FHIR R4 data model alignment | Every site stores the standard OMOP tables; one feature extractor runs unchanged everywhere |
| Treatment response prediction for therapy selection | Steroid-free remission at week 52 on the index biologic (infliximab, adalimumab, vedolizumab, ustekinumab) |
| Sex-disaggregated analysis as a core requirement | Every metric is reported for women and men separately, and calibration is reported by sex |
| GDPR and EHDS; analysis without centralising data | Only weights cross the network, over mutual TLS. Optional clipping and Gaussian noise on each update |
| Deck: Sparks as independent sites, not a cluster | One NVFlare client per Spark; decentralised swarm learning with a rotating aggregator; single-Spark compose setup as step 1 |

## Results

Synthetic data, 30 rounds of 3 local epochs, mean of 5 training seeds. Every strategy picks its stopping point on a validation split. Scores are measured on 3,000 new synthetic patients per site who took no part in training.

| AUROC | Östergötland | Örebro | Oslo | All sites |
| --- | --- | --- | --- | --- |
| Local only (each site alone) | 0.703 | 0.739 | 0.699 | 0.714 |
| **Federated (FedAvg)** | **0.720** | **0.747** | **0.704** | **0.728** |
| Pooled in one place (ceiling, not allowed in reality) | 0.729 | 0.752 | 0.704 | 0.734 |

- Federated learning beats local-only training at every site and recovers about 70% of the gap to pooling all data. No patient record leaves a site. The improvement is modest because the synthetic cohorts are small and the model is simple. That is expected and shows honestly what the PoC can demonstrate.
- By sex (all sites): the federated model scores 0.740 for women and 0.716 for men.
- Privacy versus accuracy, measured as federated AUROC: 0.728 without noise; 0.727, 0.711 and 0.687 at noise multipliers 0.05, 0.1 and 0.2. At 0.1 and above, noise costs more than federation gains. No formal ε is computed; a privacy accountant belongs in the follow-on project.
- Each update is 4.6 KB per site per round (1,185 weights).

Open `dashboard/index.html` for the interactive version, which includes a training replay, a cross-site matrix, calibration by sex and the privacy trade-off.

## What was run and what was not

| Piece | Status |
| --- | --- |
| `simulate.py` (pure PyTorch, all strategies) | Ran here; produced the numbers above |
| `nvflare_job/job.py --env sim`, swarm and FedAvg (NVFlare 2.9 simulator, 3 clients) | Ran here in WSL Ubuntu, with and without DP noise |
| Provisioned deployment: `nvflare provision`, mTLS certificates, coordinator and 3 clients as separate processes, job submitted by `admin@rise.se` | Ran here in WSL on one host: 5 rounds each of swarm and FedAvg. In swarm, the aggregator rotated between the hospitals and the coordinator did no aggregation |
| `deploy/Dockerfile`, `deploy/docker-compose.yml`, `deploy/run_site.sh` | Written but **not run**; Docker was unavailable on the build machine |
| Real DGX Sparks | Not run; no hardware access. The NGC base image tag in the Dockerfile must be checked for DGX Spark support |

OMOP source codes (ICD-10, ATC, LOINC) are real. Check standard concept IDs in Athena before using them against a real CDM; see `ibdfl/vocab.py`.

## Run it

Use Python 3.10+ on Linux or WSL (NVFlare does not run natively on Windows).

```bash
pip install -r requirements.txt
python generate_data.py --out data                       # 3 sites, OMOP CSVs
python generate_data.py --out data_eval --eval_n 3000    # held-out evaluation cohorts (simulation only)
python simulate.py                                       # all strategies, ~2 min on CPU; writes dashboard/results.js
python -m http.server 8000 --directory dashboard         # open http://localhost:8000
```

The same training code under NVFlare, three simulated hospitals in one process:

```bash
python nvflare_job/job.py --workflow swarm --env sim --data_root $PWD/data --rounds 30
```

`simulate.py` reports FedAvg numbers. Swarm computes the same weighted average; it only moves the averaging step from the server to a rotating hospital.

### On DGX Sparks

**Step 1: one Spark, four containers.** This step follows the deck's "simulate 3 hospital sites on a single Spark first". It uses the same certificates and job as step 2.

```bash
nvflare provision -p deploy/project.yml -w deploy/workspace
python generate_data.py --out data
KIT=deploy/workspace/anchor_ibd/prod_00 docker compose -f deploy/docker-compose.yml up -d --build
```

**Step 2: one Spark per hospital.** Copy each hospital only its own startup kit from `deploy/workspace/anchor_ibd/prod_00/<site>`; the kit contains that site's private key. Make `anchor-fl-server` resolve to the server host. The hospital needs only outbound TCP 8002 to the server.

```bash
docker build -f deploy/Dockerfile -t anchor-ibd-fl .                     # on every host
deploy/run_site.sh server deploy/workspace/anchor_ibd/prod_00/anchor-fl-server  # server host
deploy/run_site.sh client <kit dir for this site> /data/omop <server ip>        # each hospital Spark
```

**Submit the job** from the admin's machine:

```bash
python nvflare_job/job.py --workflow swarm --env prod --startup_kit deploy/workspace/anchor_ibd/prod_00/admin@rise.se --data_root /data
```

The full step-by-step guide is in [SPARKS.md](SPARKS.md).

Add `--dp_noise 0.05` to clip each update and add Gaussian noise before it leaves the hospital.

## Layout

| Path | What it is |
| --- | --- |
| `ibdfl/synth.py` | Synthetic cohort generator and ground-truth response model (drug × phenotype and drug × sex effects) |
| `ibdfl/vocab.py` | OMOP concepts and source codes |
| `ibdfl/features.py` | OMOP tables → 19 features; the same code at every site |
| `ibdfl/model.py`, `ibdfl/train.py` | MLP, local training, metrics (AUROC, Brier, by sex), update clipping and noise |
| `simulate.py` | Local vs FedAvg vs FedAvg+DP vs pooled, in one process |
| `nvflare_job/client.py`, `nvflare_job/job.py` | NVFlare 2.9 client script; swarm (default) and FedAvg recipes, sim or prod |
| `deploy/` | `project.yml` for provisioning, Dockerfile, single-Spark compose, per-site launcher |
| `dashboard/` | Static demo page; `results.js` is written by `simulate.py` |
| `demo/run_demo.sh` | One-command live demo: certificates, coordinator and 3 hospital clients as separate processes, swarm job, narrated output (Linux or WSL) |
| `demo/presentation.html` | 18-slide animated presentation; open in a browser, arrow keys to move, N for speaker notes, F for full screen |

`data/`, `data_eval/`, `results/` and `deploy/workspace/` (which holds private keys) are git-ignored.

## Next steps toward the follow-on project

- Replace the synthetic generator with a site ETL from SWIBREG and NorIBD exports into OMOP; the feature code stays the same.
- Agree on the outcome definition and the variable list, including sex-disaggregated variables, in WP2.
- Add a privacy accountant to report ε, and consider secure aggregation or homomorphic encryption. NVFlare includes `FedAvgRecipe` HE variants.
- Run on Sparks at Region Östergötland, Örebro and OUS once governance and ethics approval allow real data.
