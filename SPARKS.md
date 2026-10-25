# Running ANCHOR-IBD on DGX Sparks

How to install and run the decentralised training across hospitals, one DGX Spark per hospital.

## How it runs

```
            Coordinator (any small Linux host, or a 4th Spark)
            admits participants, starts/stops jobs
            holds no data and no model weights
                 ▲             ▲             ▲
     outbound    │ mTLS 8002   │             │
     only        │             │             │
   ┌─────────────┴──┐   ┌──────┴─────────┐   ┌┴───────────────┐
   │ Spark: Region  │   │ Spark: Örebro  │   │ Spark: Oslo    │
   │ Östergötland   │◄─►│ University     │◄─►│ University     │
   │ OMOP extract   │   │ Hospital       │   │ Hospital       │
   │ (read-only)    │   │ OMOP extract   │   │ OMOP extract   │
   └────────────────┘   └────────────────┘   └────────────────┘
        model weights move between hospitals; one hospital
        aggregates each round, chosen at random
```

Each round of **swarm learning** (`--workflow swarm`, the default) works like this:

1. Every hospital trains the current model for a few epochs on its own patients, on its own Spark's GPU.
2. One hospital is picked as that round's aggregator. The others send it their updated weights, encrypted for that hospital only.
3. The aggregator computes the weighted average, which becomes the new shared model. The model goes back to every hospital and the next round starts with a different aggregator.
4. At the end, each hospital keeps the best shared model, chosen by validation loss.

The coordinator never receives, stores or averages model weights. It checks certificates, admits the three hospitals and starts and stops jobs. In NVFlare's default network setup, hospital-to-hospital messages are routed through the coordinator's message relay, end-to-end encrypted so the relay cannot read them. Each hospital then needs only an outbound connection. Direct hospital-to-hospital links are possible, but each hospital would have to open an inbound port; that is a decision for the hospitals' IT and security teams, and it has not been tested here.

The classic setup (`--workflow fedavg`) is also available. In it the server is the aggregator.

### What each Spark does

| | |
| --- | --- |
| Runs | One Docker container with the NVFlare client and the training script |
| Reads | Its own OMOP CDM extract, mounted read-only |
| Computes | Local training and evaluation on the GB10 GPU, and averaging when it is the aggregator |
| Sends | Model weights (4.6 KB per round for this model) and summary metrics such as AUROC and validation loss. Never patient records |
| Network | Outbound TCP 8002 to the coordinator. No inbound ports |
| Identity | Its own certificate and private key from the startup kit (mutual TLS) |

The current model is small (1,185 weights) and trains in seconds even on a CPU. This PoC does not need a Spark's compute. The Spark matters because it is a standard, hospital-owned GPU node that sits next to the data. The same setup can later train much larger models, such as endoscopy image models for Mayo or UCEIS scoring, or language models on clinical notes, which do need the GPU and the 128 GB of memory.

## What is trained

| | |
| --- | --- |
| Question | Will this patient reach steroid-free clinical remission 52 weeks after starting this biologic? |
| Biologics | Infliximab, adalimumab, vedolizumab, ustekinumab |
| Inputs (19) | Age, sex, Crohn's disease or ulcerative colitis, disease duration, Montreal location or extent, smoking, prior anti-TNF, azathioprine, prednisolone, CRP, faecal calprotectin, albumin, haemoglobin, and the biologic started |
| Output | A probability of response |
| Model | Small neural network (19 → 32 → 16 → 1) |
| Reported | AUROC and calibration per site, overall, and separately for women and men |

Because the drug is an input, the model can compare predicted response across the four biologics for the same patient. That is the therapy-selection question in the project description. It is a research model trained on synthetic data and must not be used for clinical decisions.

## What you need

**Hardware**
- One DGX Spark per hospital: 3 for Region Östergötland, Örebro and OUS.
- One coordinator host: any Linux machine or VM with Docker and about 2 CPU cores, reachable from all hospitals. A 4th Spark also works, but no GPU is needed.
- An admin laptop for RISE to submit jobs. It can also be the coordinator.

**Network**
- A DNS name for the coordinator, for example `anchor-fl.rise.se`, reachable from every hospital Spark.
- Inbound TCP 8002 (training traffic) and 8003 (admin) open on the coordinator only.
- Hospitals allow outbound TCP 8002 to the coordinator. No inbound rule is needed at a hospital.

**Software** (on each Spark and the coordinator)
- DGX OS (ships with Docker and the NVIDIA Container Toolkit), Git and Python 3.10+.
- NVIDIA's PyTorch container from NGC as the base image. Pick the newest tag that lists DGX Spark support and pin it in `deploy/Dockerfile`.
- NVFlare 2.9.0, installed in the image by the Dockerfile.

**Data**
- Each hospital: an OMOP CDM extract with the tables `person`, `condition_occurrence`, `drug_exposure`, `measurement` and `observation`, in the shape `ibdfl/features.py` reads.
- For the PoC, each Spark generates its own synthetic extract. Real registry data needs ethics approval, a DPIA and the data agreements from WP1 first.

**People**
- An IT contact at each hospital for the firewall rule and the Spark.
- One person at RISE who holds the project admin role and distributes the startup kits.

## Install and run

### 1. Create identities (coordinator or admin laptop, once)

Set the coordinator's real DNS name in `deploy/project.yml` (replace `anchor-fl-server`), then:

```bash
git clone <this repository> anchor-ibd-fl-poc && cd anchor-ibd-fl-poc
pip install nvflare==2.9.0
nvflare provision -p deploy/project.yml -w deploy/workspace
```

This creates a root certificate authority and one startup kit per participant in `deploy/workspace/anchor_ibd/prod_00/`:

| Kit | Goes to |
| --- | --- |
| `<coordinator name>/` | Coordinator host |
| `rost/` | Region Östergötland's Spark |
| `oru/` | Örebro's Spark |
| `ous/` | OUS's Spark |
| `admin@rise.se/` | RISE admin |

Each kit contains that participant's private key. Send every hospital only its own kit, over a secure channel.

### 2. Coordinator

```bash
docker build -f deploy/Dockerfile --build-arg BASE_IMAGE=python:3.12-slim --build-arg INSTALL_TORCH=1 -t anchor-ibd-fl .
deploy/run_site.sh server deploy/workspace/anchor_ibd/prod_00/<coordinator name>
```

The coordinator does no training, so the small CPU image is enough.

### 3. Each hospital Spark

```bash
git clone <this repository> anchor-ibd-fl-poc && cd anchor-ibd-fl-poc
docker build -f deploy/Dockerfile -t anchor-ibd-fl .            # NGC PyTorch base, GPU build

# check that the container sees the GPU
docker run --rm --gpus all anchor-ibd-fl python -c "import torch; print(torch.cuda.get_device_name(0))"

# PoC data: a synthetic OMOP extract for this site only (example: Östergötland)
docker run --rm -v /data/omop:/out anchor-ibd-fl python /opt/anchor/generate_data.py --out /out --site rost

# start the client with this site's kit and data
deploy/run_site.sh client ~/kits/rost /data/omop/rost <coordinator IP>
```

Use `oru` or `ous` in place of `rost` at the other hospitals. The log should show `Successfully registered client:rost for project anchor_ibd`.

### 4. Start training (admin)

```bash
python nvflare_job/job.py --workflow swarm --env prod \
  --startup_kit deploy/workspace/anchor_ibd/prod_00/admin@rise.se --data_root /data
```

Each Spark logs one line per round, for example:

```
[rost] round 4: global model val loss 0.630, test AUROC 0.765 (F 0.762 / M 0.765)
```

The client logs show which hospital aggregated each round (`aggregation happens on oru`). Add `--dp_noise 0.05` to clip each update and add Gaussian noise before it leaves the hospital.

To check status, use the admin console: `deploy/workspace/anchor_ibd/prod_00/admin@rise.se/startup/fl_admin.sh`, then `check_status client` and `list_jobs`.

### Try it on one Spark first

All four roles fit on a single Spark as separate containers, and each hospital container sees only its own data folder:

```bash
nvflare provision -p deploy/project.yml -w deploy/workspace
python generate_data.py --out data
KIT=deploy/workspace/anchor_ibd/prod_00 docker compose -f deploy/docker-compose.yml up -d --build
python nvflare_job/job.py --workflow swarm --env prod --startup_kit deploy/workspace/anchor_ibd/prod_00/admin@rise.se --data_root /data
```

For the admin to reach the coordinator container, add `127.0.0.1 anchor-fl-server` to `/etc/hosts` on that Spark.

## What has been tested

| | Status |
| --- | --- |
| Swarm and FedAvg workflows in the NVFlare simulator, with and without noise | Ran (WSL Ubuntu, NVFlare 2.9.0) |
| Provisioned deployment: certificates, coordinator plus 3 hospital processes over mTLS, job submitted by `admin@rise.se`, swarm and FedAvg for 5 rounds | Ran on one host in WSL; the coordinator log showed no aggregation in swarm mode |
| Docker image, compose file, `run_site.sh` | Not run; Docker was unavailable on the build machine |
| DGX Spark hardware, NGC base image on ARM64 | Not run; no hardware access |
