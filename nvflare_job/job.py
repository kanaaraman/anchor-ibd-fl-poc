"""Build the ANCHOR-IBD training job with the NVFlare 2.9 recipe API.

Two workflows, same client script:

  swarm   decentralised (default). Hospitals exchange the model peer to peer;
          each round a randomly chosen hospital aggregates. The server only
          admits participants and starts/stops the job: it never receives,
          holds or averages model weights.
  fedavg  classic federated averaging, with the server as aggregator.

    # simulate 3 hospitals in one process (laptop or one Spark)
    python nvflare_job/job.py --env sim --data_root $PWD/data

    # submit to a provisioned deployment (one Spark per hospital)
    python nvflare_job/job.py --env prod --startup_kit ~/anchor/prod_00/admin@rise.se --data_root /data

    # just write the job folder (submit later with the admin console)
    python nvflare_job/job.py --export --export-dir jobs/anchor_ibd
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe  # noqa: E402
from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe  # noqa: E402
from nvflare.recipe import ProdEnv, SimEnv  # noqa: E402

from ibdfl.model import IBDNet  # noqa: E402
from ibdfl.synth import SITES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--workflow", choices=["swarm", "fedavg"], default="swarm")
ap.add_argument("--env", choices=["sim", "prod"], default="sim")
ap.add_argument("--startup_kit", help="admin startup kit folder (prod)")
ap.add_argument("--admin_user", default="admin@rise.se", help="admin identity in the startup kit (prod)")
ap.add_argument("--data_root", default="/data", help="each site reads <data_root>/<site name>")
ap.add_argument("--rounds", type=int, default=30)
ap.add_argument("--local_epochs", type=int, default=3)
ap.add_argument("--dp_noise", type=float, default=0.0)
ap.add_argument("--workspace", default="/tmp/nvflare/anchor_ibd")
args, _ = ap.parse_known_args()

sites = list(SITES)
script = os.path.join(HERE, "client.py")
script_args = f"--data_root {args.data_root} --local_epochs {args.local_epochs} --dp_noise {args.dp_noise}"

if args.workflow == "swarm":
    recipe = SwarmLearningRecipe(
        name="anchor_ibd_swarm",
        model=IBDNet(),
        num_rounds=args.rounds,
        train_script=script,
        train_args={"script_args": script_args},
        min_clients=len(sites),
        key_metric="val_loss",
        key_metric_mode="min",
    )
else:
    recipe = FedAvgRecipe(
        name="anchor_ibd_fedavg",
        model=IBDNet(),
        min_clients=len(sites),
        num_rounds=args.rounds,
        train_script=script,
        train_args=script_args,
        key_metric="val_loss",
        key_metric_mode="min",
    )
recipe.add_client_file(os.path.join(os.path.dirname(HERE), "ibdfl"))

if args.env == "sim":
    env = SimEnv(clients=sites, num_threads=len(sites), workspace_root=args.workspace)
else:
    env = ProdEnv(startup_kit_location=args.startup_kit, username=args.admin_user)

run = recipe.execute(env)
print("job id:", run.get_job_id())
print("status:", run.get_status())
print("result:", run.get_result())
