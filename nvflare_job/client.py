"""NVFlare client: runs inside each hospital's DGX Spark.

Per round: receive the global model, score it on the local test split, train
a few epochs on local data, and send back weights plus aggregate metrics.
Only model parameters and summary metrics leave the site, never records.
"""
import argparse
import copy
import os

import nvflare.client as flare
import torch

from ibdfl.features import load_site, split3
from ibdfl.model import IBDNet
from ibdfl.train import device, evaluate, local_train, privatize, state_delta, val_loss

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", default="/data")
ap.add_argument("--local_epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--dp_clip", type=float, default=1.0)
ap.add_argument("--dp_noise", type=float, default=0.0, help="0 disables noise; clipping still bounds each update")
args = ap.parse_args()

flare.init()
site = flare.get_site_name()
X, y, female, _ = load_site(os.path.join(args.data_root, site))
tr, va, te = split3(len(y), seed=7)
dev = device()
print(f"[{site}] {len(tr)} train / {len(va)} val / {len(te)} test patients on {dev}")

model = IBDNet().to(dev)
while flare.is_running():
    incoming = flare.receive()
    model.load_state_dict(incoming.params)
    glob = copy.deepcopy(model.state_dict())
    # val_loss drives the server's best-model selection; test_* is for reporting only
    ev = {"val_loss": val_loss(model, X[va], y[va]),
          **{f"test_{k}": v for k, v in evaluate(model, X[te], y[te], female[te]).items()}}
    print(f"[{site}] round {incoming.current_round}: global model val loss {ev['val_loss']:.3f}, "
          f"test AUROC {ev['test_auroc']:.3f} (F {ev['test_auroc_female']:.3f} / M {ev['test_auroc_male']:.3f})")
    if flare.is_evaluate():
        flare.send(flare.FLModel(metrics=ev))
        continue

    steps = local_train(model, X[tr], y[tr], epochs=args.local_epochs, lr=args.lr,
                        seed=incoming.current_round or 0)
    # bound (and optionally noise) this site's contribution before it leaves the hospital
    d, norm = privatize(state_delta(model.state_dict(), glob), args.dp_clip, args.dp_noise, seed=None)
    out = {k: (glob[k].float() + d[k]).cpu() for k in glob}
    flare.send(flare.FLModel(params=out, metrics={**ev, "update_norm": norm},
                             meta={"NUM_STEPS_CURRENT_ROUND": steps}))
