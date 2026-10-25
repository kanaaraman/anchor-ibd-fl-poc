"""Run the whole experiment in one process, without NVFlare.

Compares, on the same synthetic sites and test splits:
  local     each hospital trains only on its own data
  fedavg    federated averaging across the hospitals (what the Sparks run)
  fedavg_dp fedavg with clipped, noised updates at several noise levels
  central   all data pooled in one place (the upper bound FL tries to reach;
            not allowed in reality, possible here only because data is synthetic)

Every strategy picks its stopping point (epoch or round) on a validation
split; the test split is touched only for the reported numbers.

Writes results/results.json and dashboard/results.js.

    python simulate.py --data data --rounds 30 --seeds 5
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from ibdfl.features import FEATURES, load_site, split3
from ibdfl.model import IBDNet
from ibdfl.synth import SITES
from ibdfl.train import evaluate, local_train, privatize, state_delta, val_loss

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--eval_data", default="data_eval",
                help="held-out evaluation cohorts (generate_data.py --eval_n); if missing, a local test split is used")
ap.add_argument("--rounds", type=int, default=30)
ap.add_argument("--local_epochs", type=int, default=3)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--clip", type=float, default=1.0)
ap.add_argument("--noise", type=float, nargs="*", default=[0.05, 0.1, 0.2])
args = ap.parse_args()

sites = list(SITES)
data = {}
for s in sites:
    X, y, female, drug = load_site(os.path.join(args.data, s))
    if os.path.isdir(os.path.join(args.eval_data, s)):
        tr, va, _ = split3(len(y), val_frac=0.2, test_frac=0.0, seed=7)
        test = load_site(os.path.join(args.eval_data, s))
    else:
        tr, va, te = split3(len(y), seed=7)
        test = (X[te], y[te], female[te], drug[te])
    data[s] = dict(train=(X[tr], y[tr]), val=(X[va], y[va]), test=test)
pooled_test = tuple(np.concatenate([data[s]["test"][i] for s in sites]) for i in range(4))
n_train = {s: len(data[s]["train"][1]) for s in sites}
total = sum(n_train.values())
max_epochs = args.rounds * args.local_epochs


def eval_all(model):
    out = {s: evaluate(model, *data[s]["test"][:3]) for s in sites}
    out["pooled"] = evaluate(model, *pooled_test[:3])
    # rare subgroup: ustekinumab-treated patients, pooled across sites
    ust = pooled_test[3] == 3
    out["ustekinumab"] = evaluate(model, pooled_test[0][ust], pooled_test[1][ust], pooled_test[2][ust])
    return out


def fed_val_loss(model):
    """Validation loss averaged over sites, weighted by size. Each site computes its own term."""
    return sum(val_loss(model, *data[s]["val"]) * n_train[s] / total for s in sites)


def train_early_stopped(X, y, Xv, yv, seed):
    """Train up to max_epochs; keep the epoch with the lowest validation loss."""
    torch.manual_seed(seed)
    m = IBDNet()
    best, best_state, best_ep = float("inf"), None, 0
    for ep in range(max_epochs):
        local_train(m, X, y, epochs=1, seed=seed * 1000 + ep)
        v = val_loss(m, Xv, yv)
        if v < best:
            best, best_state, best_ep = v, copy.deepcopy(m.state_dict()), ep + 1
    m.load_state_dict(best_state)
    return m, best_ep


def run_local(seed):
    res = {}
    for s in sites:
        m, ep = train_early_stopped(*data[s]["train"], *data[s]["val"], seed)
        res[s] = eval_all(m)
        res[s]["best_epoch"] = ep
    return res


def run_central(seed):
    X = np.concatenate([data[s]["train"][0] for s in sites])
    y = np.concatenate([data[s]["train"][1] for s in sites])
    Xv = np.concatenate([data[s]["val"][0] for s in sites])
    yv = np.concatenate([data[s]["val"][1] for s in sites])
    m, ep = train_early_stopped(X, y, Xv, yv, seed)
    out = eval_all(m)
    out["best_epoch"] = ep
    return out


def run_fedavg(seed, noise=None, history=None):
    torch.manual_seed(seed)
    glob = IBDNet()
    n_params = sum(p.numel() for p in glob.parameters())
    best, best_state, best_round = float("inf"), None, 0
    for r in range(args.rounds):
        g_state = copy.deepcopy(glob.state_dict())
        agg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in g_state.items()}
        norms = {}
        for s in sites:
            m = IBDNet()
            m.load_state_dict(g_state)
            local_train(m, *data[s]["train"], epochs=args.local_epochs, seed=seed * 1000 + r)
            d = state_delta(m.state_dict(), g_state)
            if noise is not None:
                d, norms[s] = privatize(d, args.clip, noise, seed=seed * 10000 + r * 10 + sites.index(s))
            else:
                norms[s] = torch.sqrt(sum((v ** 2).sum() for v in d.values())).item()
            for k in agg:
                agg[k] += d[k] * (n_train[s] / total)
        glob.load_state_dict({k: g_state[k] + agg[k] for k in g_state})
        v = fed_val_loss(glob)
        if v < best:
            best, best_state, best_round = v, copy.deepcopy(glob.state_dict()), r + 1
        if history is not None:
            ev = eval_all(glob)
            history.append(dict(round=r + 1, val_loss=round(v, 4), update_norm={k: round(x, 4) for k, x in norms.items()},
                                bytes_up=n_params * 4 * len(sites), bytes_down=n_params * 4 * len(sites),
                                **{s: round(ev[s]["auroc"], 4) for s in sites + ["pooled"]},
                                **{f"{s}_female": round(ev[s]["auroc_female"], 4) for s in sites + ["pooled"]},
                                **{f"{s}_male": round(ev[s]["auroc_male"], 4) for s in sites + ["pooled"]}))
    glob.load_state_dict(best_state)
    out = eval_all(glob)
    out["best_round"] = best_round
    return out, n_params


def summarise(runs):
    """Mean and sd over seeds for every nested numeric leaf."""
    first = runs[0]
    if isinstance(first, dict):
        return {k: summarise([r[k] for r in runs]) for k in first}
    a = np.array(runs, dtype=float)
    return {"mean": round(float(np.nanmean(a)), 4), "sd": round(float(np.nanstd(a)), 4)}


t0 = time.time()
local, central, fed = [], [], []
dp = {z: [] for z in args.noise}
history = []
for seed in range(args.seeds):
    local.append(run_local(seed))
    central.append(run_central(seed))
    r, n_params = run_fedavg(seed, history=history if seed == 0 else None)
    fed.append(r)
    for z in args.noise:
        dp[z].append(run_fedavg(seed, noise=z)[0])
    print(f"seed {seed} done ({time.time() - t0:.0f}s)")

out = dict(
    config=dict(rounds=args.rounds, local_epochs=args.local_epochs, seeds=args.seeds, clip=args.clip,
                model="MLP 19-32-16-1", n_params=n_params, features=FEATURES,
                outcome="Steroid-free remission at week 52 on index biologic"),
    sites={s: dict(**{k: v for k, v in SITES[s].items() if k in ("name", "registry", "country")},
                   n_train=n_train[s], n_test=int(len(data[s]["test"][1])),
                   n_val=int(len(data[s]["val"][1])),
                   response_rate=round(float(np.concatenate([data[s][k][1] for k in ("train", "val")]).mean()), 3),
                   share_female=round(float(data[s]["test"][2].mean()), 3),
                   share_ustekinumab=round(float((data[s]["test"][3] == 3).mean()), 3))
           for s in sites},
    local=summarise(local), central=summarise(central), fedavg=summarise(fed),
    fedavg_dp={str(z): summarise(v) for z, v in dp.items()},
    history=history,
    patient_records_moved=0,
    eval_source="held-out synthetic cohort per site" if os.path.isdir(args.eval_data) else "25% local test split",
)
os.makedirs("results", exist_ok=True)
with open("results/results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
os.makedirs("dashboard", exist_ok=True)
with open("dashboard/results.js", "w", encoding="utf-8") as f:
    f.write("window.RESULTS = " + json.dumps(out, ensure_ascii=False) + ";\n")

print("\nAUROC on each site's own test set (mean over seeds)")
print(f"{'':10s}" + "".join(f"{s:>10s}" for s in sites + ["pooled"]))
for name, res in [("central", out["central"]), ("fedavg", out["fedavg"])] + \
        [(f"dp z={z}", out["fedavg_dp"][str(z)]) for z in args.noise]:
    print(f"{name:10s}" + "".join(f"{res[s]['auroc']['mean']:10.3f}" for s in sites + ["pooled"]))
print(f"{'local':10s}" + "".join(f"{out['local'][s][s]['auroc']['mean']:10.3f}" for s in sites) +
      f"{np.mean([out['local'][s]['pooled']['auroc']['mean'] for s in sites]):10.3f}  (pooled = avg of 3 local models)")
print(f"\nustekinumab subgroup AUROC: local(rost) {out['local']['rost']['ustekinumab']['auroc']['mean']:.3f}  "
      f"fedavg {out['fedavg']['ustekinumab']['auroc']['mean']:.3f}  central {out['central']['ustekinumab']['auroc']['mean']:.3f}")
