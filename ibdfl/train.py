"""Local training, evaluation and privacy helpers shared by the simulator and the NVFlare client."""
import numpy as np
import torch
from torch import nn


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def local_train(model, X, y, epochs=3, lr=1e-3, batch=32, weight_decay=1e-4, seed=0):
    """Train in place on one site's data. Returns number of optimiser steps."""
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    steps = 0
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss_fn(model(Xt[b].to(dev)), yt[b].to(dev)).backward()
            opt.step()
            steps += 1
    return steps


@torch.no_grad()
def predict(model, X):
    dev = next(model.parameters()).device
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(X).to(dev))).cpu().numpy()


@torch.no_grad()
def val_loss(model, X, y):
    p = np.clip(predict(model, X), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auroc(y, p):
    """Rank-based AUROC (Mann-Whitney U); NaN if only one class present."""
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def evaluate(model, X, y, female):
    p = predict(model, X)
    return {
        "auroc": auroc(y, p), "brier": brier(y, p), "n": int(len(y)),
        "auroc_female": auroc(y[female], p[female]), "auroc_male": auroc(y[~female], p[~female]),
        "pred_rate_female": float(p[female].mean()), "obs_rate_female": float(y[female].mean()),
        "pred_rate_male": float(p[~female].mean()), "obs_rate_male": float(y[~female].mean()),
    }


def state_delta(new, old):
    return {k: new[k].detach().float() - old[k].detach().float() for k in new}


def privatize(delta, clip=1.0, noise_multiplier=0.0, seed=0):
    """Clip the update's global L2 norm to `clip`, then add N(0, (noise_multiplier*clip)^2).

    This is the per-site Gaussian mechanism applied before an update leaves the
    hospital. It bounds what any single update can reveal; it is not a full
    privacy accountant (no epsilon is computed here).
    """
    norm = torch.sqrt(sum((v ** 2).sum() for v in delta.values())).item()
    scale = min(1.0, clip / (norm + 1e-12))
    g = torch.Generator()
    if seed is None:
        g.seed()  # fresh, unpredictable noise: what a real deployment must use
    else:
        g.manual_seed(seed)  # reproducible noise for the simulator only
    out = {}
    for k, v in delta.items():
        v = v * scale
        if noise_multiplier > 0:
            v = v + torch.randn(v.shape, generator=g).to(v.device) * noise_multiplier * clip
        out[k] = v
    return out, norm
