"""Track C, part 2: a hypergraph neural network over the change stream as the system's memory.

Pipeline:  git log stream -> hypergraph builder -> temporal graph encoder -> learned architecture matrix
           -> rule generator (prediction + anomaly)

- Hypergraph per time window: nodes = components, hyperedges = commits (the set of components a commit touched),
  node features = log activity in the window (commits, files, churn) plus the layer level.
- Spatial: two hypergraph convolutions (Feng et al. 2019): X' = D_v^-1/2 H W D_e^-1 H^T D_v^-1/2 X Theta,
  so a component's state mixes with the components it co-evolves with inside the same hyperedges.
- Temporal: a GRU over the windows (batch = components) captures growing / stabilising / declining.
- Decoder: from the states of x and y, predict next window's directional co-change confidence C(x -> y).
  The matrix of these predictions at the last window is the *learned architecture matrix*.
- Dynamic baseline: exponential moving averages of per-window prediction error and of each component's
  state; a window whose error is > BASELINE_Z standard deviations above its moving average is a drift event;
  a component whose state moved more than DRIFT_COS from its moving average is a structural drift.
- Rules: predicted C(x -> y) >= RULE_CONFIDENCE with support -> implicit coupling rule
  "a change to x forces a change to y" (formal: G(changed(x) -> changed(y)) per commit); predicted coupling
  across a layer boundary in the wrong direction -> layer-constraint candidate, same card format as the
  structural rules so the Differ can check it against the import graph.

Needs torch (uv pip install torch). Deterministic: seeded, CPU, single thread.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sarms.rules.cochange import level

RULE_CONFIDENCE = 0.98
BASELINE_Z = 3.0
DRIFT_COS = 0.5
MIN_SUPPORT_WINDOWS = 3


def windows_of(df: pd.DataFrame, days: int = 90):
    """Split the change stream into consecutive windows of `days`; returns list of (start, end, frame)."""
    t0, t1 = df.ts.min(), df.ts.max()
    out, start = [], t0
    while start <= t1:
        end = start + pd.Timedelta(days=days)
        w = df[(df.ts >= start) & (df.ts < end)]
        if len(w):
            out.append((start, end, w))
        start = end
    return out


def incidence(w: pd.DataFrame, nodes: list[str]):
    """Hyperedges = commits touching >= 2 components; incidence H (n x m), node features (n x f), directional C."""
    index = {c: i for i, c in enumerate(nodes)}
    groups = w.dropna(subset=["component"]).groupby("idx")
    cols, feats = [], np.zeros((len(nodes), 4), dtype=np.float32)
    single, pair = np.zeros(len(nodes)), np.zeros((len(nodes), len(nodes)))
    for _, g in groups:
        comps = sorted(set(g.component))
        ids = [index[c] for c in comps if c in index]
        for i in ids:
            single[i] += 1
        for i in ids:
            for j in ids:
                if i != j:
                    pair[i, j] += 1
        if len(ids) >= 2:
            col = np.zeros(len(nodes), dtype=np.float32)
            col[ids] = 1.0
            cols.append(col)
    act = w.dropna(subset=["component"]).groupby("component").agg(commits=("idx", "nunique"), files=("path", "nunique"), churn=("added", "sum"))
    for c, r in act.iterrows():
        if c in index:
            feats[index[c], :3] = np.log1p([r.commits, r.files, r.churn])
    for c, i in index.items():
        lv = level(c)
        feats[i, 3] = (lv / 4.0) if lv is not None else 0.5
    H = np.stack(cols, axis=1) if cols else np.zeros((len(nodes), 0), dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(single[:, None] > 0, pair / np.maximum(single[:, None], 1), 0.0)
    return H, feats, C, single


def hgnn_operator(H: np.ndarray):
    """The normalised hypergraph Laplacian operator D_v^-1/2 H D_e^-1 H^T D_v^-1/2 (W = I), with self-loops so a
    component with no hyperedge keeps its own state."""
    n = H.shape[0]
    if H.shape[1] == 0:
        return np.eye(n, dtype=np.float32)
    Hs = np.concatenate([H, np.eye(n, dtype=np.float32)], axis=1)
    dv = Hs.sum(1)
    de = Hs.sum(0)
    Dv = np.diag(1.0 / np.sqrt(dv))
    De = np.diag(1.0 / de)
    return (Dv @ Hs @ De @ Hs.T @ Dv).astype(np.float32)


def train(df: pd.DataFrame, nodes: list[str], days: int = 90, hidden: int = 32, epochs: int = 300, seed: int = 2569, log=print,
          train_until: pd.Timestamp | None = None):
    """Fit the encoder on windows ending before train_until (default: all). Returns a dict with the model, the
    learned architecture matrix, per-window errors, drift events and the baseline."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    wins = windows_of(df, days)
    data = [incidence(w, nodes) for _, _, w in wins]
    n = len(nodes)
    ops = [torch.tensor(hgnn_operator(H)) for H, _, _, _ in data]
    X = [torch.tensor(f) for _, f, _, _ in data]
    targets = [torch.tensor(C, dtype=torch.float32) for _, _, C, _ in data]
    masks = [torch.tensor((s > 0).astype(np.float32)) for _, _, _, s in data]  # x changed in that window
    T = len(wins)
    n_train = T if train_until is None else sum(1 for s, e, _ in wins if e <= train_until)
    if n_train < 3:
        raise ValueError("need at least 3 windows to train")

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.theta1 = nn.Linear(4, hidden)
            self.theta2 = nn.Linear(hidden, hidden)
            self.gru = nn.GRUCell(hidden, hidden)
            self.dec = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def spatial(self, A, x):
            h = torch.relu(A @ self.theta1(x))
            return torch.relu(A @ self.theta2(h))

        def forward(self, upto):
            h = torch.zeros(n, hidden)
            states = []
            for t in range(upto):
                s = self.spatial(ops[t], X[t])
                h = self.gru(s, h)
                states.append(h)
            return states

        def predict(self, h):
            hx = h.unsqueeze(1).expand(n, n, hidden)
            hy = h.unsqueeze(0).expand(n, n, hidden)
            return torch.sigmoid(self.dec(torch.cat([hx, hy, hx * hy], dim=-1)).squeeze(-1))

    model = Encoder()
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    bce = nn.BCELoss(reduction="none")
    eye = torch.eye(n)
    for ep in range(epochs):
        opt.zero_grad()
        states = model.forward(n_train - 1)
        loss = 0.0
        for t, h in enumerate(states):            # state after window t predicts window t+1
            P = model.predict(h)
            m = masks[t + 1][:, None] * (1 - eye)
            loss = loss + (bce(P, targets[t + 1]) * m).sum() / m.sum().clamp(min=1)
        loss = loss / len(states)
        loss.backward()
        opt.step()
        if ep % 100 == 0:
            log(f"    epoch {ep} loss {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        states = model.forward(T)
        errors, preds = [], []
        for t, h in enumerate(states[:-1]):
            P = model.predict(h)
            m = masks[t + 1][:, None] * (1 - eye)
            err = ((P - targets[t + 1]).abs() * m).sum() / m.sum().clamp(min=1)
            errors.append(float(err))
            preds.append(P.numpy())
        A = model.predict(states[-1]).numpy()
        # dynamic baseline: EMA of error and of states; drift events where the error leaves the band
        ema, var, drift = None, 0.0, []
        for t, e in enumerate(errors):
            if ema is None:
                ema = e
            else:
                z = (e - ema) / (np.sqrt(var) if var > 0 else 1.0)
                if t >= 3 and z > BASELINE_Z:
                    drift.append({"window": t + 1, "start": str(wins[t + 1][0].date()), "error": round(e, 4), "baseline": round(ema, 4), "z": round(float(z), 2)})
                var = 0.8 * var + 0.2 * (e - ema) ** 2
                ema = 0.8 * ema + 0.2 * e
        sm = torch.stack(states).numpy()  # T x n x hidden
        struct_drift = []
        base = sm[0]
        for t in range(1, T):
            cos = 1 - np.sum(sm[t] * base, 1) / (np.linalg.norm(sm[t], axis=1) * np.linalg.norm(base, axis=1) + 1e-9)
            for i in np.where(cos > DRIFT_COS)[0]:
                struct_drift.append({"window": t, "start": str(wins[t][0].date()), "component": nodes[i], "cosine_distance": round(float(cos[i]), 3)})
            base = 0.8 * base + 0.2 * sm[t]
    support = np.sum([(s > 0) for _, _, _, s in data[:n_train]], axis=0)
    return {"model": model, "windows": [(str(s.date()), str(e.date())) for s, e, _ in wins], "n_train_windows": n_train,
            "matrix": A, "errors": errors, "preds": preds, "targets": [t.numpy() for t in targets], "masks": [m.numpy() for m in masks],
            "drift_events": drift, "structural_drift": struct_drift[:50], "support_windows": support, "nodes": nodes}


def rules_from_matrix(fit, confidence=RULE_CONFIDENCE, min_support=MIN_SUPPORT_WINDOWS):
    """Implicit coupling rules and layer-constraint candidates from the learned architecture matrix."""
    A, nodes, sup = fit["matrix"], fit["nodes"], fit["support_windows"]
    coupling, layer = [], []
    for i, x in enumerate(nodes):
        if sup[i] < min_support:
            continue
        for j, y in enumerate(nodes):
            if i == j or A[i, j] < confidence:
                continue
            r = {"x": x, "y": y, "predicted_C": round(float(A[i, j]), 4), "support_windows": int(sup[i]),
                 "constraint": f"G(changed({x}) -> changed({y}))", "kind": "coupling"}
            coupling.append(r)
            lx, ly = level(x), level(y)
            if lx is not None and ly is not None and lx < ly:
                layer.append({**r, "kind": "layer", "constraint": f"{x} must not depend on {y}"})
    return coupling, layer


def evaluate_forward(fit, confidence=RULE_CONFIDENCE):
    """How did predictions made from the training windows fare on the held-out windows? For pairs predicted
    >= confidence at the end of training: realised C in the test windows (precision of 'forced change').
    Baseline: plain count-based C over the training windows for the same pairs."""
    k = fit["n_train_windows"]
    preds, targets, masks = fit["preds"], fit["targets"], fit["masks"]
    if k - 1 >= len(preds):
        return {"test_windows": 0}
    P = preds[k - 1]                         # prediction made after the last training window
    n = P.shape[0]
    num, den = np.zeros((n, n)), np.zeros((n, n))
    for t in range(k, len(targets)):
        num += targets[t] * masks[t][:, None]
        den += masks[t][:, None]
    realised = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    tr_num = sum(targets[t] * masks[t][:, None] for t in range(k))
    tr_den = sum(masks[t][:, None] for t in range(k))
    count_C = np.where(tr_den > 0, tr_num / np.maximum(tr_den, 1), 0.0)
    out = {"test_windows": len(targets) - k}
    for name, M in (("hgnn", P), ("count", count_C)):
        sel = (M >= confidence) & ~np.eye(n, dtype=bool) & (den > 0)
        out[name] = {"pairs_predicted": int(sel.sum()), "mean_realised_C": round(float(np.nanmean(realised[sel])), 3) if sel.any() else None,
                     "share_realised_over_0_5": round(float(np.nanmean(realised[sel] >= 0.5)), 3) if sel.any() else None,
                     "mae_all_pairs": round(float(np.nanmean(np.abs(M - realised)[den > 0])), 4)}
    return out


def save(fit, out_dir: Path, coupling, layer, forward):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fit["matrix"], index=fit["nodes"], columns=fit["nodes"]).to_csv(out_dir / "architecture_matrix.csv")
    (out_dir / "hgnn.json").write_text(json.dumps({"windows": fit["windows"], "n_train_windows": fit["n_train_windows"], "errors": fit["errors"],
                                                   "drift_events": fit["drift_events"], "structural_drift": fit["structural_drift"],
                                                   "coupling_rules": coupling, "layer_candidates": layer, "forward": forward}, indent=1, default=str))
