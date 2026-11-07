"""Honest dev run: fit on public/train only, score on public/validation (never used for fitting).
Usage: python dev.py [--data public] [--max-points 60] [--folds 4]"""
import argparse
import time

import numpy as np

import he

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="public")
ap.add_argument("--max-points", type=float, default=60.0)
ap.add_argument("--folds", type=int, default=4)
a = ap.parse_args()
t0 = time.time()
tr, va = he.Split(f"{a.data}/train"), he.Split(f"{a.data}/validation")
prior = he.global_prior([tr])
Ctr, Cva = he.Corpus([tr], prior), he.Corpus([va], prior)
cfg = dict(he.DEFAULT_CFG)
print(f"features ready {time.time()-t0:.0f}s  train ex={len(Ctr.hX)} val ex={len(Cva.hX)} F={Ctr.hX.shape[1]}")


def folds_for(C, K, seed=0):
    r = np.random.RandomState(seed)
    fm = np.zeros(C.M, int)
    for c in np.unique(C.cond):
        eps = np.unique(C.ep[C.cond == c])
        r.shuffle(eps)
        for i, e in enumerate(eps):
            fm[C.ep == e] = i % K
    return fm


# 1) OOF on train -> fit the gate (outcome targets only)
fm = folds_for(Ctr, a.folds)
oof = {k: np.zeros((Ctr.M, 5)) for k in ("s1", "s2", "s3", "s4")}
for k in range(a.folds):
    M = he.fit_models(Ctr, fm != k, cfg)
    sel = np.nonzero(fm == k)[0]
    for name, p in he.predict_models(M, Ctr, sel).items():
        oof[name][sel] = p
    print(f"  fold {k} done {time.time()-t0:.0f}s")
names = ("s1", "s2", "s3", "s4")
Wg = he.fit_gate(np.stack([oof[n] for n in names]), Ctr.qY, he.gate_feats(Ctr.qX))

# 2) models on all train -> validation
M = he.fit_models(Ctr, np.ones(Ctr.M, bool), cfg)
pv = he.predict_models(M, Cva, np.arange(Cva.M))
pv["s5"] = he.apply_gate(Wg, np.stack([pv[n] for n in names]), he.gate_feats(Cva.qX))
pv["mean4"] = np.mean([pv[n] for n in names], 0)
print(f"\nS1 chosen: half_life={M['s1']['hl']} strength={M['s1']['strength']}")
print(f"reference pooled baseline would score {a.max_points/2:.1f}\n")
for n in ("s1", "s2", "s3", "s4", "mean4", "s5"):
    r = he.contest(pv[n], Cva, a.max_points)
    print(f"{n:6s} points={r['points']:.2f}  R={r['R']:.3f}  {r['by_cond']}")
print(f"total {time.time()-t0:.0f}s")
