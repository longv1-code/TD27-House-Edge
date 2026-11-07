"""Fit all five solutions on a kit's train+validation and write a checkpoint.
Usage: python fit_final.py --data public --ckpt model_main.npz [--no-val] [--extra OTHER_PUBLIC] [--max-points 60]
--no-val : train on public/train only (validation stays a clean holdout)
--extra  : also train on another round's train+validation (round two: test if main data helps via dev.py first)."""
import argparse, time
import numpy as np
import he

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="public")
ap.add_argument("--ckpt", default="model.npz")
ap.add_argument("--no-val", action="store_true")
ap.add_argument("--extra", default=None)
ap.add_argument("--max-points", type=float, default=60.0)
ap.add_argument("--folds", type=int, default=4)
a = ap.parse_args()
t0 = time.time()
tr, va = he.Split(f"{a.data}/train"), he.Split(f"{a.data}/validation")
splits = [tr] + ([] if a.no_val else [va])
if a.extra:
    splits += [he.Split(f"{a.extra}/train"), he.Split(f"{a.extra}/validation")]
prior = he.global_prior(splits)
C = he.Corpus(splits, prior)
cfg = dict(he.DEFAULT_CFG)
fm = np.zeros(C.M, int)
r = np.random.RandomState(0)
for c in np.unique(C.cond):
    eps = np.unique(C.ep[C.cond == c]); r.shuffle(eps)
    for i, e in enumerate(eps):
        fm[C.ep == e] = i % a.folds
names = ("s1", "s2", "s3", "s4")
oof = {n: np.zeros((C.M, 5)) for n in names}
for k in range(a.folds):
    M = he.fit_models(C, fm != k, cfg)
    sel = np.nonzero(fm == k)[0]
    for n, p in he.predict_models(M, C, sel).items():
        oof[n][sel] = p
    print(f"fold {k} done {time.time()-t0:.0f}s", flush=True)
P = np.stack([oof[n] for n in names])
G = he.gate_feats(C.qX)
# cross-fitted gate for an honest OOF estimate, then final gate on everything
oof5 = np.zeros((C.M, 5))
for k in range(a.folds):
    Wk = he.fit_gate(P[:, fm != k], C.qY[fm != k], G[fm != k])
    oof5[fm == k] = he.apply_gate(Wk, P[:, fm == k], G[fm == k])
oof["s5"] = oof5
Wg = he.fit_gate(P, C.qY, G)
if not a.no_val:  # score OOF on the primary validation machines (true targets + pooled reference)
    lo, hi = tr.M, tr.M + va.M
    class V: pass
    v = V(); v.cond, v.ep, v.tgt, v.ref = C.cond[lo:hi], C.ep[lo:hi], C.tgt[lo:hi], C.ref[lo:hi]
    print("\nOOF on validation machines (models never saw that episode's fold):")
    for n in ("s1", "s2", "s3", "s4", "s5"):
        res = he.contest(oof[n][lo:hi], v, a.max_points)
        print(f"  {n}: points={res['points']:.2f} R={res['R']:.3f}")
M = he.fit_models(C, np.ones(C.M, bool), cfg)
he.save_ckpt(a.ckpt, M, Wg, prior)
print(f"saved {a.ckpt}  total {time.time()-t0:.0f}s")
