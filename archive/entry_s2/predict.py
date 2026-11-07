"""Deterministic forward pass from a saved checkpoint. python predict.py --data /data --out /output/submission.csv [--solution s5]"""
import argparse, csv
from pathlib import Path
import numpy as np
import he

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--solution", default="s5", choices=["s1", "s2", "s3", "s4", "s5"])
ap.add_argument("--ckpt", default=str(Path(__file__).resolve().parent / "model.npz"))
a = ap.parse_args()
M, Wg, prior = he.load_ckpt(a.ckpt)
S = he.Split(Path(a.data) / "test")
C = he.Corpus([S], prior)
P = he.predict_models(M, C, np.arange(C.M))
if a.solution == "s5":
    out = he.apply_gate(Wg, np.stack([P[n] for n in ("s1", "s2", "s3", "s4")]), he.gate_feats(C.qX))
else:
    out = P[a.solution]
out = np.clip(out, 1e-6, None)
out = out / out.sum(1, keepdims=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"])
    for i in np.argsort(C.qid):
        w.writerow([C.qid[i]] + [format(float(x), ".17g") for x in out[i]])
