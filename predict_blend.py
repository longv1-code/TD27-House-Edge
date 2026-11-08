"""Team ensemble: 0.5 x plus model (predict.py, unchanged) + 0.5 x generative switching model (gen.py).
python predict_blend.py --data /data --out /output/submission.csv"""
import argparse, csv, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import gen, gen_data

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--weight", type=float, default=0.5, help="weight of the generative model")
a = ap.parse_args()

tmp = Path(tempfile.mkdtemp()) / "plus.csv"
subprocess.run([sys.executable, str(HERE / "predict.py"), "--predict", "--data", a.data, "--out", str(tmp)], check=True, cwd=str(HERE))
plus = {}
with open(tmp, newline="", encoding="utf-8") as f:
    r = csv.reader(f); next(r)
    for row in r:
        plus[row[0]] = np.array([float(x) for x in row[1:]], np.float64)

S = gen_data.Split(Path(a.data) / "test")
g = gen.full_predict(S, gen.load_params(HERE / "gen_params.npz"))
assert set(plus) == set(S.qid) and len(plus) == len(S.qid), "query sets differ"
out = np.stack([a.weight * g[i] + (1 - a.weight) * plus[q] for i, q in enumerate(S.qid)])
out = np.clip(out, 1e-6, None)
out = out / out.sum(1, keepdims=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"])
    for i in np.argsort(S.qid):
        w.writerow([S.qid[i]] + [format(float(x), ".17g") for x in out[i]])
