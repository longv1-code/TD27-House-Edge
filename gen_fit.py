"""Fit generative-model parameters by EM on <data>/train (+validation unless --no-val); save gen_params.npz."""
import argparse, time, numpy as np
import gen, gen_data
ap = argparse.ArgumentParser(); ap.add_argument("--data", default="public"); ap.add_argument("--out", default="gen_params.npz")
ap.add_argument("--no-val", action="store_true"); ap.add_argument("--iters", type=int, default=3)
a = ap.parse_args(); t0 = time.time()
splits = [gen_data.Split(f"{a.data}/train")] + ([] if a.no_val else [gen_data.Split(f"{a.data}/validation")])
P = gen.em(splits, gen.init_params(splits), iters=a.iters, log=lambda s: print(s, f"{time.time()-t0:.0f}s", flush=True))
np.savez(a.out, **{k: np.asarray(v, np.float64) for k, v in P.items()})
print("saved", a.out)
