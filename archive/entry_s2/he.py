"""House Edge toolkit (numpy only). Features use ticks strictly below the target tick."""
import csv
import json
from pathlib import Path

import numpy as np

PAY = (0, 1, 2, 5, 20)
PIDX = {p: i for i, p in enumerate(PAY)}
TYPES = ("steady", "balanced", "jackpot")
PCOLS = ("p_0", "p_1", "p_2", "p_5", "p_20")
T = 260
NL = 8
WIN = (8, 16, 32, 64, 128)
NWIN = (8, 32, 128)
TMIN = 24
LAYOUT = {}


def _rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ----------------------------------------------------------------------------- data
class Split:
    def __init__(self, d):
        d = Path(d)
        Q, H = _rows(d / "queries.csv"), _rows(d / "history.csv")
        self.qid = [q["query_id"] for q in Q]
        M = self.M = len(Q)
        eps = {}
        self.ep = np.array([eps.setdefault(q["episode_id"], len(eps)) for q in Q])
        self.E = len(eps)
        self.mtype = np.array([TYPES.index(q["machine_type"]) for q in Q])
        self.sig = np.array([int(q["signal"]) + 1 for q in Q])
        self.cut = np.array([int(q["cutoff"]) for q in Q])
        mid = {(q["episode_id"], q["machine_id"]): i for i, q in enumerate(Q)}
        C = -np.ones((M, T), np.int8)
        for r in H:
            i = mid.get((r["episode_id"], r["machine_id"]))
            if i is not None:
                C[i, int(r["tick"])] = PIDX[int(r["payout"])]
        C[np.arange(T)[None, :] >= self.cut[:, None]] = -1  # nothing at/after cutoff
        self.C = C
        pos = {q: i for i, q in enumerate(self.qid)}
        self.out = self.tgt = self.ref = self.cond = None
        if (d / "labels.csv").exists():
            self.out = np.zeros(M, int)
            for r in _rows(d / "labels.csv"):
                self.out[pos[r["query_id"]]] = PIDX[int(r["outcome"])]
        if (d / "targets.csv").exists():
            self.tgt = np.zeros((M, 5))
            self.out = np.zeros(M, int)
            for r in _rows(d / "targets.csv"):
                i = pos[r["query_id"]]
                self.tgt[i] = [float(r[c]) for c in PCOLS]
                self.out[i] = PIDX[int(r["outcome"])]
        if (d / "reference.csv").exists():
            self.ref = np.zeros((M, 5))
            for r in _rows(d / "reference.csv"):
                self.ref[pos[r["query_id"]]] = [float(r[c]) for c in PCOLS]
        if (d / "conditions.csv").exists():
            self.cond = np.empty(M, object)
            for r in _rows(d / "conditions.csv"):
                self.cond[pos[r["query_id"]]] = r["condition"]
        self._tensors()

    def _tensors(self):
        M = self.M
        OH = np.zeros((M, T, 5), np.float32)
        mi, ti = np.nonzero(self.C >= 0)
        OH[mi, ti, self.C[mi, ti]] = 1
        self.OH = OH
        self.CS = np.zeros((M, T + 1, 5), np.float32)
        self.CS[:, 1:] = np.cumsum(OH, 1)
        EOH = np.zeros((self.E, T, 5), np.float32)
        np.add.at(EOH, self.ep, OH)
        self.NOH = EOH[self.ep] - OH
        ECS = np.zeros((self.E, T + 1, 5), np.float32)
        ECS[:, 1:] = np.cumsum(EOH, 1)
        self.NCS = ECS[self.ep] - self.CS
        self.esz = np.bincount(self.ep)[self.ep].astype(np.float32)
        # per-machine lag tables: LT[k-1][m, t, prev_state(loss/win), class_now] = #pairs (s-k, s) with s < t
        self.LT = []
        C = self.C.astype(np.int16)
        for k in range(1, NL + 1):
            ind = np.zeros((M, T + 1, 2, 5), np.float32)
            cur, prv = C[:, k:], C[:, :-k]
            ok = (cur >= 0) & (prv >= 0)
            m_, s_ = np.nonzero(ok)
            ind[m_, s_ + k + 1, (prv[m_, s_] > 0).astype(int), cur[m_, s_]] = 1  # counted for t > s
            self.LT.append(np.cumsum(ind, 1))


def feats(S, mi, ti, prior):
    """Feature rows for (machine, tick) pairs. Only ticks < ti are read."""
    f, pos = [], [0]
    first = not LAYOUT

    def add(name, a):
        a = np.asarray(a, np.float32)
        if a.ndim == 1:
            a = a[:, None]
        f.append(a)
        if first:
            LAYOUT[name] = slice(pos[0], pos[0] + a.shape[1])
        pos[0] += a.shape[1]

    mt, sg = S.mtype[mi], S.sig[mi]
    add("type", np.eye(3)[mt])
    add("sig", np.eye(3)[sg])
    add("typesig", np.eye(9)[mt * 3 + sg])
    add("t", ti / 256.0)
    lagoh, W, L, nfrac = [], [], [], []
    for k in range(1, NL + 1):
        tk = np.maximum(ti - k, 0)
        ok = (ti - k >= 0)[:, None]
        v = S.OH[mi, tk] * ok
        nv = S.NOH[mi, tk] * ok
        lagoh.append(v)
        W.append(v[:, 1:].sum(1))
        L.append(v[:, 0])
        nfrac.append((nv[:, 1:].sum(1) + 0.8) / (nv.sum(1) + 2.0))
    add("lagoh", np.concatenate(lagoh, 1))
    wl = np.concatenate([np.stack(W, 1), np.stack(L, 1)], 1)
    add("wl", wl)
    add("wl_t", np.concatenate([wl * (mt == j)[:, None] for j in range(3)], 1))
    add("wl_s", np.concatenate([wl * (sg == j)[:, None] for j in range(3)], 1))
    add("nfrac", np.stack(nfrac, 1))
    for w in WIN:
        lo = np.maximum(ti - w, 0)
        cnt = S.CS[mi, ti] - S.CS[mi, lo]
        n = cnt.sum(1)
        add(f"rate{w}", (cnt + 4 * prior) / (n + 4)[:, None])
        add(f"cov{w}", n / np.maximum(ti - lo, 1))
    for w in NWIN:
        lo = np.maximum(ti - w, 0)
        cnt = S.NCS[mi, ti] - S.NCS[mi, lo]
        n = cnt.sum(1)
        add(f"nrate{w}", (cnt + 4 * prior) / (n + 4)[:, None])
        add(f"ncov{w}", n / (np.maximum(ti - lo, 1) * (S.esz[mi] - 1)))

    def wr(CS, a, b):
        lo, hi = np.maximum(ti - b, 0), np.maximum(ti - a, 0)
        cnt = CS[mi, hi] - CS[mi, lo]
        return (cnt[:, 1:].sum(1) + 0.8) / (cnt.sum(1) + 2.0)

    mr = (S.CS[mi, ti] + 4 * prior) / (S.CS[mi, ti].sum(1) + 4)[:, None]
    dist, lift, sup = [], [], []
    ar = np.arange(len(mi))
    for k in range(1, NL + 1):
        tab = S.LT[k - 1][mi, ti]
        pv = S.C[mi, np.maximum(ti - k, 0)]
        has = (ti - k >= 0) & (pv >= 0)
        tab_s = np.where(has[:, None], tab[ar, (pv > 0).astype(int)], tab.sum(1))
        dist.append((tab_s + 6 * mr) / (tab_s.sum(1) + 6)[:, None])
        sup.append(tab_s.sum(1) / 50.0)
        lift.append((tab[:, 1, 1:].sum(1) + 0.8) / (tab[:, 1].sum(1) + 2) - (tab[:, 0, 1:].sum(1) + 0.8) / (tab[:, 0].sum(1) + 2))
    add("mdist", np.concatenate(dist, 1))
    add("mlift", np.stack(lift, 1))
    add("msup", np.stack(sup, 1))
    d_own = wr(S.CS, 0, 32) - wr(S.CS, 32, 128)
    d_nb = wr(S.NCS, 0, 32) - wr(S.NCS, 32, 128)
    add("drift", np.stack([d_own, d_nb, np.abs(d_own), np.abs(d_nb)], 1))
    return np.concatenate(f, 1)


class Corpus:
    """One or more splits stacked: history examples, query rows, targets."""

    def __init__(self, splits, prior):
        self.prior = np.asarray(prior, np.float64)
        parts = dict(ep=[], mtype=[], sig=[], cut=[], OH=[], cond=[], tgt=[], ref=[], hmi=[], hti=[], hX=[], hY=[], qX=[], qY=[], qsoft=[])
        off_m, off_e = 0, 0
        for S in splits:
            m = np.arange(S.M)
            mi, ti = np.nonzero((S.C >= 0) & (np.arange(T)[None, :] >= TMIN))
            parts["hX"].append(feats(S, mi, ti, self.prior))
            parts["hY"].append(np.eye(5, dtype=np.float32)[S.C[mi, ti]])
            parts["hmi"].append(mi + off_m)
            parts["hti"].append(ti)
            parts["qX"].append(feats(S, m, S.cut, self.prior))
            if S.tgt is not None:
                parts["qY"].append(S.tgt)
                parts["qsoft"].append(np.ones(S.M, bool))
            elif S.out is not None:
                parts["qY"].append(np.eye(5)[S.out])
                parts["qsoft"].append(np.zeros(S.M, bool))
            else:
                parts["qY"].append(np.full((S.M, 5), np.nan))
                parts["qsoft"].append(np.zeros(S.M, bool))
            parts["ep"].append(S.ep + off_e)
            for k in ("mtype", "sig", "cut", "OH"):
                parts[k].append(getattr(S, k))
            parts["cond"].append(S.cond if S.cond is not None else np.full(S.M, "?", object))
            parts["tgt"].append(S.tgt if S.tgt is not None else np.full((S.M, 5), np.nan))
            parts["ref"].append(S.ref if S.ref is not None else np.full((S.M, 5), np.nan))
            off_m += S.M
            off_e += S.E
        for k, v in parts.items():
            setattr(self, k, np.concatenate(v))
        self.M = off_m
        self.E = off_e
        self.qid = [q for S in splits for q in S.qid]


def global_prior(splits):
    c = np.zeros(5)
    for S in splits:
        c += S.OH.sum((0, 1))
    return (c + 0.5) / (c.sum() + 2.5)


# ----------------------------------------------------------------------------- optimisers / nets
class Adam:
    def __init__(self, lr):
        self.lr, self.m, self.v, self.t = lr, {}, {}, 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            m = self.m.setdefault(k, np.zeros_like(params[k]))
            v = self.v.setdefault(k, np.zeros_like(params[k]))
            m *= 0.9
            m += 0.1 * grads[k]
            v *= 0.999
            v += 0.001 * grads[k] ** 2
            mh, vh = m / (1 - 0.9 ** self.t), v / (1 - 0.999 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


class SoftmaxModel:
    def __init__(self, F=None, prior=None):
        self.p = None if F is None else {"W": np.zeros((F, 5), np.float32), "b": np.log(prior).astype(np.float32)}

    def logits(self, X):
        return X @ self.p["W"] + self.p["b"]

    def fit(self, X, Y, w, l2, iters, lr=0.05):
        opt, sw = Adam(lr), w.sum()
        for _ in range(iters):
            P = softmax(self.logits(X))
            R = (P - Y) * w[:, None]
            g = {"W": X.T @ R / sw + 2 * l2 * self.p["W"], "b": R.sum(0) / sw}
            opt.step(self.p, {k: v.astype(np.float32) for k, v in g.items()})
        return self


class MLP:
    def __init__(self, F=None, H=96, prior=None, seed=0):
        if F is not None:
            r = np.random.RandomState(seed)
            self.p = {"W1": (r.randn(F, H) * np.sqrt(2.0 / F)).astype(np.float32), "b1": np.zeros(H, np.float32),
                      "W2": (r.randn(H, 5) * 0.05).astype(np.float32), "b2": np.log(prior).astype(np.float32)}

    @staticmethod
    def apply(p, X):
        return np.maximum(X @ p["W1"] + p["b1"], 0) @ p["W2"] + p["b2"]

    def logits(self, X):
        return self.apply(self.p, X)

    def fit(self, X, Y, w, epochs, bs, lr, l2, seed=0):
        r, opt, p = np.random.RandomState(seed), Adam(lr), self.p
        n = len(X)
        for _ in range(epochs):
            order = r.permutation(n)
            for s in range(0, n, bs):
                ix = order[s:s + bs]
                x, y, ww = X[ix], Y[ix], w[ix]
                a = x @ p["W1"] + p["b1"]
                h = np.maximum(a, 0)
                P = softmax(h @ p["W2"] + p["b2"])
                d = (P - y) * (ww / ww.sum())[:, None]
                dh = (d @ p["W2"].T) * (a > 0)
                g = {"W2": h.T @ d + 2 * l2 * p["W2"], "b2": d.sum(0), "W1": x.T @ dh + 2 * l2 * p["W1"], "b1": dh.sum(0)}
                opt.step(p, {k: v.astype(np.float32) for k, v in g.items()})
        return self


# ----------------------------------------------------------------------------- solution 1: tuned decayed frequency
def s1_predict(OH, cut, ts, prior_ts, hl, strength):
    age = (cut[:, None] - 1 - np.arange(T)[None, :]).astype(np.float64)
    W = np.where(age >= 0, 0.5 ** (age / hl), 0.0)
    counts = np.einsum("mt,mtc->mc", W, OH.astype(np.float64)) + strength * prior_ts[ts]
    return counts / counts.sum(1, keepdims=True)


def s1_fit(C, trm, cfg):
    ts = C.mtype * 3 + C.sig
    cnt = np.zeros((9, 5))
    for t in range(9):
        cnt[t] = C.OH[trm & (ts == t)].sum((0, 1))
    g = cnt.sum(0) + 0.5
    g /= g.sum()
    prior_ts = (cnt + 12 * g) / (cnt.sum(1, keepdims=True) + 12)
    best = None
    idx = np.nonzero(trm)[0]
    for hl in cfg["s1_hl"]:
        for st in cfg["s1_strength"]:
            P = s1_predict(C.OH[idx], C.cut[idx], ts[idx], prior_ts, hl, st)
            e = ((P - C.qY[idx]) ** 2).sum(1).mean()
            if best is None or e < best[0]:
                best = (e, hl, st)
    return {"prior_ts": prior_ts, "hl": best[1], "strength": best[2]}


# ----------------------------------------------------------------------------- solution 3: per-machine adaptation of S2 logits
def s3_predict(C, sel, base_logits, qbase, cfg):
    """Fit per-machine lag adjustments on its own history (ticks < cutoff), shrunk toward the global model."""
    ex = np.nonzero(np.isin(C.hmi, sel))[0]
    row = np.searchsorted(sel, C.hmi[ex])
    order = np.argsort(row, kind="stable")
    ex, row = ex[order], row[order]
    R = len(sel)
    cnt = np.bincount(row, minlength=R)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    pos = np.arange(len(ex)) - start[row]
    N = cnt.max()
    sl = LAYOUT["wl"]
    Fx = np.concatenate([np.ones((len(ex), 1), np.float32), C.hX[ex][:, sl]], 1)
    nf = Fx.shape[1]
    Fp = np.zeros((R, N, nf), np.float32)
    Zp = np.zeros((R, N, 5), np.float32)
    Yp = np.zeros((R, N, 5), np.float32)
    Wp = np.zeros((R, N), np.float32)
    Fp[row, pos], Zp[row, pos], Yp[row, pos] = Fx, base_logits(C.hX[ex]), C.hY[ex]
    age = C.cut[sel][row] - 1 - C.hti[ex]
    Wp[row, pos] = 0.5 ** (age / cfg["s3_hl"])
    A = {"A": np.zeros((R, nf, 5), np.float32)}
    reg = np.ones((1, nf, 1), np.float32)
    reg[0, 0] = 0.1
    opt = Adam(cfg["s3_lr"])
    for _ in range(cfg["s3_steps"]):
        P = softmax((Zp + np.einsum("rnf,rfc->rnc", Fp, A["A"])).reshape(-1, 5)).reshape(R, N, 5)
        G = np.einsum("rnf,rnc->rfc", Fp, (P - Yp) * Wp[..., None]) + 2 * cfg["s3_l2"] * reg * A["A"]
        opt.step(A, {"A": G})
    fq = np.concatenate([np.ones((R, 1), np.float32), C.qX[sel][:, sl]], 1)
    return softmax(qbase + np.einsum("rf,rfc->rc", fq, A["A"]))


# ----------------------------------------------------------------------------- fitting / predicting all models
NB_BLOCKS = ("nrate8", "ncov8", "nrate32", "ncov32", "nrate128", "ncov128", "nfrac")


def cols_for(kind):
    keep = [k for k in LAYOUT if kind == "B" or k not in NB_BLOCKS]
    return np.concatenate([np.arange(LAYOUT[k].start, LAYOUT[k].stop) for k in keep])


def _ens_logits(nets, cols, X):
    P = np.mean([softmax(MLP.apply(p, X[:, cols])) for p in nets], 0)
    return np.log(np.clip(P, 1e-9, None))


def fit_models(C, trm, cfg, verbose=False):
    hm = trm[C.hmi]
    X = np.concatenate([C.hX[hm], C.qX[trm]])
    Y = np.concatenate([C.hY[hm], C.qY[trm].astype(np.float32)])
    w = np.concatenate([np.ones(hm.sum(), np.float32), np.where(C.qsoft[trm], cfg["wq"], 1.0).astype(np.float32)])
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-6] = 1.0
    Xs = ((X - mu) / sd).astype(np.float32)
    M = {"mu": mu, "sd": sd, "s1": s1_fit(C, trm, cfg), "cfg": cfg, "nets": {}}
    for kind, H, seeds in (("A", cfg["h_a"], cfg["seeds_a"]), ("B", cfg["h_b"], cfg["seeds_b"])):
        cols = cols_for(kind)
        M["nets"][kind] = [MLP(len(cols), H, C.prior, s).fit(Xs[:, cols], Y, w, cfg["mlp_epochs"], cfg["mlp_bs"], cfg["mlp_lr"], cfg["mlp_l2"], s).p
                           for s in seeds]
    return M


def predict_models(M, C, sel):
    """Return {s1..s4: (len(sel),5)} for machines `sel` (sorted ints)."""
    cfg = M["cfg"]
    std = lambda X: ((X - M["mu"]) / M["sd"]).astype(np.float32)
    ts = C.mtype * 3 + C.sig
    Xq = std(C.qX[sel])
    colsA, colsB = cols_for("A"), cols_for("B")
    zA = _ens_logits(M["nets"]["A"], colsA, Xq)
    zB = _ens_logits(M["nets"]["B"], colsB, Xq)
    out = {
        "s1": s1_predict(C.OH[sel], C.cut[sel], ts[sel], M["s1"]["prior_ts"], M["s1"]["hl"], M["s1"]["strength"]),
        "s2": softmax(zA),
        "s3": softmax(zB),
        "s4": s3_predict(C, sel, lambda X: _ens_logits(M["nets"]["A"], colsA, std(X)), zA, cfg),
    }
    return {k: np.clip(v, 1e-6, None) / np.clip(v, 1e-6, None).sum(1, keepdims=True) for k, v in out.items()}


# ----------------------------------------------------------------------------- solution 5: regime gate
def gate_feats(qX):
    L = LAYOUT
    g = [np.ones((len(qX), 1), np.float32), qX[:, L["cov64"]], qX[:, L["cov128"]], (qX[:, L["cov64"]] < 0.9).astype(np.float32),
         qX[:, L["drift"]][:, 2:4], qX[:, L["ncov8"]], qX[:, L["type"]][:, :2]]
    return np.concatenate(g, 1).astype(np.float64)


def fit_gate(P, Y, g, l2=1e-2, iters=400, lr=0.05):
    K = P.shape[0]
    Wg = {"W": np.zeros((g.shape[1], K))}
    opt = Adam(lr)
    for _ in range(iters):
        w = softmax(g @ Wg["W"])
        p = np.einsum("mk,kmc->mc", w, P)
        dw = 2 * np.einsum("mc,kmc->mk", p - Y, P)
        du = w * (dw - (w * dw).sum(1, keepdims=True))
        opt.step(Wg, {"W": g.T @ du / len(g) + 2 * l2 * Wg["W"]})
    return Wg["W"]


def apply_gate(Wg, P, g):
    w = softmax(g @ Wg)
    return np.einsum("mk,kmc->mc", w, P)


# ----------------------------------------------------------------------------- evaluation
def contest(P, C, max_points):
    """Official-style score on a corpus with true targets and pooled reference."""
    ratios, rows = [], {}
    for c in ("stable", "short_history", "drift", "shared_change"):
        ids = np.nonzero(C.cond == c)[0]

        def err(X):
            e = ((X[ids] - C.tgt[ids]) ** 2).sum(1)
            return np.mean([e[C.ep[ids] == u].mean() for u in np.unique(C.ep[ids])])

        e, b = err(P), err(C.ref)
        ratios.append(e / max(b, 1e-9))
        rows[c] = round(ratios[-1], 3)
    R = float(np.mean(ratios))
    return {"points": max_points / (1 + R), "R": R, "by_cond": rows}


DEFAULT_CFG = dict(wq=5.0, h_a=64, h_b=96, seeds_a=(0, 1, 2, 3, 4), seeds_b=(10, 11, 12, 13, 14), mlp_epochs=12, mlp_bs=1024, mlp_lr=2e-3, mlp_l2=1e-5,
                   s3_l2=8.0, s3_steps=80, s3_lr=0.03, s3_hl=64.0, s1_hl=(12, 24, 48, 96, 200, 1e9), s1_strength=(4, 8, 16, 32), seed=0)


# ----------------------------------------------------------------------------- checkpoint I/O
def save_ckpt(path, M, Wg, prior):
    a = {"prior": prior, "mu": M["mu"], "sd": M["sd"], "s1_prior_ts": M["s1"]["prior_ts"],
         "s1_params": np.array([M["s1"]["hl"], M["s1"]["strength"]]), "Wg": Wg}
    for kind, nets in M["nets"].items():
        a[f"n_{kind}"] = np.array(len(nets))
        for i, p in enumerate(nets):
            for k, v in p.items():
                a[f"{kind}{i}_{k}"] = v.astype(np.float64)
    cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in M["cfg"].items()}
    a["cfg_json"] = np.frombuffer(json.dumps(cfg).encode(), dtype=np.uint8)
    np.savez_compressed(path, **a)


def load_ckpt(path):
    z = np.load(path, allow_pickle=False)
    cfg = json.loads(bytes(z["cfg_json"]).decode())
    nets = {kind: [{k: z[f"{kind}{i}_{k}"] for k in ("W1", "b1", "W2", "b2")} for i in range(int(z[f"n_{kind}"]))] for kind in ("A", "B")}
    M = {"mu": z["mu"], "sd": z["sd"], "cfg": cfg, "nets": nets,
         "s1": {"prior_ts": z["s1_prior_ts"], "hl": float(z["s1_params"][0]), "strength": float(z["s1_params"][1])}}
    return M, z["Wg"], z["prior"]
