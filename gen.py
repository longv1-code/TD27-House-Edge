"""Generative switching model (numpy only). Per machine x lag pair: 4-state filter over (d = which XOR cell is high-loss,
sigma = size-state sign). size state s = big(t-a) XOR sigma. Emissions: P(loss)=TL[type,high,s,signal];
P(1x|win)=R1[type,s]; big sizes by MIX[type]. Pairs averaged by likelihood x prior. Uses ticks < t only."""
import itertools, numpy as np
T = 260
PAIRS = [(a, b) for a in range(1, 9) for b in range(a + 1, 9)]
STATES = [(0, 0), (0, 1), (1, 0), (1, 1)]


def run_pair(C, mtype, sig, a, b, P, eps_d, eps_s, collect=None, wt=None):
    """Forward filter for one pair over all machines. Returns per-machine cumulative log-lik grid LLg (M,T+1)
    and predictive class probs grid PRg (M,T+1,5) for tick t from ticks < t."""
    M = C.shape[0]
    TL, R1, MIX = P["TL"], P["R1"], P["MIX"]
    pi = np.full((M, 4), 0.25)
    d = np.array([s[0] for s in STATES]); sg = np.array([s[1] for s in STATES])
    Td = np.array([[1 - eps_d if d[i] == d[j] else eps_d for j in range(4)] for i in range(4)])
    Ts = np.array([[1 - eps_s if sg[i] == sg[j] else eps_s for j in range(4)] for i in range(4)])
    Tr = Td * Ts
    LL = np.zeros(M)
    LLg = np.zeros((M, T + 1)); PRg = np.zeros((M, T + 1, 5))
    ar = np.arange(M)
    for t in range(T + 1):
        if t >= b and t < T + 1:
            ca, cb = C[:, t - a], C[:, t - b]
            known = (ca >= 0) & (cb >= 0)
        else:
            ca = cb = np.full(M, -1); known = np.zeros(M, bool)
        eq = ((ca > 0) == (cb > 0)).astype(int)
        big = (ca >= 2).astype(int)
        high = eq[:, None] ^ d[None, :]          # (M,4)
        s = big[:, None] ^ sg[None, :]
        pl = TL[mtype[:, None], high, s, sig[:, None]]   # (M,4)
        r1 = R1[mtype[:, None], s]
        mixm = MIX[mtype]                                 # (M,3)
        cls = np.empty((M, 4, 5))
        cls[:, :, 0] = pl; cls[:, :, 1] = (1 - pl) * r1
        cls[:, :, 2:] = ((1 - pl) * (1 - r1))[:, :, None] * mixm[:, None, :]
        # unknown context: use state-free marginal (average of the 4 x both contexts) -> approximate by uniform over high,s
        unk = ~known
        if unk.any():
            pl_u = TL[mtype][:, :, :, :].mean((1, 2))[np.arange(M), sig]
            r_u = R1[mtype].mean(1)
            cu = np.empty((M, 5)); cu[:, 0] = pl_u; cu[:, 1] = (1 - pl_u) * r_u; cu[:, 2:] = ((1 - pl_u) * (1 - r_u))[:, None] * mixm
            cls[unk] = cu[unk][:, None, :]
        PRg[:, t] = np.einsum("mk,mkc->mc", pi, cls)
        LLg[:, t] = LL
        if t >= T:
            break
        y = C[:, t]
        obs = (y >= 0) & known
        lik = np.where(obs[:, None], cls[ar, :, np.maximum(y, 0)], 1.0)
        mix = (pi * lik).sum(1)
        LL = LL + np.where(obs, np.log(mix), 0.0)
        post = pi * lik / mix[:, None]
        if collect is not None:
            w = (obs * wt)[:, None] * post            # soft counts for EM
            np.add.at(collect["n"], (mtype[:, None].repeat(4, 1), high, s, sig[:, None].repeat(4, 1)), w)
            np.add.at(collect["k"], (mtype[:, None].repeat(4, 1), high, s, sig[:, None].repeat(4, 1)), w * (y == 0)[:, None])
            win = (y > 0)[:, None]
            np.add.at(collect["rn"], (mtype[:, None].repeat(4, 1), s), w * win)
            np.add.at(collect["rk"], (mtype[:, None].repeat(4, 1), s), w * (y == 1)[:, None])
        pi = post @ Tr
    return LLg, PRg


EPS_LIST = ((0.002, 0.0005), (0.02, 0.0005))
OFF = dict(hl=32.0, tau_e=0.4, tau_m=0.2)


def predict(S, P, eps_list=EPS_LIST, eps_prior=(0.5, 0.5), collect=None):
    """Bayesian average over pairs and switch rates. Returns (M,5) predictions at the cutoff."""
    C = S.C.astype(np.int64); M = C.shape[0]; ar = np.arange(M)
    logw, preds = [], []
    for e, (ed, es) in enumerate(eps_list):
        for j, (a, b) in enumerate(PAIRS):
            LLg, PRg = run_pair(C, S.mtype, S.sig, a, b, P, ed, es)
            logw.append(LLg[ar, S.cut] + P["PAIR_LP"][j] + np.log(eps_prior[e]))
            preds.append(PRg[ar, S.cut])
    logw = np.stack(logw, 1); preds = np.stack(preds, 1)
    w = np.exp(logw - logw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    out = np.einsum("mj,mjc->mc", w, preds)
    if collect is not None:   # EM soft counts weighted by pair posterior
        k = 0
        for e, (ed, es) in enumerate(eps_list):
            for j, (a, b) in enumerate(PAIRS):
                if (w[:, k] > 0.02).any():
                    run_pair(C, S.mtype, S.sig, a, b, P, ed, es, collect=collect, wt=w[:, k])
                k += 1
    return out, w


def predict_grid(S, P, eps_list, eps_prior):
    """Cutoff prediction (M,5) plus per-tick BMA loss-probability grid (M,T+1) using cutoff pair weights,
    and a mask of informative observed ticks."""
    C = S.C.astype(np.int64); M = C.shape[0]; ar = np.arange(M)
    logw = []
    for e, (ed, es) in enumerate(eps_list):
        for j, (a, b) in enumerate(PAIRS):
            LLg, _ = run_pair(C, S.mtype, S.sig, a, b, P, ed, es)
            logw.append(LLg[ar, S.cut] + P["PAIR_LP"][j] + np.log(eps_prior[e]))
    logw = np.stack(logw, 1)
    w = np.exp(logw - logw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    G = np.zeros((M, T + 1)); out = np.zeros((M, 5)); k = 0
    for e, (ed, es) in enumerate(eps_list):
        for j, (a, b) in enumerate(PAIRS):
            _, PRg = run_pair(C, S.mtype, S.sig, a, b, P, ed, es)   # no skip threshold: keeps output continuous
            G += w[:, k:k + 1] * PRg[:, :, 0]
            out += w[:, k:k + 1] * PRg[ar, S.cut]
            k += 1
    obs = np.zeros((M, T + 1), bool)
    obs[:, 8:T] = C[:, 8:T] >= 0
    obs &= np.arange(T + 1)[None, :] < S.cut[:, None]
    y = np.zeros((M, T + 1)); y[:, :T] = C == 0
    return out, G, obs, y


def offsets(S, G, obs, y, hl, tau_e, tau_m, iters=4):
    """Episode then machine logit offsets from residuals of ticks < cutoff (decay half-life hl, 0 = none)."""
    M = len(S.cut)
    age = S.cut[:, None] - 1 - np.arange(T + 1)[None, :]
    kw = np.where(obs, 1.0 if hl <= 0 else 2.0 ** (-np.maximum(age, 0) / hl), 0.0)
    z = np.log(np.clip(G, 1e-6, 1 - 1e-6) / (1 - np.clip(G, 1e-6, 1 - 1e-6)))
    th_e = np.zeros(S.E)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(z + th_e[S.ep][:, None])))
        g = np.zeros(S.E); h = np.zeros(S.E)
        np.add.at(g, S.ep, (kw * (y - p)).sum(1)); np.add.at(h, S.ep, (kw * p * (1 - p)).sum(1))
        th_e += (g - th_e / tau_e ** 2) / (h + 1 / tau_e ** 2)
    th_m = np.zeros(M)
    if tau_m > 0:
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(z + th_e[S.ep][:, None] + th_m[:, None])))
            g = (kw * (y - p)).sum(1); h = (kw * p * (1 - p)).sum(1)
            th_m += (g - th_m / tau_m ** 2) / (h + 1 / tau_m ** 2)
    return th_e[S.ep] + th_m


def shift(p, dz):
    lg = np.log(p[:, 0] / (1 - p[:, 0])) + dz
    L = 1 / (1 + np.exp(-lg))
    out = p.copy(); out[:, 0] = L
    out[:, 1:] = p[:, 1:] / p[:, 1:].sum(1, keepdims=True) * (1 - L)[:, None]
    return out


def init_params(splits):
    MIX = np.ones((3, 3))
    for S in splits:
        for t in range(3):
            c = S.C[S.mtype == t]; MIX[t] += np.bincount(c[c >= 2], minlength=5)[2:]
    MIX /= MIX.sum(1, keepdims=True)
    TL = np.zeros((3, 2, 2, 3))
    TL[:, 1, :, :] = 0.85; TL[:, 0, :, :] = 0.1
    R1 = np.array([[0.4, 0.7], [0.25, 0.5], [0.15, 0.35]])
    return {"TL": TL, "R1": R1, "MIX": MIX, "PAIR_LP": np.zeros(len(PAIRS))}


def em(splits, P, iters=3, log=print):
    for it in range(iters):
        col = {"n": np.full((3, 2, 2, 3), 1.0), "k": np.zeros((3, 2, 2, 3)), "rn": np.full((3, 2), 1.0), "rk": np.zeros((3, 2))}
        col["k"] += P["TL"]; col["rk"] += P["R1"]
        pc = np.full(len(PAIRS), 0.5)
        for S in splits:
            _, w = predict(S, P, collect=col)
            ne = len(w[0]) // len(PAIRS)
            wp = w.reshape(len(w), ne, len(PAIRS)).sum(1)
            use = np.ones(S.M, bool) if S.cond is None else (S.cond != "short_history")
            pc += wp[use].sum(0)
        P = dict(P, TL=col["k"] / col["n"], R1=col["rk"] / col["rn"], PAIR_LP=np.log(pc / pc.sum()))
        # keep labelling: high cell = higher loss, s=1 = higher 1x share
        log(f"EM iter {it}: R1={np.round(P['R1'],3).tolist()} TL[steady]={np.round(P['TL'][0],3).tolist()}")
    return P


def load_params(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in ("TL", "R1", "MIX", "PAIR_LP")}


def full_predict(S, P):
    """Cutoff predictions (M,5): filter + Bayesian pair/switch averaging + decayed episode/machine logit offsets."""
    out, G, obs, y = predict_grid(S, P, EPS_LIST, (0.5, 0.5))
    p = shift(out, offsets(S, G, obs, y, OFF["hl"], OFF["tau_e"], OFF["tau_m"]))
    p = np.clip(p, 1e-6, None)
    return p / p.sum(1, keepdims=True)
