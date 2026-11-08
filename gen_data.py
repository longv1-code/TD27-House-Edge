"""Minimal loader for the generative model: payout classes per machine/tick (-1 = unknown), ticks < cutoff only."""
import csv
from pathlib import Path
import numpy as np
PIDX = {0: 0, 1: 1, 2: 2, 5: 3, 20: 4}
TYPES = ("steady", "balanced", "jackpot")


def _rows(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


class Split:
    def __init__(self, d, T=260):
        d = Path(d)
        Q = _rows(d / "queries.csv")
        self.qid = [q["query_id"] for q in Q]
        self.M = len(Q)
        eps = {}
        self.ep = np.array([eps.setdefault(q["episode_id"], len(eps)) for q in Q])
        self.E = len(eps)
        self.mtype = np.array([TYPES.index(q["machine_type"]) for q in Q])
        self.sig = np.array([int(q["signal"]) + 1 for q in Q])
        self.cut = np.array([int(q["cutoff"]) for q in Q])
        mid = {(q["episode_id"], q["machine_id"]): i for i, q in enumerate(Q)}
        C = -np.ones((self.M, T), np.int64)
        for r in _rows(d / "history.csv"):
            i = mid.get((r["episode_id"], r["machine_id"]))
            if i is not None:
                C[i, int(r["tick"])] = PIDX[int(r["payout"])]
        C[np.arange(T)[None, :] >= self.cut[:, None]] = -1
        self.C = C
        self.cond = None
        if (d / "conditions.csv").exists():
            pos = {q: i for i, q in enumerate(self.qid)}
            self.cond = np.empty(self.M, object)
            for r in _rows(d / "conditions.csv"):
                self.cond[pos[r["query_id"]]] = r["condition"]
