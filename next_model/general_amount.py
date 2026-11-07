import numpy as np
from itertools import combinations
from .amount_engine import Amount
from slot_forecasting.schema import index_history,PAYOUTS

class GeneralAmount(Amount):
    def __init__(self,queries,history,templates,patterns=None):
        if patterns is None:patterns=[(0,())]+[(kind,p) for kind in range(4) for k in (1,2) for p in combinations(range(8),k)]
        self.queries=queries;self.patterns=patterns;self.templates=np.array([templates[q.machine_type] for q in queries]);n=len(queries);T=max(q.cutoff for q in queries)+1
        self.y=np.full((n,T),-1,dtype=int);self.bits=np.full((n,T,len(patterns)),-1,dtype=int);self.cut=np.array([q.cutoff for q in queries]);ix=index_history(queries,history)
        for i,q in enumerate(queries):
            for r in ix.get((q.episode_id,q.machine_id),[]):self.y[i,r.tick]=PAYOUTS.index(r.payout)
            for j,(kind,pat) in enumerate(patterns):
                ts=np.arange(max(pat,default=-1)+1,T)
                if pat:
                    vals=self.y[i,ts[:,None]-1-np.array(pat)[None,:]]
                    selected=vals>0 if kind==0 else vals>=2 if kind==1 else vals==1 if kind==2 else vals>=3
                    b=selected.sum(1)%2;b[(vals<0).any(1)]=-1;self.bits[i,ts,j]=b
                else:self.bits[i,:,j]=0
