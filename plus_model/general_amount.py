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

    def soft_contexts(self,centers,strength=10,half=10000):
        from scipy.special import expit
        n,T=self.y.shape
        lohi=np.array([centers[q.machine_type+'|'+str(q.signal)] for q in self.queries])
        win=(expit(lohi[:,0])+expit(lohi[:,1]))/2
        positive=self.templates.mean(axis=1)
        prior=np.column_stack((win,win*(1-positive[:,0]),win*positive[:,0],win*positive[:,2:].sum(1)))
        selected=np.stack((self.y>0,self.y>=2,self.y==1,self.y>=3),axis=2).astype(float)
        total=np.zeros(n);success=np.zeros((n,4));rates=np.zeros((n,T,4));decay=2.**(-1/half)
        for t in range(T):
            rates[:,t]=(success+strength*prior)/(total[:,None]+strength)
            active=(self.y[:,t]>=0)&(t<self.cut)
            success=decay*success+selected[:,t]*active[:,None];total=decay*total+active
        times=np.arange(T);out=np.zeros_like(self.bits,dtype=float)
        for j,(kind,pattern) in enumerate(self.patterns):
            product=np.ones((n,T))
            for lag in pattern:
                tick=times-lag-1;value=self.y[:,np.maximum(tick,0)]
                known=(tick[None,:]>=0)&(value>=0)
                bit=selected[:,np.maximum(tick,0),kind]
                probability=np.where(known,bit,rates[:,:,kind]);product*=1-2*probability
            out[:,:,j]=(1-product)/2
        return out
