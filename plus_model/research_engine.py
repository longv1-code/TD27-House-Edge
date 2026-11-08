"""Bayesian lag-mixture experiments; all features use strictly earlier ticks."""
import numpy as np
from scipy.special import betaln, logsumexp
from itertools import combinations
from slot_forecasting.schema import index_history, PAYOUTS

PATTERNS=[()] + [(i,) for i in range(8)] + list(combinations(range(8),2)) + list(combinations(range(8),3))

class Engine:
    def __init__(self,queries,history,base):
        self.queries=queries; self.base=base; self.cache={};self.ev={}
        indexed=index_history(queries,history);self.rows=[];self.priors=[]
        for q in queries:
            seq=np.full(q.cutoff+1,-1,dtype=int)
            for r in indexed.get((q.episode_id,q.machine_id),[]):seq[r.tick]=PAYOUTS.index(r.payout)
            ticks=np.where(seq[:-1]>=0)[0]; ts=np.r_[ticks,q.cutoff]
            codes=np.zeros((len(ts),len(PATTERNS)),dtype=int)
            bits=np.full((len(ts),8),-1,dtype=int)
            for k in range(8):
                valid=ts>=k+1; bits[valid,k]=np.where(seq[ts[valid]-k-1]>=0,(seq[ts[valid]-k-1]>0).astype(int),-1)
            for j,p in enumerate(PATTERNS[1:],1):
                b=bits[:,p]; codes[:,j]=np.where((b>=0).all(1),b.sum(1)%2,-1)
            self.rows.append((q.cutoff-1-ticks,seq[ticks],codes[:-1],codes[-1]))
            self.priors.append(base.context_priors.get(base._key(q),base.global_prior))
        self.priors=np.array(self.priors); self.wanted=np.array([r[3] for r in self.rows])

    def counts(self,half,chunk=0):
        key=(half,chunk)
        if key in self.cache:return self.cache[key]
        n=len(self.rows);bins=9 if chunk else 1
        out=np.zeros((n,bins,len(PATTERNS),2,5))
        for i,(ages,y,codes,wanted) in enumerate(self.rows):
            w=2.**(-ages/half);b=ages//chunk if chunk else np.zeros_like(ages)
            for s in range(2):
                for k in range(5):
                    for z in np.unique(b):out[i,z,:,s,k]=((codes[(b==z)&(y==k)]==s)*w[(b==z)&(y==k),None]).sum(0)
        self.cache[key]=out
        return out

    def evidence(self,half,alpha,chunk=0):
        key=half,alpha,chunk
        if key in self.ev:return self.ev[key]
        c=self.counts(half,chunk);a=(alpha*self.priors[:,0])[:,None,None];b=(alpha*(1-self.priors[:,0]))[:,None,None]
        lose=c[...,0];win=c[...,1:].sum(-1)
        ev=(betaln(a[...,None]+lose,b[...,None]+win).sum(-1)-betaln(a+lose.sum(-1),b+win.sum(-1))-betaln(a,b)).sum(1)
        self.ev[key]=ev;return ev

    def prediction(self,cfg,pattern_prior=None,templates=None):
        half=cfg.get('half',12);alpha=cfg.get('alpha',.5)
        counts=self.counts(half)[:,0];ev=self.evidence(half,alpha).copy()
        ev+=cfg.get('long',.25)*self.evidence(96,alpha)
        ev+=cfg.get('chunk',1)*self.evidence(128,alpha,32)
        orders=np.array([len(p) for p in PATTERNS]); ev=ev/cfg.get('temp',.25)-.5*orders
        if pattern_prior is not None:ev+=cfg.get('prior_weight',1)*np.log(pattern_prior)
        ev[:,orders>cfg.get('order',2)]=-1e10
        ev[self.wanted<0]=-1e10
        weights=np.exp(ev-logsumexp(ev,axis=1)[:,None])
        idx=np.maximum(0,self.wanted)
        c=np.take_along_axis(counts,idx[:,:,None,None],axis=2)[:,:,0,:]
        basec=counts[:,0,0];base=(basec+alpha*self.priors)/(basec.sum(1)[:,None]+alpha)
        wins=c[...,1:].sum(-1);ws=cfg.get('ws',.25)
        win=(wins+ws*(1-base[:,0,None]))/(c.sum(-1)+ws)
        posprior=self.priors[:,1:]/self.priors[:,1:].sum(1)[:,None];ps=cfg.get('ps',64)
        amount=(c[...,1:]+ps*posprior[:,None,:])/(wins[:,:,None]+ps)
        pwin=(weights*win).sum(1);power=cfg.get('power',1.3)
        adjusted=pwin**power/(pwin**power+(1-pwin)**power)
        positive=(weights[:,:,None]*win[:,:,None]*amount).sum(1)/np.maximum(pwin[:,None],1e-10)
        if templates is not None:
            positive=self.positive(cfg,weights,templates)
        pred=np.c_[1-adjusted,adjusted[:,None]*positive]
        blend=cfg.get('blend',.1)
        return pred*(1-blend)+self.priors*blend

    def positive(self,cfg,weights,templates):
        half=cfg.get('pos_half',24);out=[]
        # Integrate over a two-component positive-payout regime in each context.
        grid=np.linspace(0,1,41);strength=cfg.get('pos_strength',1)
        priorw=np.maximum(grid*(1-grid),.001)**(strength-1)
        for i,q in enumerate(self.queries):
            tmpl=np.array(templates[q.machine_type]);mix=grid[:,None]*tmpl[1]+(1-grid[:,None])*tmpl[0]
            c=self.counts(half)[i,0,:,:,1:];wanted=np.maximum(self.wanted[i],0)
            obs=c[np.arange(len(PATTERNS)),wanted]
            ll=obs@np.log(mix).T+np.log(priorw)
            post=np.exp(ll-logsumexp(ll,axis=1)[:,None]);p=post@mix
            out.append(weights[i]@p)
        return np.array(out)

def learn_prior(engine,alpha=.5):
    ev=engine.evidence(32,alpha)+engine.evidence(128,alpha,32)
    orders=np.array([len(p) for p in PATTERNS]);ev[:,orders>2]=-1e10
    prior=np.ones(len(PATTERNS));prior[orders>2]=1e-8;prior/=prior.sum()
    for _ in range(20):
        scores=ev+np.log(prior);post=np.exp(scores-logsumexp(scores,axis=1)[:,None])
        prior=(post.sum(0)+2)/(len(post)+2*len(PATTERNS));prior[orders>2]=1e-8;prior/=prior.sum()
    return prior
