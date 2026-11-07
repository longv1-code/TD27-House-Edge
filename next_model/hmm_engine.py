import numpy as np
from scipy.special import expit,logsumexp
from slot_forecasting.schema import index_history

class HMM:
    def __init__(self,queries,history,patterns,centers):
        self.queries=queries;self.patterns=patterns;self.centers=centers
        n=len(queries);T=max(q.cutoff for q in queries)+1;p=len(patterns)
        self.y=np.full((n,T),-1.);self.bits=np.full((n,T,p),-1.)
        ix=index_history(queries,history)
        for i,q in enumerate(queries):
            for r in ix.get((q.episode_id,q.machine_id),[]):self.y[i,r.tick]=float(r.payout>0)
            for j,pat in enumerate(patterns):
                for t in range(T):
                    if t>=max(pat)+1:
                        b=self.y[i,t-1-np.array(pat)]
                        if (b>=0).all():self.bits[i,t,j]=b.sum()%2
        self.cut=np.array([q.cutoff for q in queries]);self.c=np.array([centers[q.machine_type+'|'+str(q.signal)] for q in queries])

    def predict(self,hazard=.005,spread=.6,amplitude_scale=1,forget=1):
        n,T,P=self.bits.shape
        # Grid integrates uncertainty about machine-specific intercept and amplitude.
        offsets=np.array([-1.5,-.75,0,.75,1.5])*spread
        scales=np.array([.65,1.,1.35])*amplitude_scale
        grid=np.array([(o,a) for o in offsets for a in scales]);G=len(grid)
        mid=self.c.mean(1);amp=(self.c[:,1]-self.c[:,0])/2
        lo=expit(mid[:,None]+grid[None,:,0]-amp[:,None]*grid[None,:,1])
        hi=expit(mid[:,None]+grid[None,:,0]+amp[:,None]*grid[None,:,1])
        gw=np.exp(-.5*(grid[:,0]/max(spread,.01))**2-.5*((grid[:,1]/amplitude_scale-1)/.3)**2);gw/=gw.sum()
        phase=np.full((n,P,G),.5);logw=np.broadcast_to(np.log(gw)[None,None,:],(n,P,G)).copy()-np.log(P)
        result=np.zeros(n)
        for t in range(T):
            phase=phase*(1-2*hazard)+hazard
            b=self.bits[:,t,:,None]
            e0=np.where(b<0,(lo[:,None]+hi[:,None])/2,np.where(b>0,hi[:,None],lo[:,None]))
            e1=np.where(b<0,(lo[:,None]+hi[:,None])/2,np.where(b>0,lo[:,None],hi[:,None]))
            need=self.cut==t
            if need.any():
                ll=logw[need];w=np.exp(ll-logsumexp(ll,axis=(1,2))[:,None,None])
                result[need]=(w*((1-phase[need])*e0[need]+phase[need]*e1[need])).sum((1,2))
            active=(self.y[:,t]>=0)&(t<self.cut)
            obs=self.y[:,t,None,None]
            e0=np.where(obs>0,e0,1-e0);e1=np.where(obs>0,e1,1-e1)
            likelihood=(1-phase)*e0+phase*e1
            phase=np.where(active[:,None,None],phase*e1/likelihood,phase)
            logw=np.where(active[:,None,None],logw*forget+np.log(likelihood),logw)
        return result
