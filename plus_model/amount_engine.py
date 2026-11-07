import numpy as np
from scipy.special import logsumexp
from slot_forecasting.schema import index_history,PAYOUTS

class Amount:
    def __init__(self,queries,history,templates,patterns):
        self.queries=queries;self.patterns=patterns;self.templates=np.array([templates[q.machine_type] for q in queries]);n=len(queries);T=max(q.cutoff for q in queries)+1
        self.y=np.full((n,T),-1,dtype=int);self.bits=np.full((n,T,len(patterns)),-1,dtype=int);self.cut=np.array([q.cutoff for q in queries]);ix=index_history(queries,history)
        for i,q in enumerate(queries):
            for r in ix.get((q.episode_id,q.machine_id),[]):self.y[i,r.tick]=PAYOUTS.index(r.payout)
            for j,pat in enumerate(patterns):
                ts=np.arange(max(pat,default=-1)+1,T)
                if len(pat):
                    vals=self.y[i,ts[:,None]-1-np.array(pat)[None,:]];b=(vals>0).sum(1)%2;b[(vals<0).any(1)]=-1;self.bits[i,ts,j]=b
                else:self.bits[i,:,j]=0

    def predict(self,hazard=.001,temperature=1,prior=None,forget=1,return_evidence=False,bit_probs=None):
        n,T,P=self.bits.shape;phase=np.full((n,P),.5);logw=np.zeros((n,P));out=np.zeros((n,4));evidence=None
        if prior is not None:logw+=np.log(prior)[None,:]
        self.last_details={'pattern_weights':np.zeros((n,P)),'state1':np.zeros((n,P)),'phase':np.zeros((n,P))}
        for t in range(T):
            phase=phase*(1-2*hazard)+hazard;b=self.bits[:,t]
            need=self.cut==t
            if need.any():
                w=np.exp(logw[need]/temperature-logsumexp(logw[need]/temperature,axis=1)[:,None])
                state1=(phase[need]+bit_probs[need,t]*(1-2*phase[need])) if bit_probs is not None else np.where(b[need]<0,.5,np.where(b[need]>0,1-phase[need],phase[need]))
                s=(w*state1).sum(1);out[need]=(1-s[:,None])*self.templates[need,0]+s[:,None]*self.templates[need,1]
                self.last_details['pattern_weights'][need]=w;self.last_details['state1'][need]=state1;self.last_details['phase'][need]=phase[need]
            active=(self.y[:,t]>0)&(t<self.cut);k=np.maximum(0,self.y[:,t]-1)
            low=self.templates[np.arange(n),0,k][:,None];hi=self.templates[np.arange(n),1,k][:,None]
            if bit_probs is not None:
                probability=bit_probs[:,t];e0=(1-probability)*low+probability*hi;e1=(1-probability)*hi+probability*low
            else:e0=np.where(b<0,(low+hi)/2,np.where(b>0,hi,low));e1=np.where(b<0,(low+hi)/2,np.where(b>0,low,hi))
            likelihood=(1-phase)*e0+phase*e1
            phase=np.where(active[:,None],phase*e1/likelihood,phase)
            logw=np.where(active[:,None],forget*logw+np.log(likelihood),logw)
        self.last_details['log_evidence']=logw.copy()
        return (out,logw) if return_evidence else out
