import numpy as np
from scipy.special import expit,logsumexp,logit
from slot_forecasting.schema import index_history,PAYOUTS
from .amount_features import features as old_features
from .hmm_engine import HMM


def fit_population(queries,targets,mask):
    y=np.array(targets);templates={};centers={}
    for typ in sorted({q.machine_type for q in queries}):
        m=mask&np.array([q.machine_type==typ for q in queries])
        positive=y[m,1:]/(1-y[m,0,None]);threshold=(positive[:,0].min()+positive[:,0].max())/2
        templates[typ]=[positive[positive[:,0]<threshold].mean(0).tolist(),positive[positive[:,0]>=threshold].mean(0).tolist()]
        for sig in (-1,0,1):
            a=logit(1-y[m&np.array([q.signal==sig for q in queries]),0])
            lo,hi=np.quantile(a,[.25,.75])
            for _ in range(20):
                side=a>(lo+hi)/2
                lo,hi=a[~side].mean(),a[side].mean()
            centers[f'{typ}|{sig}']=[float(lo),float(hi)]
    return templates,centers


class TraceHMM(HMM):
    def predict_trace(self,hazard=3e-5,spread=.3,amplitude_scale=1,forget=1,diffusion=0,quadrature=False,amplitude_sd=.15):
        n,T,P=self.bits.shape
        offsets=np.array([-1.5,-.75,0,.75,1.5])*spread
        scales=np.array([.65,1,1.35])*amplitude_scale
        if quadrature:
            from numpy.polynomial.hermite import hermgauss
            nodes,ow=hermgauss(7);offsets=nodes*np.sqrt(2)*spread;ow=ow/np.sqrt(np.pi)
            if amplitude_sd:
                nodes,aw=hermgauss(3);scales=amplitude_scale+nodes*np.sqrt(2)*amplitude_sd;aw=aw/np.sqrt(np.pi)
            else:scales=np.array([amplitude_scale]);aw=np.ones(1)
        grid=np.array([(o,a) for o in offsets for a in scales]);G=len(grid)
        mid=self.c.mean(1);amp=(self.c[:,1]-self.c[:,0])/2
        lo=expit(mid[:,None]+grid[None,:,0]-amp[:,None]*grid[None,:,1])
        hi=expit(mid[:,None]+grid[None,:,0]+amp[:,None]*grid[None,:,1])
        if quadrature:gw=np.outer(ow,aw).ravel()
        else:gw=np.exp(-.5*(grid[:,0]/max(spread,.01))**2-.5*((grid[:,1]/amplitude_scale-1)/.3)**2);gw/=gw.sum()
        phase=np.full((n,P,G),.5);logw=np.broadcast_to(np.log(gw)[None,None,:],(n,P,G)).copy()-np.log(P)
        result=np.zeros(n);trace=np.zeros((n,T));posterior=np.zeros((n,12+P))
        for t in range(T):
            phase=phase*(1-2*hazard)+hazard
            if diffusion:
                weights=np.exp(logw-logsumexp(logw,axis=(1,2))[:,None,None])
                joint0=(weights*(1-phase)).reshape(n,P,5,3);joint1=(weights*phase).reshape(n,P,5,3)
                def move(j):
                    shifted=(1-2*diffusion)*j
                    shifted[:,:,1:]+=diffusion*j[:,:,:-1];shifted[:,:,:-1]+=diffusion*j[:,:,1:]
                    shifted[:,:,0]+=diffusion*j[:,:,0];shifted[:,:,-1]+=diffusion*j[:,:,-1]
                    return shifted.reshape(n,P,G)
                p0,p1=move(joint0),move(joint1)
                weights=p0+p1;phase=p1/np.maximum(weights,1e-300);logw=np.log(np.maximum(weights,1e-300))
            b=self.bits[:,t,:,None]
            e0=np.where(b<0,(lo[:,None]+hi[:,None])/2,np.where(b>0,hi[:,None],lo[:,None]))
            e1=np.where(b<0,(lo[:,None]+hi[:,None])/2,np.where(b>0,lo[:,None],hi[:,None]))
            weights=np.exp(logw-logsumexp(logw,axis=(1,2))[:,None,None])
            mean=(1-phase)*e0+phase*e1
            prediction=(weights*mean).sum((1,2));trace[:,t]=prediction
            need=self.cut==t
            if need.any():
                result[need]=prediction[need]
                wp=weights.sum(2);wg=weights.sum(1);joint=np.stack((weights*(1-phase),weights*phase),axis=-1)
                uncertainty=(weights*((1-phase)*e0**2+phase*e1**2)).sum((1,2))-prediction**2
                vals=[uncertainty,(wp*np.log(np.maximum(wp,1e-300))).sum(1)*-1,
                      (weights*(lo[:,None]+hi[:,None])/2).sum((1,2)),
                      (wg*grid[None,:,0]).sum(1),(wg*grid[None,:,1]).sum(1)]
                vals.extend(wp[:,j] for j in range(P))
                vals.extend((joint[:,:,:,s]*np.where(b>=0,(b>0) if s==0 else (b==0),.5)).sum((1,2)) for s in range(2))
                vals.extend(np.mean(self.bits[:,t]==k,axis=1) for k in (-1,0,1))
                vals.extend([np.max(wp,axis=1),np.max(wg,axis=1)])
                posterior[need]=np.column_stack(vals)[need]
            active=(self.y[:,t]>=0)&(t<self.cut);obs=self.y[:,t,None,None]
            e0=np.where(obs>0,e0,1-e0);e1=np.where(obs>0,e1,1-e1)
            likelihood=(1-phase)*e0+phase*e1
            phase=np.where(active[:,None,None],phase*e1/likelihood,phase)
            logw=np.where(active[:,None,None],logw*forget+np.log(likelihood),logw)
        return result,trace,posterior


def augmented_features(queries,history,positive,win,trace,posterior):
    original=old_features(queries,history,positive,win)
    ix=index_history(queries,history);groups={}
    for i,q in enumerate(queries):groups.setdefault(q.episode_id,[]).append(i)
    out=[]
    for i,q in enumerate(queries):
        row=posterior[i].tolist()+[logit(np.clip(win[i],1e-6,1-1e-6))]
        own=ix.get((q.episode_id,q.machine_id),[])
        for neighborhood in (False,True):
            others=[j for j in groups[q.episode_id] if j!=i] if neighborhood else [i]
            for half in (2,8,24,64,256):
                numerator=0.;fisher=0.;support=0.;observed=0.;expected=0.;large=0.
                for j in others:
                    other=queries[j]
                    for r in ix.get((other.episode_id,other.machine_id),[]):
                        if r.tick>=q.cutoff:continue
                        w=2.**(-(q.cutoff-1-r.tick)/half);p=trace[j,r.tick];y=float(r.payout>0)
                        numerator+=w*(y-p);fisher+=w*p*(1-p);support+=w;observed+=w*y;expected+=w*p;large+=w*(r.payout>=2)
                row.extend([numerator/(fisher+2),numerator/np.sqrt(fisher+1),support/256,
                            observed/(support+1e-8),expected/(support+1e-8),large/(observed+1e-8)])
        # Compare short- and long-memory residual estimates.
        row.extend([trace[i,max(0,q.cutoff-k)] for k in (1,2,4,8,16,32)])
        out.append(row)
    return np.column_stack((original,np.array(out)))


def pooled_centers(queries,y,train,context):
    typ=np.array([q.machine_type for q in queries]);sig=np.array([q.signal for q in queries]);z=logit(1-y[:,0])
    side=np.array([1 if v>np.mean(context[q.machine_type+'|'+str(q.signal)]) else -1 for q,v in zip(queries,z)])
    x=np.column_stack((typ=='steady',typ=='balanced',typ=='jackpot',sig,side)).astype(float)
    coef=np.linalg.lstsq(x[train],z[train],rcond=None)[0]
    return {t+'|'+str(s):[float(coef[k]+coef[3]*s-coef[4]),float(coef[k]+coef[3]*s+coef[4])] for k,t in enumerate(('steady','balanced','jackpot')) for s in (-1,0,1)}


def fit_amount_patterns(queries,history,templates):
    from .general_amount import GeneralAmount
    from scipy.special import softmax
    model=GeneralAmount(queries,history,templates)
    _,evidence=model.predict(hazard=.0001,return_evidence=True)
    prior=np.ones(len(model.patterns))/len(model.patterns)
    for _ in range(40):prior=(softmax(evidence+np.log(prior),axis=1).sum(0)+2)/(len(queries)+2*len(prior))
    keep=np.flatnonzero(prior>=.02)
    if not len(keep):keep=np.arange(len(prior))
    return [model.patterns[i] for i in keep],prior[keep]/prior[keep].sum()


def fit_ridge(X,y,alpha):
    mean=X.mean(0);scale=X.std(0);scale[scale<1e-12]=1
    Z=(X-mean)/scale;bias=float(np.mean(y));a=Z.T@Z+alpha*np.eye(Z.shape[1]);coef=np.linalg.solve(a,Z.T@(y-bias))
    return dict(mean=mean.tolist(),scale=scale.tolist(),coef=coef.tolist(),bias=bias)


def ridge_predict(X,model):
    return ((X-np.array(model['mean']))/np.array(model['scale']))@np.array(model['coef'])+model['bias']
