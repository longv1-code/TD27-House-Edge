"""Fit and evaluate the coupled House Edge model using released data."""
import hashlib
import numpy as np
from .research_engine import Engine,learn_prior,PATTERNS
from .general_amount import GeneralAmount
from .coupling import fit_joint,coupled_amount
from .next_core import (
    fit_population,pooled_centers,TraceHMM,fit_amount_patterns,
    augmented_features,fit_ridge,ridge_predict,
)
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.schema import CONDITIONS,index_history

CONFIG={
    'hmm':{'quadrature':True,'spread':.3,'amplitude_sd':.1},
    'pooled_centers':True,'amount_hazard':1e-8,'amount_temperature':1,
    'coupling':{'concentration':1.0,'power':2.0},
    'amount_imputation':{'strength':30,'half':10000},
    'ridge_alpha':10,'ridge_weight':.5,'episode_factor':{'phi':1,'noise':.02,'weight':.25},
}


def discover_win_patterns(queries,history,labels):
    base=FrequencyForecaster('pooled_recent').fit(queries,history,labels)
    engine=Engine(queries,history,base);prior=learn_prior(engine)
    selected=[i for i,p in enumerate(prior) if p>=.03 and PATTERNS[i]]
    if not selected:
        selected=sorted([i for i,p in enumerate(PATTERNS) if p],key=lambda i:-prior[i])[:3]
    selected.sort(key=lambda i:-prior[i])
    return [PATTERNS[i] for i in selected]


def build_features(queries,history,model):
    patterns=[tuple(p) for p in model['win_patterns']]
    hmm=TraceHMM(queries,history,patterns,model['centers'])
    win,trace,posterior=hmm.predict_trace(**model['config']['hmm'])
    amount_patterns=[(int(kind),tuple(lags)) for kind,lags in model['amount_patterns']]
    amount=GeneralAmount(queries,history,model['templates'],amount_patterns)
    soft=amount.soft_contexts(model['centers'],**model['config']['amount_imputation'])
    _,evidence=amount.predict(model['config']['amount_hazard'],model['config']['amount_temperature'],return_evidence=True,bit_probs=soft)
    positive=coupled_amount(hmm.last_details['weights'].sum(2),amount,evidence,model['joint_lag_prior'])
    X=augmented_features(queries,history,positive,win,trace,posterior)
    return win,positive,X,trace


def fit_model(train_queries,train_history,queries,history,y,mask,win_patterns,config=None):
    cfg=dict(CONFIG if config is None else config)
    templates,centers=fit_population(queries,y,mask)
    if cfg['pooled_centers']:centers=pooled_centers(queries,y,mask,centers)
    amount_patterns,amount_prior=fit_amount_patterns(train_queries,train_history,templates)
    # Estimate the link from training histories only. Both component filters use
    # population parameters fitted outside the held-out validation episodes.
    train_win=TraceHMM(train_queries,train_history,win_patterns,centers)
    train_win.predict_trace(**cfg['hmm'])
    train_amount=GeneralAmount(train_queries,train_history,templates,amount_patterns)
    soft=train_amount.soft_contexts(centers,**cfg['amount_imputation'])
    _,evidence=train_amount.predict(cfg['amount_hazard'],cfg['amount_temperature'],return_evidence=True,bit_probs=soft)
    joint=fit_joint(train_win.last_details['weights'].sum(2),evidence,**cfg['coupling'])
    model={
        'version':4,'config':cfg,'win_patterns':[list(p) for p in win_patterns],
        'templates':templates,'centers':centers,
        'amount_patterns':[[kind,list(lags)] for kind,lags in amount_patterns],
        'amount_prior':amount_prior.tolist(),
        'joint_lag_prior':joint.tolist(),
    }
    win,positive,X,trace=build_features(queries,history,model)
    model['ridge']=fit_ridge(X[mask],((1-y[:,0])-win)[mask],cfg['ridge_alpha'])
    return model,finish_prediction(queries,history,model,win,positive,X,trace)


def episode_factor(queries,history,trace,phi,noise):
    from scipy.special import expit,logit
    index=index_history(queries,history);episodes={};out=np.zeros(len(queries))
    for i,q in enumerate(queries):episodes.setdefault(q.episode_id,[]).append(i)
    for ids in episodes.values():
        cutoff=max(queries[i].cutoff for i in ids)
        observations=[{r.tick:float(r.payout>0) for r in index.get((queries[i].episode_id,queries[i].machine_id),[])} for i in ids]
        mean=0.;variance=.25
        for tick in range(cutoff+1):
            mean=phi*mean;variance=phi*phi*variance+noise
            for i in ids:
                if queries[i].cutoff==tick:out[i]=mean
            values=[(trace[i,tick],observations[j][tick]) for j,i in enumerate(ids)
                    if tick<queries[i].cutoff and tick in observations[j]]
            if not values:continue
            offset=logit(np.clip(np.array([v[0] for v in values]),1e-5,1-1e-5));y=np.array([v[1] for v in values])
            prior_mean=mean;prior_var=max(variance,1e-6)
            for _ in range(3):
                p=expit(offset+mean);precision=1/prior_var+np.sum(p*(1-p))
                mean-=((mean-prior_mean)/prior_var-np.sum(y-p))/precision
            variance=1/precision
    return out


def finish_prediction(queries,history,model,win,positive,X,trace):
    from scipy.special import expit,logit
    cfg=model['config']
    win=np.clip(win+cfg['ridge_weight']*ridge_predict(X,model['ridge']),1e-5,1-1e-5)
    factor=cfg['episode_factor']
    if factor:
        shift=episode_factor(queries,history,trace,factor['phi'],factor['noise'])
        win=expit(logit(win)+factor['weight']*shift)
    p=np.column_stack((1-win,win[:,None]*positive));p/=p.sum(1)[:,None]
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('Invalid forecast probabilities')
    return p


def predict(queries,history,model):
    return finish_prediction(queries,history,model,*build_features(queries,history,model))


def episode_folds(queries,conditions):
    assignment={}
    for condition in CONDITIONS:
        episodes={q.episode_id for q in queries if conditions[q.query_id]==condition}
        ordered=sorted(episodes,key=lambda e:hashlib.sha256(('next-v3:'+e).encode()).hexdigest())
        if len(ordered)<4:raise ValueError('Four-fold validation requires at least four episodes per condition')
        assignment.update({episode:i%4 for i,episode in enumerate(ordered)})
    return np.array([assignment[q.episode_id] for q in queries]),assignment
