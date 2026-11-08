import numpy as np
from slot_forecasting.schema import index_history,PAYOUTS

def features(queries,history,positive,win):
    ix=index_history(queries,history);out=[]
    for i,q in enumerate(queries):
        seq={r.tick:PAYOUTS.index(r.payout) for r in ix.get((q.episode_id,q.machine_id),[])};ts=np.array(sorted(seq));yy=np.array([seq[t] for t in ts]);age=q.cutoff-1-ts
        row=[q.signal,win[i],positive[i,0],len(ts)/256]
        row += [float(q.machine_type==typ) for typ in ['steady','balanced','jackpot']]
        bits=[]
        for lag in range(1,9):
            v=seq.get(q.cutoff-lag,-1);b=float(v>0) if v>=0 else .5;bits.append(b);row.extend([b,v/4])
        for a in range(8):
            for b in range(a+1,8):row.append(float(bits[a]!=bits[b]))
        for half in [4,12,32,96,10000]:
            w=2.**(-age/half);counts=np.array([w[yy==k].sum() for k in range(5)])
            row.extend((counts/(counts.sum()+1e-9)).tolist());row.extend((counts[1:]/(counts[1:].sum()+1e-9)).tolist())
        # Amount estimates conditioned on each earlier win bit, including history support.
        for lag in range(1,5):
            want=bits[lag-1];valid=np.array([t-lag in seq and float(seq[t-lag]>0)==want for t in ts])
            w=2.**(-age/48)*valid;counts=np.array([w[yy==k].sum() for k in range(1,5)])
            row.extend([counts[0]/(counts.sum()+1e-9),counts.sum()/128])
        out.append(row)
    return np.array(out)
