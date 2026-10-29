import numpy as np,pandas as pd
sp='/tmp/claude-0/-home-claude/14f09c49-bb96-5f73-8ddf-c3b7c11d4322/scratchpad/'
exec(open(sp+'scores.py').read()); 
mat=np.load(sp+'mat2.npy'); names=list(pd.read_csv(sp+'names2.csv').iloc[:,0])
T=pd.read_csv('/mnt/user-data/outputs/t18_global.csv',dtype={'GEOID':str}); reg=T.region.values
mask=reg!='eastern-wa'; A=mat[mask]; n=len(A)
excl={'f4_scale_wa'}
r=names.index('t21_global'); D=A-A[:,[r]]; q=(D**2).mean(0); G=D.T@D/n
M=np.array([SC[k]**2 for k in names]); g=(M-M[r]-q)/2
ks=[i for i in range(len(names)) if i!=r and names[i] not in excl]
# LOO consistency
err={}
for i in ks:
    o=[j for j in ks if j!=i]; Go=G[np.ix_(o,o)]; b=G[o,i]
    beta=np.linalg.solve(Go+1e-6*np.trace(Go)/len(o)*np.eye(len(o)),b)
    mh=M[r]+2*beta@g[o]+q[i]
    # include residual part unknown -> compare only explained component
    err[names[i]]=np.sqrt(max(mh,0))-SC[names[i]] if (q[i]-2*beta@b+beta@Go@beta)/q[i]<0.05 else np.nan
e=pd.Series(err); print('LOO err (explained<5% resid) top abs:'); print(e.dropna().abs().sort_values().tail(8))
sig=np.array([2*0.066*max(abs(err.get(names[i],np.nan)) if not np.isnan(err.get(names[i],np.nan)) else 1e-4,2e-5) for i in ks])
Gk=G[np.ix_(ks,ks)]; gk=g[ks]
def solve(lam,gg=gk): return -np.linalg.solve(Gk+lam*np.diag(sig**2),gg)
rng=np.random.default_rng(0)
for lam in [3e4,1e5,3e5,1e6,3e6]:
    a=solve(lam); pred=M[r]+2*a@gk+a@Gk@a
    gains=[]
    for _ in range(300):
        gn=gk+rng.normal(0,sig/2*0+sig*0.5)   # noise in g
        an=solve(lam,gn); gains.append(-(2*an@gk+an@Gk@an))
    print(f"lam {lam:.0e} pred {np.sqrt(pred):.6f}  sim gain mean {np.mean(gains):.2e}  p10 {np.percentile(gains,10):.2e} |a|max {np.abs(a).max():.2f}")
np.save(sp+'ks.npy',np.array(ks)); np.save(sp+'sig.npy',sig)
