import numpy as np,pandas as pd
sp='/tmp/claude-0/-home-claude/14f09c49-bb96-5f73-8ddf-c3b7c11d4322/scratchpad/'
exec(open(sp+'scores.py').read())
mat=np.load(sp+'subs_mat.npy'); names=list(pd.read_csv(sp+'subs_names.csv').iloc[:,0]); reg=np.load(sp+'region.npy',allow_pickle=True)
mask=reg!='eastern-wa'
A=mat[mask]; n=A.shape[0]
r=names.index('t17_joint_f123')
D=A-A[:,[r]]; q=(D**2).mean(0); G_all=D.T@D/n
def loo_err(SCd, exclude=('f4_scale_wa',), only=None):
    M=np.array([SCd[nm]**2 for nm in names]); g=(M-M[r]-q)/2
    ks=[i for i in range(len(names)) if i!=r and names[i] not in exclude]
    errs={}
    for i in ks:
        if only is not None and names[i] not in only: continue
        others=[j for j in ks if j!=i]
        G=G_all[np.ix_(others,others)]; b=G_all[others,i]
        lam=1e-6*np.trace(G)/len(others)
        beta=np.linalg.solve(G+lam*np.eye(len(others)),b)
        frac=(q[i]-2*beta@b+beta@G@beta)/q[i]
        mse_hat=M[r]+2*beta@g[others]+q[i]
        errs[names[i]]=(frac,np.sqrt(max(mse_hat,0))-np.sqrt(M[i]))
    return errs
