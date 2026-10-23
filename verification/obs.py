import numpy as np, sys
from ed import cluster, levels
def observables(twoS,M2):
    N,ns,bonds,sub,sites=cluster()
    v=np.load(f'gs_{twoS}_{M2}.npy'); codes=np.load(f'codes_{twoS}_{M2}.npy')
    L=levels(codes,ns,twoS); S=twoS/2; m=L.astype(float)-S; d=twoS+1; pw=d**np.arange(ns-1,-1,-1)
    p=v**2
    def zz(i,j): return (p*m[i]*m[j]).sum()
    def pm(a,b):   # <S_a^+ S_b^->
        sel=np.nonzero((L[a]<twoS)&(L[b]>0))[0]
        tgt=np.searchsorted(codes,codes[sel]+pw[a]-pw[b])
        ma,mb=m[a][sel],m[b][sel]
        amp=np.sqrt(S*(S+1)-ma*(ma+1))*np.sqrt(S*(S+1)-mb*(mb-1))
        return (v[tgt]*amp*v[sel]).sum()
    def perp(i,j): return 0.5*(pm(i,j)+pm(j,i))
    mz=[(p*m[i]).sum() for i in range(ns)]
    # corner pairs
    same=[];nn=[]
    nnset={k for k in bonds if k[0]<N and k[1]<N}
    for i in range(N):
        for j in range(i+1,N):
            (same if sub[i]==sub[j] else nn).append((i,j))
    res={}
    res['mz_corner']=np.mean(mz[:N]); res['mz_center']=np.mean(mz[N:])
    res['perp_same']=np.mean([perp(i,j) for i,j in same]); res['perp_nn']=np.mean([perp(i,j) for i,j in nn])
    res['zz_same']=np.mean([zz(i,j) for i,j in same]); res['zz_nn']=np.mean([zz(i,j) for i,j in nn])
    res['perp_cc_center']=np.mean([perp(i,j) for (i,j) in bonds if j>=N])
    res['n_same_pairs']=len(same)
    return res,S
if __name__=='__main__':
    twoS=int(sys.argv[1]);M2=int(sys.argv[2]); r,S=observables(twoS,M2)
    print(f'S={S} Sz={M2/2}:')
    for k,v in r.items(): print(f'   {k}: {v:.4f}' + (f'   (/S^2 = {v/S**2:.3f})' if k.startswith(('perp','zz')) else '') + (f'   (/S = {v/S:.3f})' if k.startswith('mz') else ''))
