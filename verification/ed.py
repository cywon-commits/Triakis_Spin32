import numpy as np, sys, time
from scipy.sparse.linalg import LinearOperator, eigsh
from collections import Counter
def cluster():
    # corners on triangular torus T1=(3,0), T2=(-1,2); sites (x,y) reduced
    T1=np.array([3,0]);T2=np.array([-1,2]); N=6
    Minv=np.linalg.inv(np.array([T1,T2]).T)
    def red(p):
        a=Minv@p; a=a-np.floor(a+1e-9); q=np.round(np.array([T1,T2]).T@a).astype(int); return tuple(q)
    sites=sorted({red(np.array([x,y])) for x in range(-6,7) for y in range(-6,7)})
    assert len(sites)==N
    idx={s:i for i,s in enumerate(sites)}
    I=lambda x,y: idx[red(np.array([x,y]))]
    bonds=Counter(); tris=[]
    for (x,y) in sites:
        for d in [(1,0),(0,1),(-1,1)]:
            a,b=I(x,y),I(x+d[0],y+d[1]); bonds[tuple(sorted((a,b)))]+=1
        tris.append((I(x,y),I(x+1,y),I(x,y+1))); tris.append((I(x+1,y),I(x,y+1),I(x+1,y+1)))
    nsite=N+len(tris)
    for k,t in enumerate(tris):
        for i in t: bonds[(i,N+k)]+=1
    sub=[ (x+2*y)%3 for (x,y) in sites]
    return N,nsite,dict(bonds),sub,sites
def basis(nsite,twoS,M2):   # M2 = 2*Sz target; m stored as (2m+twoS)/2 in 0..twoS  -> "level" l=m+S
    d=twoS+1; target=(M2+nsite*twoS)//2   # sum of levels
    codes=np.array([0],np.int64); sums=np.array([0])
    for k in range(nsite):
        rem=nsite-k-1
        c2=(codes[:,None]*d+np.arange(d)[None,:]).ravel(); s2=(sums[:,None]+np.arange(d)[None,:]).ravel()
        ok=(s2<=target)&(s2+rem*twoS>=target); codes,sums=c2[ok],s2[ok]
    return np.sort(codes)
def levels(codes,nsite,twoS):
    d=twoS+1; L=np.zeros((nsite,len(codes)),np.int8); c=codes.copy()
    for k in range(nsite-1,-1,-1): L[k]=c%d; c//=d
    return L
def build_op(codes,L,bonds,nsite,twoS):
    d=twoS+1; S=twoS/2; pw=d**np.arange(nsite-1,-1,-1)
    m=L.astype(float)-S
    diag=np.zeros(len(codes))
    for (i,j),J in bonds.items(): diag+=J*m[i]*m[j]
    off=[]
    for (i,j),J in bonds.items():
        for (a,b) in [(i,j),(j,i)]:     # S_a^+ S_b^-
            sel=np.nonzero((L[a]<twoS)&(L[b]>0))[0].astype(np.int32)
            tgt=np.searchsorted(codes,codes[sel]+pw[a]-pw[b]).astype(np.int32)
            ma,mb=m[a][sel],m[b][sel]
            amp=(0.5*J*np.sqrt(S*(S+1)-ma*(ma+1))*np.sqrt(S*(S+1)-mb*(mb-1))).astype(np.float32)
            off.append((sel,tgt,amp))
    def mv(v):
        v=v.ravel(); out=diag*v
        for s_,t,a in off: out+=np.bincount(t,weights=a*v[s_],minlength=len(v))
        return out
    return LinearOperator((len(codes),len(codes)),matvec=mv,dtype=float)
def corr_perp(v,L,i,j,twoS):
    # <S_i^x S_j^x + S_i^y S_j^y> = 1/2<S_i^+S_j^- + h.c.>  computed via levels
    return None
if __name__=='__main__':
    twoS=int(sys.argv[1]); Ms=[int(x) for x in sys.argv[2:]]   # 2*Sz values
    N,ns,bonds,sub,sites=cluster()
    print('corners',N,'sites',ns,'bonds',len(bonds),'sum J',sum(bonds.values()),'sublattice',sub)
    for M2 in Ms:
        t=time.time(); codes=basis(ns,twoS,M2); L=levels(codes,ns,twoS)
        H=build_op(codes,L,bonds,ns,twoS)
        w,v=eigsh(H,k=1,which='SA',tol=1e-9)
        np.save(f'gs_{twoS}_{M2}.npy',v[:,0]); np.save(f'codes_{twoS}_{M2}.npy',codes)
        print(f'2S={twoS} Sz={M2/2}: dim={len(codes)} E0={w[0]:.6f}  ({time.time()-t:.0f}s)',flush=True)
