import sys; sys.path.insert(0,'/home/user/Triakis_Spin32')
import numpy as np
from triakis.finite_cyl import finite_bonds
# exact check via small ED for S=1/2 J1=0 Lx=3 (21 sites) using ed-like sparse build
from scipy.sparse.linalg import eigsh
sys.path.insert(0,'/home/user/Triakis_Spin32/verification')
from ed import basis, levels, build_op
for Lx in (2,3):
    ns,kind,b=finite_bonds(3,Lx)
    deg=np.zeros(ns)
    for (i,j),n in b.items(): deg[i]+=n['J1']+n['J2']; deg[j]+=n['J1']+n['J2']
    print('Lx',Lx,'sites',ns,'A deg',sorted(set(deg[kind==0])),'B deg',set(deg[kind>0]))
    for J1 in (0.0,1.0):
        bonds={k:J1*n['J1']+n['J2'] for k,n in b.items() if J1*n['J1']+n['J2']!=0}
        out=[]
        for M2 in range(ns%2, 2*(ns//2)+1, 2):
            codes=basis(ns,1,M2); L=levels(codes,ns,1)
            if len(codes)==0: continue
            H=build_op(codes,L,bonds,ns,1)
            k=1 if len(codes)>1 else 1
            w=eigsh(H,k=1,which='SA',tol=1e-10)[0][0] if len(codes)>2 else np.linalg.eigvalsh(H@np.eye(len(codes)))[0]
            out.append((M2/2,round(w,6)))
        print('  J1',J1,out)
