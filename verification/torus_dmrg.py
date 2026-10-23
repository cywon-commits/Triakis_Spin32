"""Sec. 3.0 check: 18-spin torus (T1=(3,0), T2=(-1,2)) by finite DMRG.

The bond list is generated from the request's 9-row bond table (triakis/model.py
BONDS, the same table used for the cylinder) and compared with ed.py's
independent triangle-based construction; then DMRG energies and highest-weight
observables are computed for comparison with the ED reference values.

usage: python3 torus_dmrg.py <2S> <2Sz> [<2Sz> ...] [--chi 1200]
"""
import sys, os, json, argparse, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from triakis.model import torus_bonds
from tenpy.networks.site import SpinSite
from tenpy.networks.mps import MPS
from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.algorithms import dmrg
import tenpy
tenpy.tools.misc.setup_logging(to_stdout=None)


def check_against_ed():
    from ed import cluster
    N, ns, bonds_ed, sub_ed, sites_ed = cluster()
    ns2, N2, b, cells, sub = torus_bonds()
    bonds = {k: v['J1'] + v['J2'] for k, v in b.items()}
    assert (N, ns) == (N2, ns2) and list(map(tuple, sites_ed)) == cells and list(sub_ed) == sub
    assert bonds == bonds_ed, 'bond tables differ!'
    nJ = sum(bonds.values())
    deg = np.zeros(ns, int)
    for (i, j), J in bonds.items():
        deg[i] += J; deg[j] += J
    print(f'bond table OK: {len(bonds)} distinct bonds, sum J = {nJ}, '
          f'corner coordination {set(deg[:N])}, center coordination {set(deg[N:])}')
    return ns, N, b, sub


class TorusModel(CouplingModel, MPOModel):
    def __init__(self, S, b, ns, J1=1.0, J2=1.0):
        site = SpinSite(S=S, conserve='Sz')
        lat = Chain(ns, site, bc_MPS='finite', bc='open')
        CouplingModel.__init__(self, lat)
        for (i, j), n in b.items():
            J = J1 * n['J1'] + J2 * n['J2']
            self.add_coupling_term(J, i, j, 'Sz', 'Sz')
            self.add_coupling_term(0.5 * J, i, j, 'Sp', 'Sm')
            self.add_coupling_term(0.5 * J, i, j, 'Sm', 'Sp')
        MPOModel.__init__(self, lat, self.calc_H_MPO())


def product_state(S, ns, N, twoSz):
    """Centers up, corners down, then raise corners (round robin) to reach 2Sz."""
    lev = np.array([0] * N + [int(2 * S)] * (ns - N))   # level = m + S
    target = (twoSz + ns * int(2 * S)) // 2
    k = 0
    while lev.sum() < target:
        i = k % ns
        if lev[i] < 2 * S:
            lev[i] += 1
        k += 1
    while lev.sum() > target:
        i = N + k % (ns - N)
        if lev[i] > 0:
            lev[i] -= 1
        k += 1
    return [int(l) for l in lev]


def observables(psi, S, N, b, sub):
    ns = psi.L
    mz = psi.expectation_value('Sz')
    pm = psi.correlation_function('Sp', 'Sm')
    zz = psi.correlation_function('Sz', 'Sz')
    perp = lambda i, j: 0.5 * (pm[i, j] + pm[j, i]).real
    same = [(i, j) for i in range(N) for j in range(i + 1, N) if sub[i] == sub[j]]
    nn = [(i, j) for i in range(N) for j in range(i + 1, N) if sub[i] != sub[j]]
    cc = [(i, j) for (i, j) in b if j >= N]
    r = dict(mz_corner=np.mean(mz[:N]) / S, mz_center=np.mean(mz[N:]) / S,
             perp_same=np.mean([perp(*p) for p in same]) / S**2,
             perp_nn=np.mean([perp(*p) for p in nn]) / S**2,
             perp_cc_center=np.mean([perp(*p) for p in cc]) / S**2,
             zz_same=np.mean([zz[p].real for p in same]) / S**2,
             zz_nn=np.mean([zz[p].real for p in nn]) / S**2)
    return {k: float(v) for k, v in r.items()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('twoS', type=int)
    ap.add_argument('twoSz', type=int, nargs='+')
    ap.add_argument('--chi', type=int, default=1200)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'torus_results.jsonl'))
    a = ap.parse_args()
    S = a.twoS / 2
    ns, N, b, sub = check_against_ed()
    M = TorusModel(S, b, ns)
    for twoSz in a.twoSz:
        t = time.time()
        psi = MPS.from_product_state(M.lat.mps_sites(), product_state(S, ns, N, twoSz), bc='finite')
        chis = [c for c in (64, 128, 256, 512, 1024, 2048) if c < a.chi] + [a.chi]
        p = dict(mixer=True, max_E_err=1e-10, max_S_err=1e-7, min_sweeps=4, max_sweeps=40,
                 chi_list={2 * k: c for k, c in enumerate(chis)},
                 trunc_params=dict(chi_max=a.chi, svd_min=1e-12))
        info = dmrg.run(psi, M, p)
        E = float(info['E'])
        res = dict(twoS=a.twoS, Sz=twoSz / 2, chi=a.chi, E=E,
                   max_trunc=float(np.max(info['sweep_statistics']['max_trunc_err'][-1])),
                   time=time.time() - t)
        res.update(observables(psi, S, N, b, sub))
        print(json.dumps(res), flush=True)
        with open(a.out, 'a') as f:
            f.write(json.dumps(res) + '\n')
