"""Finite open cylinder (brief 2 geometry): E(Sz_tot) sector scan by finite DMRG.

Ly corners around (periodic along a1), Lx rings along a2 (open). Only complete
triangles carry a center, so every center has exactly 3 corner neighbours.
MPS order: ring by ring, corners of ring y then the 2*Ly centers of the
triangles between rings y and y+1.

The ground-state multiplet S_tot shows up as E(Sz) degenerate for Sz <= S_tot;
m* per ring = slope of S_tot in Lx (edge contributions cancel).

usage: python3 -m triakis.finite_cyl --Lx 4 --Sz 16 18 20 22 24 [--S 1.5 --Ly 3 --J1 1 --J2 1 --chi 512]
"""
import os, sys, json, time, argparse
import numpy as np
from tenpy.networks.site import SpinSite
from tenpy.networks.mps import MPS
from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.algorithms import dmrg
from triakis.model import BONDS
import tenpy
tenpy.tools.misc.setup_logging(to_stdout=None)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def finite_bonds(Ly, Lx):
    idx, kind = {}, []
    for y in range(Lx):
        for x in range(Ly):
            idx[(x, y, 0)] = len(kind); kind.append(0)
        if y < Lx - 1:
            for x in range(Ly):
                for u in (1, 2):
                    idx[(x, y, u)] = len(kind); kind.append(u)
    bonds = {}
    for (x, y, u), i in list(idx.items()):
        for ui, uj, (d1, d2), k in BONDS:
            if ui != u:
                continue
            tgt = ((x + d1) % Ly, y + d2, uj)
            if tgt not in idx:
                continue
            key = tuple(sorted((i, idx[tgt])))
            bonds.setdefault(key, {'J1': 0, 'J2': 0})[k] += 1
    return len(kind), np.array(kind), bonds


class FiniteModel(CouplingModel, MPOModel):
    def __init__(self, S, ns, bonds, J1, J2):
        site = SpinSite(S=S, conserve='Sz')
        lat = Chain(ns, site, bc_MPS='finite', bc='open')
        CouplingModel.__init__(self, lat)
        for (i, j), n in bonds.items():
            J = J1 * n['J1'] + J2 * n['J2']
            if J == 0:
                continue
            self.add_coupling_term(J, i, j, 'Sz', 'Sz')
            self.add_coupling_term(0.5 * J, i, j, 'Sp', 'Sm')
            self.add_coupling_term(0.5 * J, i, j, 'Sm', 'Sp')
        MPOModel.__init__(self, lat, self.calc_H_MPO())


def levels(S, kind, Sz, rng=None):
    twoS = int(round(2 * S)); n = len(kind)
    target = int(round(Sz + n * S))
    if rng is None:
        lev = np.where(kind > 0, twoS, 0)
        order = [i for i in range(n) if kind[i] == 0] + [i for i in range(n) if kind[i] > 0]
        k = 0
        while lev.sum() < target:
            i = order[k % n]
            if lev[i] < twoS:
                lev[i] += 1
            k += 1
        k = 0
        while lev.sum() > target:
            i = order[::-1][k % n]
            if lev[i] > 0:
                lev[i] -= 1
            k += 1
    else:
        lev = np.zeros(n, int)
        while lev.sum() < target:
            i = rng.integers(n)
            if lev[i] < twoS:
                lev[i] += 1
    return [int(l) for l in lev]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--S', type=float, default=1.5)
    ap.add_argument('--Ly', type=int, default=3)
    ap.add_argument('--Lx', type=int, required=True)
    ap.add_argument('--J1', type=float, default=1.0)
    ap.add_argument('--J2', type=float, default=1.0)
    ap.add_argument('--Sz', type=float, nargs='+', required=True)
    ap.add_argument('--chi', type=int, default=512)
    ap.add_argument('--seed', type=int, default=-1)
    a = ap.parse_args()
    ns, kind, bonds = finite_bonds(a.Ly, a.Lx)
    nA, nB = int(np.sum(kind == 0)), int(np.sum(kind > 0))
    print(f'sites {ns} (A {nA}, B {nB}), bonds {len(bonds)}, S(N_B-N_A) = {a.S * (nB - nA)}', flush=True)
    M = FiniteModel(a.S, ns, bonds, a.J1, a.J2)
    for Sz in a.Sz:
        t = time.time()
        rng = np.random.default_rng(a.seed) if a.seed >= 0 else None
        psi = MPS.from_product_state(M.lat.mps_sites(), levels(a.S, kind, Sz, rng), bc='finite')
        chis = [c for c in (64, 128, 256, 512, 1024) if c < a.chi] + [a.chi]
        p = dict(mixer=True, max_E_err=1e-8, max_S_err=1e-5, min_sweeps=4, max_sweeps=24, max_trunc_err=1.0,
                 chi_list={2 * k: c for k, c in enumerate(chis)},
                 trunc_params=dict(chi_max=a.chi, svd_min=1e-10))
        info = dmrg.run(psi, M, p)
        st = info['sweep_statistics']
        sz = psi.expectation_value('Sz')
        res = dict(S=a.S, Ly=a.Ly, Lx=a.Lx, J1=a.J1, J2=a.J2, Sz=Sz, chi=a.chi, seed=a.seed,
                   E=float(info['E']), trunc=float(st['max_trunc_err'][-1]),
                   dE=float(abs(st['E'][-1] - st['E'][-2])) if len(st['E']) > 1 else None,
                   nA=nA, nB=nB, szA=float(np.mean(sz[kind == 0])), szB=float(np.mean(sz[kind > 0])),
                   sz_ring_A=[float(np.mean(sz[kind == 0][y * a.Ly:(y + 1) * a.Ly])) for y in range(a.Lx)],
                   time=time.time() - t)
        print(json.dumps(res), flush=True)
        with open(os.path.join(ROOT, 'data', 'finite_results.jsonl'), 'a') as f:
            f.write(json.dumps(res) + '\n')
