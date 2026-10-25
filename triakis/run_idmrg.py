"""iDMRG driver for the triakis cylinder at fixed magnetization per ring.

usage: python3 -m triakis.run_idmrg --S 1.5 --Ly 3 --m 5.5 --chis 256 512 1024 [--nring N]
       [--J2 1] [--J1 1] [--nocenters] [--seed K] [--tag name]

For each chi (continuing from the previous state) it writes
  data/<tag>/chi<chi>.npz   observables (energies, <Sz>, transfer-matrix spectra per
                            charge sector, entanglement, C_perp(r), C_zz(r))
  data/<tag>/psi.pkl        last MPS (for restarts)
and appends one summary line to data/summary.jsonl.
"""
import os, sys, json, time, argparse, pickle
import numpy as np
import tenpy
from tenpy.networks.mps import MPS, TransferMatrix
from tenpy.algorithms import dmrg
from triakis.model import TriakisHeisenberg, ring_site_table, sublattice

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def default_nring(S, Ly, m, centers):
    """Smallest ring count n (1,2,3,...) such that n*m is reachable by the cell."""
    nps = 3 * Ly if centers else Ly
    for n in range(1, 13):
        tot2 = 2 * m * n
        if abs(tot2 - round(tot2)) < 1e-9 and (round(tot2) - n * nps * round(2 * S)) % 2 == 0:
            return n
    raise ValueError('no compatible ring count')


def initial_levels(S, tab, m, nring, rng=None):
    """Product state levels (m+S) with total Sz = m*nring.

    classical (rng None): centers up, corners down, then corners raised round-robin
    (lowering centers if m is below the collinear value).  random: random levels with
    the right total.
    """
    twoS = int(round(2 * S))
    n = len(tab)
    target = int(round(m * nring + n * S))
    if rng is None:
        lev = np.array([twoS if u else 0 for (_, _, u) in tab])
        corners = [i for i in range(n) if tab[i][2] == 0]
        centers = [i for i in range(n) if tab[i][2] != 0]
        k = 0
        while lev.sum() < target:
            i = corners[k % len(corners)]
            if lev[i] < twoS:
                lev[i] += 1
            k += 1
            if k > 100 * n:
                raise ValueError('m too large')
        k = 0
        while lev.sum() > target:
            i = centers[k % len(centers)] if centers else corners[k % len(corners)]
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


def tm_spectrum(psi, charge, k=6):
    T = TransferMatrix(psi, psi, charge_sector=psi.chinfo.make_valid([charge]), form='B')
    E, _ = T.eigenvectors(k, which='LM')
    E = np.array(E, dtype=complex)
    return E[np.argsort(-np.abs(E))]


def measure(psi, M, S, Ly, nring, centers, Rmax):
    lat = M.lat
    tab = ring_site_table(lat)
    nps = 3 * Ly if centers else Ly        # sites per ring
    L = psi.L
    out = {}
    sz = psi.expectation_value('Sz')
    out['sz'] = sz
    out['tab'] = tab
    sub = np.array([sublattice(x, r) if u == 0 else -1 for (r, x, u) in tab])
    out['sub'] = sub
    # entanglement on all bonds (bond i = left of site i); ring boundaries are i % nps == 0
    out['S_ent'] = np.array(psi.entanglement_entropy())
    # transfer-matrix spectra: Delta(2Sz) = 0, 2 (transverse), 4
    for name, q in (('tm0', 0), ('tm2', 2), ('tm4', 4)):
        try:
            out[name] = tm_spectrum(psi, q)
        except Exception as e:      # sector may be empty for tiny chi
            print('tm', name, 'failed', e)
            out[name] = np.zeros(0, complex)
    # correlations from reference sites in ring 0 (corner x=0, and centers at x=0)
    nmax = Rmax * nps
    j_all = np.arange(1, nmax)
    for ref_u in ((0, 1, 2) if centers else (0,)):
        i0 = [i for i in range(L) if tab[i][0] == 0 and tab[i][1] == 0 and tab[i][2] == ref_u][0]
        js = j_all + i0
        pm = psi.correlation_function('Sp', 'Sm', sites1=[i0], sites2=list(js))[0]
        mp = psi.correlation_function('Sm', 'Sp', sites1=[i0], sites2=list(js))[0]
        zz = psi.correlation_function('Sz', 'Sz', sites1=[i0], sites2=list(js))[0]
        szj = sz[js % L]
        out[f'perp_u{ref_u}'] = 0.5 * (pm + mp).real
        out[f'zzc_u{ref_u}'] = zz.real - sz[i0] * szj
        out[f'js_u{ref_u}'] = js
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--S', type=float, default=1.5)
    ap.add_argument('--J1', type=float, default=1.0)
    ap.add_argument('--J2', type=float, default=1.0)
    ap.add_argument('--Ly', type=int, default=3)
    ap.add_argument('--m', type=float, required=True, help='Sz per ring')
    ap.add_argument('--nring', type=int, default=0, help='rings per MPS unit cell (0 = auto)')
    ap.add_argument('--chis', type=int, nargs='+', default=[256, 512, 1024])
    ap.add_argument('--nocenters', action='store_true')
    ap.add_argument('--seed', type=int, default=-1, help='>=0: random initial product state')
    ap.add_argument('--Rmax', type=int, default=60, help='correlations up to Rmax rings')
    ap.add_argument('--max_sweeps', type=int, default=60)
    ap.add_argument('--restart', action='store_true', help='continue from saved psi.pkl')
    ap.add_argument('--tag', default='')
    a = ap.parse_args(argv)
    centers = not a.nocenters
    nring = a.nring or default_nring(a.S, a.Ly, a.m, centers)
    tag = a.tag or (f'S{a.S:g}_Ly{a.Ly}_J1{a.J1:g}_J2{a.J2:g}_m{a.m:g}_n{nring}'
                    + ('' if centers else '_nocen') + (f'_seed{a.seed}' if a.seed >= 0 else ''))
    ddir = os.path.join(ROOT, 'data', tag)
    os.makedirs(ddir, exist_ok=True)
    tenpy.tools.misc.setup_logging(to_stdout='INFO', filename=os.path.join(ddir, 'log.txt'))

    M = TriakisHeisenberg(dict(S=a.S, J1=a.J1, J2=a.J2, Ly=a.Ly, Lx=nring, centers=centers,
                               bc_MPS='infinite', conserve='Sz'))
    tab = ring_site_table(M.lat)
    print('MPO bond dims', M.H_MPO.chi, flush=True)
    psi_file = os.path.join(ddir, 'psi.pkl')
    ckpt_file = os.path.join(ddir, 'psi_ckpt.pkl')      # mid-run state, written every checkpoint
    if a.restart and os.path.exists(ckpt_file) and (not os.path.exists(psi_file) or
                                                    os.path.getmtime(ckpt_file) > os.path.getmtime(psi_file)):
        with open(ckpt_file, 'rb') as f:
            psi = pickle.load(f)
        psi.canonical_form()
        print('restarting from mid-run checkpoint', ckpt_file, 'chi', max(psi.chi), flush=True)
    elif a.restart and os.path.exists(psi_file):
        with open(psi_file, 'rb') as f:
            psi = pickle.load(f)
    else:
        rng = np.random.default_rng(a.seed) if a.seed >= 0 else None
        lev = initial_levels(a.S, tab, a.m, nring, rng)
        psi = MPS.from_product_state(M.lat.mps_sites(), lev, bc='infinite')
    nps = 3 * a.Ly if centers else a.Ly
    ncorner = a.Ly * nring
    for i, chi in enumerate(a.chis):
        t0 = time.time()
        first = i == 0 and not a.restart
        p = dict(mixer=True,
                 mixer_params=dict(amplitude=1e-4 if first else 1e-6, decay=1.5,
                                   disable_after=12 if first else 4),
                 max_E_err=1e-9, max_S_err=1e-6, min_sweeps=8 if first else 4,
                 max_sweeps=a.max_sweeps,
                 trunc_params=dict(chi_max=chi, svd_min=1e-10))
        if first:
            ramp = [c for c in (16, 32, 64, 128, 256, 512) if c < chi] + [chi]
            p['chi_list'] = {2 * k: c for k, c in enumerate(ramp)}
        eng = dmrg.TwoSiteDMRGEngine(psi, M, p)

        def _save_ckpt(engine):          # survive container restarts during long runs
            with open(ckpt_file + '.tmp', 'wb') as f:
                pickle.dump(engine.psi, f)
            os.replace(ckpt_file + '.tmp', ckpt_file)
        eng.checkpoint.connect(_save_ckpt)
        E, psi = eng.run()
        st = eng.sweep_stats
        trunc = float(st['max_trunc_err'][-1])
        e_site = float(E)
        res = dict(tag=tag, S=a.S, J1=a.J1, J2=a.J2, Ly=a.Ly, m=a.m, nring=nring,
                   centers=centers, seed=a.seed, chi=chi, chi_real=int(max(psi.chi)),
                   e_site=e_site, e_ring=e_site * nps, e_corner=e_site * nps / a.Ly,
                   trunc=trunc, sweeps=int(len(st['sweep'])),
                   dE_last=float(abs(st['E'][-1] - st['E'][-2])) if len(st['E']) > 1 else None)
        obs = measure(psi, M, a.S, a.Ly, nring, centers, a.Rmax if a.Ly == 3 else a.Rmax // 2)
        sz, sub = obs['sz'], obs['sub']
        u = tab[:, 2]
        for s in range(3):
            res[f'sz_corner{s}'] = float(np.mean(sz[(u == 0) & (sub == s)]))
        res['sz_up'] = float(np.mean(sz[u == 1])) if centers else None
        res['sz_down'] = float(np.mean(sz[u == 2])) if centers else None
        Ls = psi.L / nps     # rings per unit cell
        for name in ('tm0', 'tm2', 'tm4'):
            ev = obs[name]
            if name == 'tm0':
                ev = ev[1:]      # drop eigenvalue 1
            if len(ev):
                lam = ev[0]
                res['xi_' + name] = float(-Ls / np.log(abs(lam)))     # in ring spacings
                res['phase_' + name] = float(np.angle(lam))
                res['xi2_' + name] = float(-Ls / np.log(abs(ev[1]))) if len(ev) > 1 else None
        res['S_ring'] = [float(x) for x in obs['S_ent'][::nps]]
        res['time'] = time.time() - t0
        np.savez(os.path.join(ddir, f'chi{chi}.npz'), **{k: v for k, v in obs.items()},
                 summary=json.dumps(res))
        with open(psi_file, 'wb') as f:
            pickle.dump(psi, f)
        with open(os.path.join(ROOT, 'data', 'summary.jsonl'), 'a') as f:
            f.write(json.dumps(res) + '\n')
        print('RESULT', json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
