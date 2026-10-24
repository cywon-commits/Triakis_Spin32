"""Task B: direct field scan without U(1) -- iDMRG of H0 - h sum S^z, measure m(h).

usage: python3 -m triakis.run_field --J1 1.0 --h 0.1 [--nring 2] [--chis 64 128] [--m0 4.5]
Appends one line per (h, chi) to data/field_results.jsonl.  The initial state is the
classical ferrimagnetic product state with Sz per ring = m0 (U(1) is not conserved, so the
magnetisation is free to change during the optimisation).
"""
import os, json, time, argparse, pickle
import numpy as np
import tenpy
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from triakis.model import TriakisHeisenberg, ring_site_table
from triakis.run_idmrg import initial_levels

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--S', type=float, default=1.5)
    ap.add_argument('--J1', type=float, default=1.0)
    ap.add_argument('--J2', type=float, default=1.0)
    ap.add_argument('--Ly', type=int, default=3)
    ap.add_argument('--h', type=float, required=True)
    ap.add_argument('--nring', type=int, default=2)
    ap.add_argument('--chis', type=int, nargs='+', default=[64, 128])
    ap.add_argument('--m0', type=float, default=4.5, help='Sz per ring of the initial product state')
    ap.add_argument('--max_sweeps', type=int, default=60)
    a = ap.parse_args(argv)
    tag = f'field_S{a.S:g}_Ly{a.Ly}_J1{a.J1:g}_J2{a.J2:g}_h{a.h:g}_n{a.nring}_m0{a.m0:g}'
    ddir = os.path.join(ROOT, 'data', tag)
    os.makedirs(ddir, exist_ok=True)
    tenpy.tools.misc.setup_logging(to_stdout='INFO', filename=os.path.join(ddir, 'log.txt'))
    M = TriakisHeisenberg(dict(S=a.S, J1=a.J1, J2=a.J2, h=a.h, Ly=a.Ly, Lx=a.nring,
                               bc_MPS='infinite', conserve=None))
    tab = ring_site_table(M.lat)
    psi = MPS.from_product_state(M.lat.mps_sites(), initial_levels(a.S, tab, a.m0, a.nring), bc='infinite')
    nps = 3 * a.Ly
    for i, chi in enumerate(a.chis):
        t0 = time.time()
        p = dict(mixer=True, mixer_params=dict(amplitude=1e-4 if i == 0 else 1e-6, decay=1.5,
                                               disable_after=12 if i == 0 else 4),
                 max_E_err=1e-9, max_S_err=1e-6, min_sweeps=8 if i == 0 else 4, max_sweeps=a.max_sweeps,
                 trunc_params=dict(chi_max=chi, svd_min=1e-10))
        if i == 0:
            ramp = [c for c in (16, 32, 64, 128) if c < chi] + [chi]
            p['chi_list'] = {2 * k: c for k, c in enumerate(ramp)}
        eng = dmrg.TwoSiteDMRGEngine(psi, M, p)
        E, psi = eng.run()
        sz = np.real(psi.expectation_value('Sz'))
        m_site = float(np.mean(sz))
        res = dict(tag=tag, S=a.S, J1=a.J1, J2=a.J2, Ly=a.Ly, h=a.h, nring=a.nring, m0=a.m0, chi=chi,
                   e_site=float(E), e0_site=float(E) + a.h * m_site, m_site=m_site, m_ring=m_site * nps,
                   sz_corner=float(np.mean(sz[tab[:, 2] == 0])), sz_center=float(np.mean(sz[tab[:, 2] != 0])),
                   trunc=float(eng.sweep_stats['max_trunc_err'][-1]), time=time.time() - t0)
        with open(os.path.join(ROOT, 'data', 'field_results.jsonl'), 'a') as f:
            f.write(json.dumps(res) + '\n')
        with open(os.path.join(ddir, 'psi.pkl'), 'wb') as f:
            pickle.dump(psi, f)
        print('RESULT', json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
