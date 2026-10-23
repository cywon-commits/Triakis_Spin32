"""Brief step 1: classical Ising check of the bond list (sigma = +-1, 'S=1 normalisation').

For every A (corner) configuration the B (center) spins are chosen optimally
(sigma_B = -sign(sum of its three A neighbours), always defined since the sum is odd).
Expectation: J_AA = J_AB -> all 2^N_A configurations have E = -3 J N_A;
J_AB = 1.2 J_AA -> unique minimum up to time reversal (A ferro), degeneracy 2.
Uses the same bond table (triakis/model.py BONDS) on tori of several shapes.
"""
import sys, os, itertools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from triakis.model import torus_bonds


def check(T1, T2, JAA, JAB):
    ns, N, b, cells, sub = torus_bonds(T1, T2)
    aa = [(i, j, n['J1']) for (i, j), n in b.items() if n['J1']]
    ab = {}
    for (i, j), n in b.items():
        if n['J2']:
            ab.setdefault(j, []).append((i, n['J2']))          # j >= N is the center
    assert all(len(v) == 3 for v in ab.values()) and len(ab) == 2 * N
    Es = []
    for conf in itertools.product((1, -1), repeat=N):
        s = np.array(conf)
        E = JAA * sum(n * s[i] * s[j] for i, j, n in aa)
        for c, nb in ab.items():
            h = sum(n * s[i] for i, n in nb)
            E += -JAB * abs(h)
        Es.append(E)
    Es = np.array(Es)
    return N, Es


if __name__ == '__main__':
    for T1, T2 in (((3, 0), (-1, 2)), ((3, 0), (0, 3)), ((4, 0), (0, 4)), ((4, 0), (-2, 4))):
        for JAA, JAB in ((1, 1), (1, 1.2), (1.2, 1)):
            N, Es = check(T1, T2, JAA, JAB)
            emin = Es.min()
            print(f'torus T1={T1} T2={T2} N_A={N:2d} J_AA={JAA} J_AB={JAB}: '
                  f'E_min={emin:+.3f} (-3N={-3 * N}), E_max={Es.max():+.3f}, '
                  f'degeneracy of minimum = {np.sum(np.abs(Es - emin) < 1e-9)} / {len(Es)}')
