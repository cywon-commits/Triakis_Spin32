"""Decorated triangulations: A = vertices of a torus triangulation, B = one spin per triangle.

Reproduces the triangulation-independent statements of REQUEST3 sec. 2 with numpy/scipy:
  (1) Euler: F = 2V, E = 3V, deg(v) = #triangles(v)  ->  (N_B - N_A)/(N_A + N_B) = 1/3
  (2) Ising (sigma = +-1, centers optimal): flat manifold at J_AA = J_AB, degeneracy 2 at J_AB = 1.2 J_AA
  (3) O(lambda^2) potential  E2 = -S^2 sum_{<ij>: s_i s_j = -1} 1/(S(|n_i|+|n_j|) - 1)
      over the Ising manifold (J_AA = J_AB = 1): minimiser and degeneracy
  (4) classical Heisenberg: minimise sum_T [(J_AA/4)(|L_T|^2 - 3S^2) - J_AB S |L_T|] over A spins
Lattices: triangular L x L, tetrakis square (Union Jack) L x L.

usage: python3 general_triangulation.py
"""
import itertools
import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------- lattices
def triangular(L):
    idx = lambda x, y: (x % L) + L * (y % L)
    tris = []
    for x in range(L):
        for y in range(L):
            tris.append((idx(x, y), idx(x + 1, y), idx(x, y + 1)))
            tris.append((idx(x + 1, y), idx(x, y + 1), idx(x + 1, y + 1)))
    return L * L, tris


def tetrakis(L):
    """Square-lattice corners (x,y) plus one face-centre per square, each square cut into 4 triangles."""
    corner = lambda x, y: (x % L) + L * (y % L)
    centre = lambda x, y: L * L + (x % L) + L * (y % L)
    tris = []
    for x in range(L):
        for y in range(L):
            c = centre(x, y)
            a, b, d, e = corner(x, y), corner(x + 1, y), corner(x + 1, y + 1), corner(x, y + 1)
            tris += [(a, b, c), (b, d, c), (d, e, c), (e, a, c)]
    return 2 * L * L, tris


def edges_of(tris):
    cnt = {}
    for t in tris:
        for i, j in itertools.combinations(t, 2):
            k = tuple(sorted((i, j)))
            cnt[k] = cnt.get(k, 0) + 1
    return cnt


def euler_check(V, tris):
    E = edges_of(tris)
    F = len(tris)
    deg = np.zeros(V, int)
    for i, j in E:
        deg[i] += 1; deg[j] += 1
    ntri = np.zeros(V, int)
    for t in tris:
        for v in t:
            ntri[v] += 1
    ok = (F == 2 * V) and (len(E) == 3 * V) and all(c == 2 for c in E.values()) and np.all(deg == ntri)
    return dict(V=V, E=len(E), F=F, F_eq_2V=F == 2 * V, E_eq_3V=len(E) == 3 * V,
                each_edge_in_2_tris=all(c == 2 for c in E.values()), deg_eq_ntri=bool(np.all(deg == ntri)),
                degrees=sorted(set(deg.tolist())), ratio=(F - V) / (V + F), ok=bool(ok))


# ---------------------------------------------------------------- Ising
def all_configs(V):
    c = ((np.arange(2 ** V)[:, None] >> np.arange(V)[None, :]) & 1).astype(np.int8)
    return 1 - 2 * c          # +-1


def ising_energies(V, tris, JAA, JAB, confs):
    E = np.zeros(len(confs))
    for (i, j) in edges_of(tris):
        E += JAA * confs[:, i] * confs[:, j]
    for t in tris:
        E -= JAB * np.abs(confs[:, t[0]] + confs[:, t[1]] + confs[:, t[2]])
    return E


def lambda2_energies(V, tris, S, confs):
    """O(lambda^2) potential on the Ising manifold; centres sigma_B = -sign(sum of its A)."""
    nA = np.zeros((len(confs), V))
    sB = []
    for t in tris:
        h = confs[:, t[0]] + confs[:, t[1]] + confs[:, t[2]]
        sB.append(-np.sign(h))
    sB = np.array(sB).T                     # (nconf, F)
    E = list(edges_of(tris))
    for (i, j) in E:                        # A-A neighbours
        nA[:, i] += confs[:, j]; nA[:, j] += confs[:, i]
    for k, t in enumerate(tris):            # A-B neighbours
        for v in t:
            nA[:, v] += sB[:, k]
    nA *= confs
    nB = np.array([sB[:, k] * (confs[:, t[0]] + confs[:, t[1]] + confs[:, t[2]]) for k, t in enumerate(tris)]).T
    E2 = np.zeros(len(confs))
    bad = 0
    for (i, j) in E:
        anti = confs[:, i] * confs[:, j] == -1
        den = S * (np.abs(nA[:, i]) + np.abs(nA[:, j])) - 1
        bad += int(np.sum(anti & (den <= 0)))
        E2 -= np.where(anti & (den > 0), S * S / np.where(den > 0, den, 1), 0)
    for k, t in enumerate(tris):
        for v in t:
            anti = confs[:, v] * sB[:, k] == -1
            den = S * (np.abs(nA[:, v]) + np.abs(nB[:, k])) - 1
            bad += int(np.sum(anti & (den <= 0)))
            E2 -= np.where(anti & (den > 0), S * S / np.where(den > 0, den, 1), 0)
    return E2, bad


# ---------------------------------------------------------------- classical Heisenberg
def heisenberg_min(V, tris, JAA, JAB, S=1.0, starts=12, seed=0):
    T = np.array(tris)
    rng = np.random.default_rng(seed)

    def energy(x):
        th, ph = x[:V], x[V:]
        s = S * np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
        Lv = s[T].sum(1)
        Ln = np.sqrt((Lv ** 2).sum(1) + 1e-14)
        return np.sum(JAA / 4 * (Ln ** 2 - 3 * S * S) - JAB * S * Ln)
    best = None
    for _ in range(starts):
        x0 = np.concatenate([np.arccos(rng.uniform(-1, 1, V)), rng.uniform(0, 2 * np.pi, V)])
        r = minimize(energy, x0, method='L-BFGS-B', options=dict(maxiter=20000, gtol=1e-10))
        if best is None or r.fun < best.fun:
            best = r
    th, ph = best.x[:V], best.x[V:]
    s = S * np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
    Ln = np.linalg.norm(s[T].sum(1), axis=1)
    return dict(E_per_tri=best.fun / len(tris), L_mean=Ln.mean(), L_std=Ln.std())


if __name__ == '__main__':
    lat = {'triangular L=3': triangular(3), 'triangular L=4': triangular(4),
           'tetrakis L=3': tetrakis(3), 'tetrakis L=4': tetrakis(4)}
    print('== (1) Euler checks')
    for name, (V, tris) in lat.items():
        print(f'  {name:16s}', euler_check(V, tris))

    print('== (2) Ising manifold (sigma=+-1, centres optimal)')
    for name in ('triangular L=3', 'tetrakis L=3'):
        V, tris = lat[name]
        confs = all_configs(V)
        for JAA, JAB in ((1, 1), (1, 1.2)):
            E = ising_energies(V, tris, JAA, JAB, confs)
            nsite = V + len(tris)
            emin = E.min()
            print(f'  {name:16s} J_AA={JAA} J_AB={JAB}: configs {len(E)}, distinct E {len(np.unique(np.round(E, 9)))}, '
                  f'E_min/site {emin / nsite:+.4f}, degeneracy {np.sum(np.abs(E - emin) < 1e-9)}')

    print('== (3) O(lambda^2) potential on the Ising manifold, J_AA = J_AB')
    for name in ('triangular L=3', 'tetrakis L=3'):
        V, tris = lat[name]
        confs = all_configs(V)
        nsite = V + len(tris)
        for S in (1.0, 1.5):
            E2, bad = lambda2_energies(V, tris, S, confs)
            emin = E2.min()
            arg = np.nonzero(np.abs(E2 - emin) < 1e-9)[0]
            ferro = [bool(np.all(confs[a] == confs[a][0])) for a in arg]
            print(f'  {name:16s} S={S}: min E2/site {emin / nsite:+.4f} (pred {-2 * S * S / (3 * S - 1):+.4f}), '
                  f'degeneracy {len(arg)}, minima are A-ferro: {all(ferro)}, nonpositive denominators skipped: {bad}')

    print('== (4) classical Heisenberg (S=1): |L_T| and energy per triangle')
    Js = [0.4, 0.6, 2 / 3, 0.7, 0.8, 1.0, 1.2]
    print('  J_AA/J_AB      ' + ' '.join(f'{j:7.3f}' for j in Js))
    print('  |L| predicted  ' + ' '.join(f'{min(3.0, 2.0 / j):7.3f}' for j in Js))
    for name in ('triangular L=3', 'tetrakis L=4'):
        V, tris = lat[name]
        res = [heisenberg_min(V, tris, j, 1.0) for j in Js]
        print(f'  |L| {name:11s}' + ' '.join(f'{r["L_mean"]:7.3f}' for r in res) +
              '   (max std %.1e)' % max(r['L_std'] for r in res))
        print(f'  E/T {name:11s}' + ' '.join(f'{r["E_per_tri"]:7.3f}' for r in res))

    print('== (5) quantum gain / classical loss = 12/(3S-1)')
    for S in (0.5, 1, 1.5, 2, 2.5, 3, 4, 13 / 3):
        print(f'  S={S:5.3f}: {12 / (3 * S - 1):6.2f}')
