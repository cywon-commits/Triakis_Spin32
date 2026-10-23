"""Tables and figures from data/summary.jsonl and data/<tag>/chi*.npz.

usage: python3 -m triakis.analyze
writes results/table.csv, results/table.md and figs/*.png
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from triakis.model import sublattice, A1, A2

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
COLS = ['S', 'J1', 'J2', 'Ly', 'm', 'nring', 'seed', 'chi', 'e_corner', 'trunc', 'sz_corner0', 'sz_corner1',
        'sz_corner2', 'sz_up', 'sz_down', 'xi_tm2', 'xi_tm0', 'xi_tm4', 'phase_tm2', 'S_ring0']


def load_summary():
    rows = {}
    with open(os.path.join(ROOT, 'data', 'summary.jsonl')) as f:
        for line in f:
            r = json.loads(line)
            if r['tag'].startswith('test'):
                continue
            r['S_ring0'] = r['S_ring'][0]
            rows[(r['tag'], r['chi'])] = r      # keep the latest entry
    return sorted(rows.values(), key=lambda r: (r['S'], r['J1'], r['J2'], not r['centers'], r['Ly'],
                                                r['m'], r['nring'], r['seed'], r['chi']))


def corr_profile(npz, Ly, centers=True):
    """Same-sublattice / other corner and corner-center C_perp, C_zz vs ring distance.

    Returns dict of arrays indexed by ring distance n = 0..R: averages over target
    sites on ring n of the given class, for the corner reference at ring 0, x=0.
    """
    d = np.load(npz)
    tab = d['tab']
    L = len(tab)
    nps = 3 * Ly if centers else Ly
    js = d['js_u0']
    ring = js // nps
    x = tab[js % L, 1]
    u = tab[js % L, 2]
    R = ring.max()
    out = {}
    for name, sel in (('same', (u == 0) & (sublattice(x, ring) == 0)),
                      ('other', (u == 0) & (sublattice(x, ring) != 0)),
                      ('center', u > 0)):
        for q in ('perp', 'zzc'):
            v = d[f'{q}_u0']
            prof = np.full(R + 1, np.nan)
            for n in range(R + 1):
                s = sel & (ring == n)
                if s.any():
                    prof[n] = v[s].mean()
            out[f'{q}_{name}'] = prof
    return out


def bond_energies(npz, Ly, centers=True):
    """<S.S> on A-A (corner-corner) and A-B (corner-center) bonds from ring-0 references."""
    d = np.load(npz)
    tab, sz = d['tab'], d['sz']
    L = len(tab)
    nps = 3 * Ly if centers else Ly

    def site(ring, x, u):
        cand = [j for j in range(L) if tab[j][1] == x % Ly and tab[j][2] == u and tab[j][0] == ring % (L // nps)]
        return cand[0] + (ring - ring % (L // nps)) * nps

    def SS(ref_u, ring, x, u):
        i0 = site(0, 0, ref_u)
        j = site(ring, x, u)
        k = np.nonzero(d[f'js_u{ref_u}'] == j)[0][0]
        return d[f'perp_u{ref_u}'][k] + d[f'zzc_u{ref_u}'][k] + sz[i0 % L] * sz[j % L]
    # corner (0,0) to corners (1,0), (0,1), (-1,1)   [tab x is along a1, ring along a2]
    aa = [SS(0, 0, 1, 0), SS(0, 1, 0, 0), SS(0, 1, -1, 0)]
    out = dict(SS_AA=float(np.mean(aa)), SS_AA_list=[float(v) for v in aa])
    if centers:
        # down center of cell (0,0): corners (1,0),(0,1),(1,1); up center (0,0): (1,0),(0,1)
        ab = [SS(2, 0, 1, 0), SS(2, 1, 0, 0), SS(2, 1, 1, 0), SS(1, 0, 1, 0), SS(1, 1, 0, 0)]
        out.update(SS_AB=float(np.mean(ab)), SS_AB_list=[float(v) for v in ab])
    return out


def structure_factor_A(npz, Ly, centers=True, nk=121):
    """Connected S^zS^z structure factor of A (corner) sites, from the ring-0 corner reference.

    S(q) = sum_dr C(dr) cos(q.dr) (symmetrised; on-site term omitted -> constant offset).
    Returns the (qx,qy) of the maximum and the values at Gamma, K, M.
    """
    d = np.load(npz)
    tab = d['tab']
    L = len(tab)
    nps = 3 * Ly if centers else Ly
    js = d['js_u0']
    ring = js // nps
    x = tab[js % L, 1]
    u = tab[js % L, 2]
    sel = u == 0
    r = np.outer(x[sel], A1) + np.outer(ring[sel], A2)
    C = d['zzc_u0'][sel]

    def S(q):
        return 2 * np.sum(C * np.cos(r @ q))
    b1 = 2 * np.pi * np.array([1, -1 / np.sqrt(3)])
    b2 = 2 * np.pi * np.array([0, 2 / np.sqrt(3)])
    pts = {'Gamma': 0 * b1, 'K': (2 * b1 + b2) / 3, 'M': b1 / 2}
    grid = [(i / nk) * b1 + (j / nk) * b2 for i in range(nk) for j in range(nk)]
    vals = np.array([S(q) for q in grid])
    qmax = grid[int(np.argmax(vals))]
    return dict(qmax=[float(v) for v in qmax], Smax=float(vals.max()),
                **{f'S_{k}': float(S(q)) for k, q in pts.items()})


def pick(rows, **kw):
    """Classical-init entry with the largest chi for the given parameters (None if absent)."""
    s = [r for r in rows if all(r.get(k) == v for k, v in kw.items()) and r['seed'] < 0]
    return max(s, key=lambda r: r['chi']) if s else None


def fig_scan(rows):
    """J_AA/J_AB scan on Ly=3 with 2-ring cells: e(m) per J and the plateau gap e(5)-e(4.5)."""
    base = [r for r in rows if r['S'] == 1.5 and r['J2'] == 1 and r['Ly'] == 3 and r['centers']
            and r['seed'] < 0]

    def e(J, m, chi):
        c = [r for r in base if r['J1'] == J and r['m'] == m and r['chi'] == chi and r['nring'] == 2]
        if not c and m == 4.5:   # at J=1 the 1-, 2- and 3-ring cells agree; allow the 1-ring value
            c = [r for r in base if r['J1'] == J and r['m'] == m and r['chi'] == chi]
        return c[0] if c else None
    J1s = sorted({r['J1'] for r in base})
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    out = []
    for J in J1s:
        ms = sorted({r['m'] for r in base if r['J1'] == J and r['nring'] == 2 and r['chi'] == 128} | {4.5})
        pts = [(m, e(J, m, 128)) for m in ms]
        pts = [(m, r) for m, r in pts if r is not None and m in (4.0, 4.5, 5.0, 5.5, 6.0, 6.5)]
        if len(pts) >= 2:
            e0 = min(r['e_ring'] for _, r in pts)
            axs[0].plot([m for m, _ in pts], [r['e_ring'] - e0 for _, r in pts], 'o-', label=f'J_AA={J:g}')
    for chi, mk in ((128, 'o-'), (256, 's--')):
        xs, ds, xi = [], [], []
        for J in J1s:
            a, b = e(J, 4.5, chi), e(J, 5.0, chi)
            if a and b:
                xs.append(J); ds.append(b['e_ring'] - a['e_ring']); xi.append(a['xi_tm2'])
                out.append(dict(J_AA=J, chi=chi, e45=a['e_ring'], e5=b['e_ring'], gap=b['e_ring'] - a['e_ring'],
                                xi_perp_45=a['xi_tm2'], cell45=a['nring']))
        axs[1].plot(xs, ds, mk, label=f'χ={chi}')
        axs[2].plot(xs, xi, mk, label=f'χ={chi}')
    axs[0].set_xlabel('m (Sz per ring)'); axs[0].set_ylabel('e_ring − min  (χ=128, 2-ring cells)')
    axs[0].set_title('e(m) for each J_AA/J_AB')
    axs[1].axhline(0, color='k', lw=0.8)
    axs[1].set_xlabel('J_AA / J_AB'); axs[1].set_ylabel('Δ₊ = e_ring(5) − e_ring(4.5)')
    axs[1].set_title('plateau gap (> 0: m = S/3 ferrimagnet)')
    axs[2].set_xlabel('J_AA / J_AB'); axs[2].set_ylabel('ξ⊥ at m=4.5 [ring spacings]'); axs[2].set_yscale('log')
    for ax in axs:
        ax.legend(fontsize=7)
    fig.suptitle('S=3/2, Ly=3 infinite cylinder: J_AA/J_AB scan')
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, 'figs', 'jscan.png'), dpi=130)
    plt.close(fig)
    return out


def fig_finite(path=None):
    path = path or os.path.join(ROOT, 'data', 'finite_results.jsonl')
    if not os.path.exists(path):
        return []
    res = [json.loads(l) for l in open(path)]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for Lx in sorted({r['Lx'] for r in res}):
        s = sorted([r for r in res if r['Lx'] == Lx], key=lambda r: r['Sz'])
        e0 = min(r['E'] for r in s)
        ax.plot([r['Sz'] for r in s], [r['E'] - e0 for r in s], 'o-',
                label=f"Lx={Lx} (S(N_B−N_A)={1.5 * (s[0]['nB'] - s[0]['nA']):g})")
    ax.set_xlabel('S^z_tot')
    ax.set_ylabel('E − E_min')
    ax.set_title('finite open cylinder, Ly=3, S=3/2, J_AA=J_AB')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, 'figs', 'finite_E_Sz.png'), dpi=130)
    plt.close(fig)
    return res


def write_table(rows):
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    with open(os.path.join(ROOT, 'results', 'table.csv'), 'w') as f:
        f.write(','.join(COLS + ['centers', 'tag']) + '\n')
        for r in rows:
            f.write(','.join(str(r.get(c)) for c in COLS) + f",{r['centers']},{r['tag']}\n")
    hdr = ['S', 'J_AA', 'J_AB', 'Ly', 'm', 'cell', 'init', 'χ', 'e/corner', 'trunc', '⟨Sz⟩ A,B,C', '⟨Sz⟩ up,dn',
           'ξ⊥', 'ξ∥', 'ξ(ΔSz=2)', 'S_ent']
    lines = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    f3 = lambda v: '—' if v is None else f'{v:.3f}'
    for r in rows:
        lines.append('| ' + ' | '.join([
            f"{r['S']:g}" + ('' if r['centers'] else ' (corners only)'), f"{r['J1']:g}", f"{r['J2']:g}", str(r['Ly']),
            f"{r['m']:g}", str(r['nring']), 'cl' if r['seed'] < 0 else f"rnd{r['seed']}", str(r['chi']),
            f"{r['e_corner']:.6f}", f"{r['trunc']:.1e}",
            ', '.join(f"{r[f'sz_corner{s}']:.3f}" for s in range(3)),
            f3(r['sz_up']) + ', ' + f3(r['sz_down']),
            f3(r.get('xi_tm2')), f3(r.get('xi_tm0')), f3(r.get('xi_tm4')), f"{r['S_ring0']:.3f}"]) + ' |')
    with open(os.path.join(ROOT, 'results', 'table.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return lines


def group(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def fig_em(rows):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for Ly, mk in ((3, 'o-'), (6, 's--')):
        sel = [r for r in rows if r['S'] == 1.5 and r['J2'] == 1 and r['J1'] == 1 and r['Ly'] == Ly
               and r['centers'] and r['seed'] < 0]
        for chi in sorted({r['chi'] for r in sel}):
            best = {}
            for r in sorted([r for r in sel if r['chi'] == chi], key=lambda r: r['nring']):
                best.setdefault(r['m'], r)
            ms = sorted(best)
            ax.plot([m * 3 / Ly for m in ms], [best[m]['e_corner'] for m in ms], mk, label=f'Ly={Ly}, χ={chi}')
    ax.axvline(4.5, color='gray', ls=':', lw=1)
    ax.axvline(6, color='gray', ls='--', lw=0.8)
    ax.text(4.52, ax.get_ylim()[1], ' S/3 per site', va='top', fontsize=7, color='gray')
    ax.text(6.02, ax.get_ylim()[1], ' classical', va='top', fontsize=7, color='gray')
    ax.set_xlabel('Sz per 3 corners (= per ring for Ly=3)')
    ax.set_ylabel('energy per corner (J)')
    ax.set_title('S=3/2, J_AA=J_AB: e(m)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, 'figs', 'e_of_m.png'), dpi=130)
    plt.close(fig)


def fig_xi(rows, cases):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for label, sel in cases:
        s = sorted(sel, key=lambda r: r['chi'])
        if not s:
            continue
        chis = [r['chi'] for r in s]
        l, = axs[0].plot(chis, [r.get('xi_tm2') for r in s], 'o-', label=label)
        axs[1].plot(chis, [r.get('xi_tm0') for r in s], 's-', color=l.get_color(), label=label)
    for ax, t in zip(axs, ('ξ⊥ (ΔSz=±1 sector)', 'ξ∥ (ΔSz=0 sector)')):
        ax.set_xscale('log', base=2)
        ax.set_xlabel('χ')
        ax.set_ylabel('ξ  [ring spacings, √3/2 each]')
        ax.set_title(t)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, 'figs', 'xi_vs_chi.png'), dpi=130)
    plt.close(fig)


def fig_corr(cases, fname, title):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for label, npz, Ly, centers, S in cases:
        p = corr_profile(npz, Ly, centers)
        n = np.arange(len(p['perp_same']))
        for ax, key, sub in ((axs[0], 'perp_same', 'C⊥ same sublattice'),
                             (axs[1], 'perp_other', 'C⊥ other-sublattice corners'),
                             (axs[2], 'zzc_same', 'connected Czz same sublattice')):
            v = p[key] / S**2
            ok = np.isfinite(v) & (n > 0)
            l, = ax.semilogy(n[ok], np.abs(v[ok]), '-', lw=1, label=label)
            pos = ok & (v > 0)
            neg = ok & (v < 0)
            ax.semilogy(n[pos], v[pos], 'o', ms=3, color=l.get_color())
            ax.semilogy(n[neg], -v[neg], 'x', ms=4, color=l.get_color())
            ax.set_title(sub + '  (o: +, x: −)')
    for ax in axs:
        ax.set_xlabel('ring separation n  (axial distance n·√3/2)')
        ax.set_ylabel('|C| / S²')
        ax.set_ylim(1e-12, 2)
    axs[0].axhline(5 / 9, color='gray', ls=':', lw=1)
    axs[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, 'figs', fname), dpi=130)
    plt.close(fig)


def latest_npz(tag, chi=None):
    fs = glob.glob(os.path.join(ROOT, 'data', tag, 'chi*.npz'))
    if not fs:
        return None, None
    chis = sorted(int(os.path.basename(f)[3:-4]) for f in fs)
    c = chi if chi in chis else chis[-1]
    return os.path.join(ROOT, 'data', tag, f'chi{c}.npz'), c


def verification_table():
    """ED reference vs DMRG on the 18-spin torus (sec. 3.0)."""
    ref_E = {(0.5, 0): -8.704151, (0.5, 1): -8.704151, (0.5, 2): -8.704151, (0.5, 3): -8.704151,
             (0.5, 4): -8.485800, (0.5, 5): -7.632865, (1, 10): -24.165890, (1, 9): -26.326669,
             (1, 8): -27.996179, (1, 7): -28.372395}
    ref_obs = {(0.5, 3): dict(mz_corner=-0.432, mz_center=0.716, perp_same=-0.193, perp_nn=0.070,
                              perp_cc_center=-0.623),
               (1, 7): dict(mz_corner=-0.487, mz_center=0.827, perp_same=-0.277, perp_nn=-0.012,
                            perp_cc_center=-0.368)}
    rows = {}
    f = os.path.join(ROOT, 'verification', 'torus_results.jsonl')
    if os.path.exists(f):
        for l in open(f):
            r = json.loads(l)
            rows[(r['twoS'] / 2, r['Sz'])] = r
    lines = ['| S | Sz | E (ED 기준) | E (DMRG) | 차이 |', '|---|---|---|---|---|']
    for (S, Sz), r in sorted(rows.items()):
        e = ref_E.get((S, Sz))
        lines.append(f"| {S:g} | {Sz:g} | {'—' if e is None else f'{e:.6f}'} | {r['E']:.6f} | "
                     f"{'—' if e is None else f'{r[chr(69)] - e:+.1e}'} |")
    lines += ['', '| 관측량 | S=1/2 ED | S=1/2 DMRG | S=1 ED | S=1 DMRG |', '|---|---|---|---|---|']
    for k in ('mz_corner', 'mz_center', 'perp_same', 'perp_nn', 'perp_cc_center'):
        c = [f"{ref_obs[(0.5, 3)][k]:+.3f}", f"{rows[(0.5, 3)][k]:+.3f}" if (0.5, 3) in rows else '—',
             f"{ref_obs[(1, 7)][k]:+.3f}", f"{rows[(1, 7)][k]:+.3f}" if (1, 7) in rows else '—']
        lines.append(f'| {k} | ' + ' | '.join(c) + ' |')
    with open(os.path.join(ROOT, 'results', 'verification.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return lines


def main_tags(rows):
    t = {}
    for r in rows:
        t.setdefault(r['tag'], []).append(r)
    return t


if __name__ == '__main__':
    os.makedirs(os.path.join(ROOT, 'figs'), exist_ok=True)
    rows = load_summary()
    print('\n'.join(write_table(rows)))
    print('\n'.join(verification_table()))
    fig_em(rows)
    scan = fig_scan(rows)
    fin = fig_finite()
    json.dump(dict(scan=scan, finite=fin), open(os.path.join(ROOT, 'results', 'derived.json'), 'w'), indent=1)
    tags = main_tags(rows)
    lab = lambda r: (f"S={r['S']:g}" + ('' if r['centers'] else ' corners only') + f" J_AA={r['J1']:g} J_AB={r['J2']:g} Ly={r['Ly']}"
                     f" m={r['m']:g} n={r['nring']}" + ('' if r['seed'] < 0 else f" rnd{r['seed']}"))
    # xi vs chi: main S=3/2 cases
    cases = [(lab(v[0]), v) for k, v in tags.items() if v[0]['S'] == 1.5 and v[0]['J2'] == 1 and v[0]['J1'] == 1 and v[0]['centers']]
    fig_xi(rows, cases)
    import shutil
    shutil.move(os.path.join(ROOT, 'figs', 'xi_vs_chi.png'), os.path.join(ROOT, 'figs', 'xi_vs_chi_S32.png'))
    cases = [(lab(v[0]), v) for k, v in tags.items() if not (v[0]['S'] == 1.5 and v[0]['J2'] == 1 and v[0]['centers'])]
    cases += [(lab(v[0]), v) for k, v in tags.items() if v[0]['S'] == 1.5 and v[0]['J2'] == 1 and v[0]['J1'] != 1]
    fig_xi(rows, cases)
    shutil.move(os.path.join(ROOT, 'figs', 'xi_vs_chi.png'), os.path.join(ROOT, 'figs', 'xi_vs_chi_controls.png'))
    # correlations: S=3/2 J2=1 at each m (largest chi), and chi dependence at the m* candidates
    def ccase(sel):
        out = []
        for k, v in tags.items():
            r = v[0]
            if sel(r):
                for rr in sorted(v, key=lambda x: x['chi']):
                    f, c = latest_npz(k, rr['chi'])
                    out.append((lab(r) + f' χ={c}', f, r['Ly'], r['centers'], r['S']))
        return out
    import sys as _s
    for name, sel, title in (
        ('corr_S32_Ly3.png', lambda r: r['S'] == 1.5 and r['J2'] == 1 and r['J1'] == 1 and r['centers'] and r['Ly'] == 3
         and r['seed'] < 0 and r['m'] in (4.5, 5), 'S=3/2, J2=J1, Ly=3'),
        ('corr_S32_Ly6.png', lambda r: r['S'] == 1.5 and r['J2'] == 1 and r['Ly'] == 6, 'S=3/2, J2=J1, Ly=6'),
        ('corr_controls.png', lambda r: not (r['S'] == 1.5 and r['J2'] == 1 and r['J1'] == 1 and r['centers']) and not (r['J1'] in (1.2, 1.3, 1.8, 0.8)),
         'controls (largest χ)')):
        cs = ccase(sel)
        if name == 'corr_controls.png':
            best = {}
            for c in cs:
                best[c[0].rsplit(' χ=', 1)[0]] = c
            cs = list(best.values())
        if cs:
            fig_corr(cs, name, title)
