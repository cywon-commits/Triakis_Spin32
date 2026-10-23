"""Tables and figures from data/summary.jsonl and data/<tag>/chi*.npz.

usage: python3 -m triakis.analyze
writes results/table.csv, results/table.md and figs/*.png
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from triakis.model import sublattice

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
COLS = ['S', 'J2', 'Ly', 'm', 'nring', 'seed', 'chi', 'e_corner', 'trunc', 'sz_corner0', 'sz_corner1',
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


def write_table(rows):
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    with open(os.path.join(ROOT, 'results', 'table.csv'), 'w') as f:
        f.write(','.join(COLS + ['centers', 'tag']) + '\n')
        for r in rows:
            f.write(','.join(str(r.get(c)) for c in COLS) + f",{r['centers']},{r['tag']}\n")
    hdr = ['S', 'J2', 'Ly', 'm', 'cell', 'init', 'χ', 'e/corner', 'trunc', '⟨Sz⟩ A,B,C', '⟨Sz⟩ up,dn',
           'ξ⊥', 'ξ∥', 'ξ(ΔSz=2)', 'S_ent']
    lines = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    f3 = lambda v: '—' if v is None else f'{v:.3f}'
    for r in rows:
        lines.append('| ' + ' | '.join([
            f"{r['S']:g}" + ('' if r['centers'] else ' (corners only)'), f"{r['J2']:g}", str(r['Ly']),
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
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, Ly in zip(axs, (3, 6)):
        sel = [r for r in rows if r['S'] == 1.5 and r['J2'] == 1 and r['J1'] == 1 and r['Ly'] == Ly
               and r['centers'] and r['seed'] < 0]
        for chi in sorted({r['chi'] for r in sel}):
            s = sorted([r for r in sel if r['chi'] == chi], key=lambda r: r['m'])
            # one entry per m: prefer the smallest cell
            best = {}
            for r in s:
                best.setdefault(r['m'], r)
            ms = sorted(best)
            ax.plot(ms, [best[m]['e_ring'] for m in ms], 'o-', label=f'χ={chi}')
        ax.set_xlabel('m = Sz per ring')
        ax.set_ylabel('energy per ring')
        ax.set_title(f'S=3/2, J2=J1, Ly={Ly}')
        ax.axvline(6, color='gray', ls=':', lw=1)
        ax.legend(fontsize=8)
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
    tags = main_tags(rows)
    lab = lambda r: (f"S={r['S']:g}" + ('' if r['centers'] else ' corners only') + f" J2={r['J2']:g} Ly={r['Ly']}"
                     f" m={r['m']:g} n={r['nring']}" + ('' if r['seed'] < 0 else f" rnd{r['seed']}"))
    # xi vs chi: main S=3/2 cases
    cases = [(lab(v[0]), v) for k, v in tags.items() if v[0]['S'] == 1.5 and v[0]['J2'] == 1 and v[0]['centers']]
    fig_xi(rows, cases)
    import shutil
    shutil.move(os.path.join(ROOT, 'figs', 'xi_vs_chi.png'), os.path.join(ROOT, 'figs', 'xi_vs_chi_S32.png'))
    cases = [(lab(v[0]), v) for k, v in tags.items() if not (v[0]['S'] == 1.5 and v[0]['J2'] == 1 and v[0]['centers'])]
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
        ('corr_S32_Ly3.png', lambda r: r['S'] == 1.5 and r['J2'] == 1 and r['centers'] and r['Ly'] == 3
         and r['seed'] < 0 and r['m'] in (4.5, 5), 'S=3/2, J2=J1, Ly=3'),
        ('corr_S32_Ly6.png', lambda r: r['S'] == 1.5 and r['J2'] == 1 and r['Ly'] == 6, 'S=3/2, J2=J1, Ly=6'),
        ('corr_controls.png', lambda r: not (r['S'] == 1.5 and r['J2'] == 1 and r['centers']),
         'controls (largest χ)')):
        cs = ccase(sel)
        if name == 'corr_controls.png':
            best = {}
            for c in cs:
                best[c[0].rsplit(' χ=', 1)[0]] = c
            cs = list(best.values())
        if cs:
            fig_corr(cs, name, title)
