"""Triakis triangular lattice S Heisenberg AF: lattice, bond table, TeNPy model.

Request coordinates: a1=(1,0), a2=(1/2, sqrt3/2), R = x a1 + y a2.
Unit cell sites: 0 = corner at R, 1 = up-center of (R, R+a1, R+a2),
2 = down-center of (R+a1, R+a2, R+a1+a2).

Cylinder: circumference Ly*a1 (periodic, Ly multiple of 3), length along a2.
In TeNPy the first lattice direction is the (infinite) length direction, so
TeNPy coordinates are (ring, x) = (y, x) and a request dx=(d1, d2) becomes
TeNPy dx=(d2, d1).  Default TeNPy order = lexicographic in (ring, x, u):
each ring is ordered corner, up, down at x=0, then x=1, ...
"""
import numpy as np
from tenpy.models.lattice import Lattice
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinSite

A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, np.sqrt(3) / 2])

# (site_i, site_j, (dx1, dx2), kind)  -- exactly the table of REQUEST sec. 2.2
BONDS = [
    (0, 0, (1, 0), 'J1'),
    (0, 0, (0, 1), 'J1'),
    (0, 0, (-1, 1), 'J1'),
    (1, 0, (0, 0), 'J2'),
    (1, 0, (1, 0), 'J2'),
    (1, 0, (0, 1), 'J2'),
    (2, 0, (1, 0), 'J2'),
    (2, 0, (0, 1), 'J2'),
    (2, 0, (1, 1), 'J2'),
]


def sublattice(x, y):
    """3-sublattice index of corner R = x a1 + y a2."""
    return (x + 2 * y) % 3


def torus_bonds(T1=(3, 0), T2=(-1, 2), with_centers=True):
    """Explicit bond dict {(i,j): J-multiplicity-by-kind} on a torus, built from BONDS.

    Returns (nsite, ncorner, bonds{(i,j): {'J1': n, 'J2': n}}, corner_coords, sub).
    Site numbering: corners 0..N-1, then centers (u=1,2 per corner cell).
    Used to cross-check the bond table against ed.py's independent construction.
    """
    M = np.array([T1, T2]).T.astype(float)
    Minv = np.linalg.inv(M)

    def red(p):
        a = Minv @ np.asarray(p, float)
        a = a - np.floor(a + 1e-9)
        return tuple(np.round(M @ a).astype(int))

    cells = sorted({red((x, y)) for x in range(-8, 9) for y in range(-8, 9)})
    N = len(cells)
    idx = {}
    for k, c in enumerate(cells):
        idx[(c, 0)] = k
    if with_centers:
        for k, c in enumerate(cells):
            idx[(c, 1)] = N + 2 * k
            idx[(c, 2)] = N + 2 * k + 1
    bonds = {}
    for c in cells:
        for ui, uj, (d1, d2), kind in BONDS:
            if not with_centers and (ui or uj):
                continue
            i = idx[(c, ui)]
            j = idx[(red((c[0] + d1, c[1] + d2)), uj)]
            key = tuple(sorted((i, j)))
            bonds.setdefault(key, {'J1': 0, 'J2': 0})[kind] += 1
    sub = [sublattice(*c) for c in cells]
    return len(idx), N, bonds, cells, sub


class TriakisCylinder(Lattice):
    """TeNPy lattice; Ls=[Lx rings, Ly]; unit cell [corner, up, down] (or [corner])."""

    def __init__(self, Lx, Ly, site, centers=True, **kwargs):
        if centers:
            sites = [site] * 3
            pos_req = [0 * A1, (A1 + A2) / 3, 2 * (A1 + A2) / 3]
        else:
            sites = [site]
            pos_req = [0 * A1]
        # TeNPy basis order: (length a2, circumference a1)
        kwargs.setdefault('basis', np.array([A2, A1]))
        kwargs.setdefault('positions', np.array(pos_req))
        pairs = {'J1': [], 'J2': []}
        for ui, uj, (d1, d2), kind in BONDS:
            if not centers and (ui or uj):
                continue
            pairs[kind].append((ui, uj, np.array([d2, d1])))
        kwargs.setdefault('pairs', pairs)
        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


class TriakisHeisenberg(CouplingMPOModel):
    """H = J1 sum_cc S.S + J2 sum_{corner-center} S.S on an (infinite) cylinder.

    Options: S, J1, J2, h, Ly, Lx (rings per MPS unit cell), bc_MPS, centers, conserve.
    """

    def init_sites(self, model_params):
        S = model_params.get('S', 1.5)
        conserve = model_params.get('conserve', 'Sz')
        return SpinSite(S=S, conserve=conserve)

    def init_lattice(self, model_params):
        site = self.init_sites(model_params)
        Lx = model_params.get('Lx', 1)
        Ly = model_params.get('Ly', 3)
        bc_MPS = model_params.get('bc_MPS', 'infinite')
        centers = model_params.get('centers', True)
        bc_x = 'periodic' if bc_MPS == 'infinite' else 'open'
        return TriakisCylinder(Lx, Ly, site, centers=centers, bc=[bc_x, 'periodic'],
                               bc_MPS=bc_MPS)

    def init_terms(self, model_params):
        J = {'J1': model_params.get('J1', 1.0), 'J2': model_params.get('J2', 1.0)}
        for kind in ('J1', 'J2'):
            if J[kind] == 0:
                continue
            for u1, u2, dx in self.lat.pairs[kind]:
                self.add_coupling(J[kind], u1, 'Sz', u2, 'Sz', dx)
                self.add_coupling(0.5 * J[kind], u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
        h = model_params.get('h', 0.0)          # Zeeman term -h sum_i S^z_i (field scan, task B)
        if h:
            for u in range(len(self.lat.unit_cell)):
                self.add_onsite(-h, u, 'Sz')


def ring_site_table(lat):
    """For each MPS index: (ring, x, u) in request coordinates (x along a1, ring = y)."""
    tab = []
    for i in range(lat.N_sites):
        ring, x, u = lat.mps2lat_idx(i)
        tab.append((int(ring), int(x), int(u)))
    return np.array(tab)
