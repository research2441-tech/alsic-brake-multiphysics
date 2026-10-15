"""In-house axisymmetric finite-element solvers for the nanocomposite brake-disc study.

 - transient heat conduction (linear triangles, r-weighted, implicit Euler, lumped capacity)
 - axisymmetric thermoelasticity (constant-strain triangles, strains [e_r, e_z, e_t, g_rz])
 - harmonic modal analysis of an annular Kirchhoff plate (Hermite cubic elements in r,
   w(r, theta) = W(r) cos(n theta))
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh


# ------------------------------------------------------------------ mesh
def mesh_rect(r0, r1, z0, z1, nr, nz, rgrade=None):
    rs = np.linspace(r0, r1, nr + 1) if rgrade is None else rgrade
    zs = np.linspace(z0, z1, nz + 1)
    R, Z = np.meshgrid(rs, zs, indexing='ij')
    nodes = np.c_[R.ravel(), Z.ravel()]
    idx = lambda i, j: i * (nz + 1) + j
    tris = []
    for i in range(len(rs) - 1):
        for j in range(nz):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            tris += [[a, b, c], [a, c, d]]
    return nodes, np.array(tris), rs, zs


def tri_geom(xy):
    (r1, z1), (r2, z2), (r3, z3) = xy
    A = 0.5 * ((r2 - r1) * (z3 - z1) - (r3 - r1) * (z2 - z1))
    b = np.array([z2 - z3, z3 - z1, z1 - z2]) / (2 * A)    # dN/dr
    c = np.array([r3 - r2, r1 - r3, r2 - r1]) / (2 * A)    # dN/dz
    return A, b, c


# ------------------------------------------------------------------ heat conduction
class Thermal:
    def __init__(self, nodes, tris, k, rho_c):
        n = len(nodes); self.n = n; self.nodes = nodes
        I, J, V = [], [], []
        C = np.zeros(n)
        for t in tris:
            A, b, c = tri_geom(nodes[t]); rc = nodes[t, 0].mean()
            Ke = k * (np.outer(b, b) + np.outer(c, c)) * A * 2 * np.pi * rc
            for a in range(3):
                C[t[a]] += rho_c * 2 * np.pi * A * (nodes[t[a], 0] + nodes[t, 0].sum()) / 12   # lumped r-weighted capacity
                for bb in range(3):
                    I.append(t[a]); J.append(t[bb]); V.append(Ke[a, bb])
        self.K = sp.csr_matrix((V, (I, J)), shape=(n, n))
        self.C = C

    def surface_ops(self, edges):
        """edges: list of (node_i, node_j) on a boundary. Returns lumped r-weighted area per node."""
        w = np.zeros(self.n)
        for i, j in edges:
            ri, rj = self.nodes[i, 0], self.nodes[j, 0]
            L = np.hypot(*(self.nodes[i] - self.nodes[j]))
            # integral of N_i * 2 pi r ds for linear N
            w[i] += 2 * np.pi * L * (2 * ri + rj) / 6
            w[j] += 2 * np.pi * L * (ri + 2 * rj) / 6
        return w


def edges_where(nodes, tris, pred):
    out = set()
    for t in tris:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            i, j = t[a], t[b]
            if pred(nodes[i]) and pred(nodes[j]):
                out.add((min(i, j), max(i, j)))
    return sorted(out)


# ------------------------------------------------------------------ thermoelasticity
def thermoelastic(nodes, tris, E, nu, alpha, dT, fix_r_nodes, fix_z_nodes):
    n = len(nodes); ndof = 2 * n
    lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu))
    D = np.array([[lam + 2 * mu, lam, lam, 0], [lam, lam + 2 * mu, lam, 0], [lam, lam, lam + 2 * mu, 0], [0, 0, 0, mu]])
    I, J, V = [], [], []; f = np.zeros(ndof); Bs = []
    for t in tris:
        A, b, c = tri_geom(nodes[t]); rc = nodes[t, 0].mean()
        B = np.zeros((4, 6))
        for a in range(3):
            B[0, 2 * a] = b[a]; B[1, 2 * a + 1] = c[a]; B[2, 2 * a] = 1 / (3 * rc)
            B[3, 2 * a] = c[a]; B[3, 2 * a + 1] = b[a]
        w = 2 * np.pi * rc * A
        Ke = B.T @ D @ B * w
        eth = alpha * dT[t].mean() * np.array([1, 1, 1, 0])
        fe = B.T @ D @ eth * w
        dofs = np.ravel([[2 * i, 2 * i + 1] for i in t])
        for a in range(6):
            f[dofs[a]] += fe[a]
            for bb in range(6):
                I.append(dofs[a]); J.append(dofs[bb]); V.append(Ke[a, bb])
        Bs.append((B, eth, dofs))
    K = sp.csr_matrix((V, (I, J)), shape=(ndof, ndof))
    fixed = np.r_[2 * np.asarray(fix_r_nodes), 2 * np.asarray(fix_z_nodes) + 1]
    free = np.setdiff1d(np.arange(ndof), fixed)
    u = np.zeros(ndof)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])
    vm = np.zeros(len(tris))
    for e, (B, eth, dofs) in enumerate(Bs):
        s = D @ (B @ u[dofs] - eth)      # [s_r, s_z, s_t, t_rz]
        sr, sz, st, trz = s
        vm[e] = np.sqrt(0.5 * ((sr - sz) ** 2 + (sz - st) ** 2 + (st - sr) ** 2) + 3 * trz ** 2)
    return u, vm


# ------------------------------------------------------------------ annular plate modal analysis
def _hermite(xi, h):
    N = np.array([1 - 3 * xi ** 2 + 2 * xi ** 3, h * (xi - 2 * xi ** 2 + xi ** 3), 3 * xi ** 2 - 2 * xi ** 3, h * (-xi ** 2 + xi ** 3)])
    dN = np.array([-6 * xi + 6 * xi ** 2, h * (1 - 4 * xi + 3 * xi ** 2), 6 * xi - 6 * xi ** 2, h * (-2 * xi + 3 * xi ** 2)]) / h
    d2N = np.array([-6 + 12 * xi, h * (-4 + 6 * xi), 6 - 12 * xi, h * (-2 + 6 * xi)]) / h ** 2
    return N, dN, d2N


def plate_modes(rgrid, n, D, rho_h, nu, inner='clamped', outer='free', nmodes=3):
    """natural angular frequencies (rad/s) of harmonic n for an annular plate."""
    ne = len(rgrid) - 1; ndof = 2 * (ne + 1)
    K = np.zeros((ndof, ndof)); Mm = np.zeros((ndof, ndof))
    gp, gw = np.polynomial.legendre.leggauss(6)
    for e in range(ne):
        r0, r1 = rgrid[e], rgrid[e + 1]; h = r1 - r0
        dofs = np.arange(2 * e, 2 * e + 4)
        for x, w in zip(gp, gw):
            xi = (x + 1) / 2; r = r0 + xi * h; wt = w * h / 2 * r
            N, dN, d2N = _hermite(xi, h)
            k_rr = d2N
            k_tt = dN / r - n ** 2 * N / r ** 2
            k_rt = n * (dN / r - N / r ** 2)
            Ke = (np.outer(k_rr + k_tt, k_rr + k_tt) - (1 - nu) * (np.outer(k_rr, k_tt) + np.outer(k_tt, k_rr))
                  + 2 * (1 - nu) * np.outer(k_rt, k_rt))
            K[np.ix_(dofs, dofs)] += D * Ke * wt
            Mm[np.ix_(dofs, dofs)] += rho_h * np.outer(N, N) * wt
    fixed = []
    if inner == 'clamped':
        fixed += [0, 1]
    elif inner == 'centre':           # solid plate centre regularity
        fixed += [1] if n == 0 else ([0] if n == 1 else [0, 1])
    if outer == 'clamped':
        fixed += [ndof - 2, ndof - 1]
    free = np.setdiff1d(np.arange(ndof), fixed)
    w2 = eigh(K[np.ix_(free, free)], Mm[np.ix_(free, free)], eigvals_only=True)
    return np.sqrt(np.clip(w2[:nmodes], 0, None))
