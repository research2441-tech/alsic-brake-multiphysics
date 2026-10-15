"""Verification of the three in-house solvers against analytical solutions (Table IV of the paper)."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from fem import mesh_rect, Thermal, edges_where, thermoelastic, plate_modes

# 1) slab under constant surface flux: surface rise = 2 q sqrt(t/pi) / sqrt(k rho c)
k, rc, q = 120.0, 2780 * 875.0, 1e6
nodes, tris, _, _ = mesh_rect(0.10, 0.11, 0, 0.02, 2, 80)
th = Thermal(nodes, tris, k, rc)
w = th.surface_ops(edges_where(nodes, tris, lambda p: abs(p[1] - 0.02) < 1e-12))
T = np.zeros(th.n); dt = 0.005; lu = spla.splu((sp.diags(th.C / dt) + th.K).tocsc())
for _ in range(int(1.0 / dt)):
    T = lu.solve(th.C / dt * T + q * w)
print('slab: FE %.2f K, analytical %.2f K' % (T[np.isclose(nodes[:, 1], 0.02)].mean(), 2 * q * np.sqrt(1 / np.pi) / np.sqrt(k * rc)))

# 2) free annulus under uniform heating: zero stress, u_r = alpha dT r
nodes, tris, _, _ = mesh_rect(0.06, 0.14, 0, 0.01, 16, 4)
u, vm = thermoelastic(nodes, tris, 74.5e9, 0.33, 23.6e-6, np.full(len(nodes), 100.0), [], [0])
print('annulus: max von Mises %.2e Pa, u_r(outer) %.4e vs %.4e m' % (vm.max(), u[2 * np.argmax(nodes[:, 0])], 23.6e-6 * 100 * 0.14))

# 3) clamped circular plate frequency parameters
r = np.linspace(0, 1, 81) ** 1.5; r[0] = 1e-6
for n, ref in [(0, 10.2158), (1, 21.26), (2, 34.88)]:
    lam = plate_modes(r, n, 1.0, 1.0, 0.3, inner='centre', outer='clamped', nmodes=1)[0]
    print('plate n=%d: FE %.4f, reference %.4f' % (n, lam, ref))
