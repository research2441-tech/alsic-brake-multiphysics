"""Property-to-performance study of Al-2024 / nano-SiC composites for a solid automotive brake disc.

Chain: measured strength and hardness (Al-2024 + 20 nm SiC, 0-4 wt%) -> micromechanical effective
properties -> transient axisymmetric thermal FE of repeated stops -> thermoelastic stress ->
Archard wear of the rubbing track -> harmonic modal analysis -> Monte Carlo uncertainty.
"""
import json, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from fem import mesh_rect, Thermal, edges_where, thermoelastic, plate_modes

# ------------------------------------------------------------------ verified experimental data (Al-2024 + nano SiC, ref [18] in paper)
WT = np.array([0, 1, 2, 3, 4.0])
YS = np.array([125.3, 146.3, 163.0, 188.2, 162.5])        # MPa, yield strength
BHN = np.array([90.2, 117.0, 131.2, 132.3, 120.0])        # Brinell hardness
RHO_M, RHO_P = 2780.0, 3210.0                             # kg/m3
ALPHA_M, ALPHA_P = 23.6e-6, 4.6e-6                        # 1/K
G_M = 28e9; B_BURG = 0.286e-9; D_P = 20e-9                # Pa, m, m
NU_M = 0.33
# nominal values with uncertainty ranges (sampled in the Monte Carlo study)
NOM = dict(E_p=410e9, nu_p=0.17, k_p=145.0, cp_p=925.0, k_m=121.0, cp_m=875.0, h=60.0, mu=0.40, Kred=0.5)
RANGE = dict(E_p=(370e9, 490e9), nu_p=(0.14, 0.21), k_p=(120.0, 170.0), cp_p=(670.0, 1180.0), k_m=(103.0, 139.0),
             cp_m=(790.0, 960.0), h=(30.0, 90.0), mu=(0.30, 0.50), Kred=(0.0, 1.0))
K_BASE = 9.56e-6        # dimensionless Archard coefficient of the unreinforced alloy (from A356 wear data, ref [41])
K_RATIO = 0.487         # wear-coefficient ratio, nanocomposite / base, from the same data set


def vol_frac(wt):
    w = wt / 100.0
    return (w / RHO_P) / (w / RHO_P + (1 - w) / RHO_M)


# ------------------------------------------------------------------ micromechanics
def effective(f, P):
    E_m = 2 * G_M * (1 + NU_M); K_m = E_m / (3 * (1 - 2 * NU_M))
    E_p, nu_p = P['E_p'], P['nu_p']; K_p = E_p / (3 * (1 - 2 * nu_p)); G_p = E_p / (2 * (1 + nu_p))
    K = K_m + f * (K_p - K_m) / (1 + (1 - f) * (K_p - K_m) / (K_m + 4 / 3 * G_M))
    zeta = G_M * (9 * K_m + 8 * G_M) / (6 * (K_m + 2 * G_M))
    G = G_M + f * (G_p - G_M) / (1 + (1 - f) * (G_p - G_M) / (G_M + zeta))
    E = 9 * K * G / (3 * K + G); nu = (3 * K - 2 * G) / (2 * (3 * K + G))
    rho = (1 - f) * RHO_M + f * RHO_P
    wm = (1 - f) * RHO_M / rho
    cp = wm * P['cp_m'] + (1 - wm) * P['cp_p']
    km, kp = P['k_m'], P['k_p']
    k = km * (kp + 2 * km + 2 * f * (kp - km)) / (kp + 2 * km - f * (kp - km))
    alpha = (ALPHA_M * K_m * (1 - f) + ALPHA_P * K_p * f) / (K_m * (1 - f) + K_p * f)
    return dict(E=E, nu=nu, rho=rho, cp=cp, k=k, alpha=alpha, G=G, K=K)


def ideal_strengthening(f, dT=730.0, beta=1.25):
    """ideal-dispersion increments (MPa): Orowan, CTE-mismatch dislocations, load transfer."""
    if f <= 0:
        return 0.0, 0.0, 0.0
    lam = D_P * ((1 / (2 * f)) ** (1 / 3) - 1)
    d_or = 0.13 * G_M * B_BURG / lam * np.log(D_P / (2 * B_BURG)) / 1e6
    rho_d = 12 * (ALPHA_M - ALPHA_P) * dT * f / (B_BURG * D_P * (1 - f))
    d_cte = np.sqrt(3) * beta * G_M * B_BURG * np.sqrt(rho_d) / 1e6
    d_lt = 0.5 * f * YS[0]
    return d_or, d_cte, d_lt


def efficiency_table():
    """strengthening efficiency: measured increment over the ideal Orowan + load-transfer increment.
    The CTE-mismatch term is reported but excluded from the reference (see paper)."""
    rows = []
    for wt, ys in zip(WT[1:], YS[1:]):
        f = vol_frac(wt)
        d_or, d_cte, d_lt = ideal_strengthening(f)
        ideal = d_or + d_lt
        meas = ys - YS[0]
        rows.append(dict(wt=wt, f=f, orowan=d_or, cte=d_cte, lt=d_lt, ideal=ideal, measured=meas, eff=meas / ideal))
    return rows


# ------------------------------------------------------------------ brake duty
GEOM = dict(r_in=0.060, r_p0=0.095, r_out=0.140, half_t=0.014, pad_angle=np.pi / 3)
DUTY = dict(mass=1200.0, front=0.65, v0=27.78, decel=0.6 * 9.81, r_wheel=0.30, n_stops=5, t_cool=45.0, T_amb=25.0)


def duty_quantities(mu):
    t_stop = DUTY['v0'] / DUTY['decel']
    torque = DUTY['mass'] * DUTY['decel'] * DUTY['r_wheel'] * DUTY['front'] / 2      # per front disc
    r0, r1 = GEOM['r_p0'], GEOM['r_out']; th = GEOM['pad_angle']
    # uniform pressure p on both pad faces: torque = 2 * mu * p * th * (r1^3 - r0^3) / 3
    p = torque / (2 * mu * th * (r1 ** 3 - r0 ** 3) / 3)
    return t_stop, torque, p


def simulate(wt, P, nr=28, nz=8, dt_brake=0.02, dt_cool=0.5, want_field=False):
    f = vol_frac(wt); M = effective(f, P)
    nodes, tris, rs, zs = mesh_rect(GEOM['r_in'], GEOM['r_out'], 0.0, GEOM['half_t'], nr, nz)
    th = Thermal(nodes, tris, M['k'], M['rho'] * M['cp'])
    top = edges_where(nodes, tris, lambda q: abs(q[1] - GEOM['half_t']) < 1e-12)
    w_top = th.surface_ops(top)
    rub = (nodes[:, 0] >= GEOM['r_p0'] - 1e-9) & np.isclose(nodes[:, 1], GEOM['half_t'])
    outer = edges_where(nodes, tris, lambda q: abs(q[0] - GEOM['r_out']) < 1e-12)
    w_conv = w_top + th.surface_ops(outer)
    t_stop, torque, p = duty_quantities(P['mu'])
    h = P['h']; Ta = DUTY['T_amb']
    T = np.full(th.n, Ta)
    Hc = sp.diags(h * w_conv)
    lu_b = spla.splu((sp.diags(th.C / dt_brake) + th.K + Hc).tocsc())
    lu_c = spla.splu((sp.diags(th.C / dt_cool) + th.K + Hc).tocsc())
    hb = np.interp(wt, WT, BHN) * 9.80665e6               # Pa
    Kw = K_BASE * (1.0 if wt == 0 else (1 - P['Kred'] * (1 - K_RATIO)))
    wear = np.zeros(th.n)
    Tmax = Ta; hist = []; t_now = 0.0; T_at_peak = T.copy()
    for s in range(DUTY['n_stops']):
        nb = int(round(t_stop / dt_brake))
        for i in range(nb):
            t = (i + 1) * dt_brake
            v = max(DUTY['v0'] - DUTY['decel'] * t, 0.0); om = v / DUTY['r_wheel']
            q = np.where(rub, P['mu'] * p * om * nodes[:, 0] * GEOM['pad_angle'] / (2 * np.pi), 0.0)  # W/m2 averaged per revolution
            T = lu_b.solve(th.C / dt_brake * T + q * w_top + h * w_conv * Ta)
            wear += np.where(rub, Kw * p * om * nodes[:, 0] * GEOM['pad_angle'] / (2 * np.pi) / hb * dt_brake, 0.0)
            t_now += dt_brake
            if T.max() > Tmax:
                Tmax = T.max(); T_at_peak = T.copy()
            if want_field and i % 5 == 0:
                hist.append((t_now, T.max(), T[rub].mean()))
        for i in range(int(round(DUTY['t_cool'] / dt_cool))):
            T = lu_c.solve(th.C / dt_cool * T + h * w_conv * Ta)
            t_now += dt_cool
            if want_field and i % 4 == 0:
                hist.append((t_now, T.max(), T[rub].mean()))
    # thermoelastic stress at the instant of peak temperature: radially free disc, mid-plane symmetry (u_z = 0)
    fix_r = np.array([], dtype=int)
    fix_z = np.where(np.isclose(nodes[:, 1], 0.0))[0]
    u, vm = thermoelastic(nodes, tris, M['E'], M['nu'], M['alpha'], T_at_peak - Ta, fix_r, fix_z)
    ys = np.interp(wt, WT, YS) * 1e6
    # modal: annular plate clamped at r_in, free at r_out, full thickness
    hth = 2 * GEOM['half_t']; Dp = M['E'] * hth ** 3 / (12 * (1 - M['nu'] ** 2))
    rg = np.linspace(GEOM['r_in'], GEOM['r_out'], 41)
    freqs = {n: float(plate_modes(rg, n, Dp, M['rho'] * hth, M['nu'], nmodes=1)[0] / (2 * np.pi)) for n in range(0, 6)}
    out = dict(wt=wt, f=f, Tmax=float(Tmax), vm_max=float(vm.max() / 1e6), SR=float(vm.max() / ys),
               wear_um=float(wear.max() * 1e6), f_min=float(min(freqs.values())), freqs=freqs, pressure=p / 1e6,
               torque=torque, t_stop=t_stop, **{k_: float(v_) for k_, v_ in M.items()})
    if want_field:
        out['hist'] = hist; out['nodes'] = nodes; out['tris'] = tris; out['T_peak'] = T_at_peak; out['vm'] = vm; out['wear_field'] = wear
    return out


def mesh_study():
    res = []
    for nr, nz in [(14, 4), (28, 8), (56, 16)]:
        o = simulate(3.0, NOM, nr=nr, nz=nz)
        res.append(dict(nr=nr, nz=nz, Tmax=o['Tmax'], vm=o['vm_max'], wear=o['wear_um'], f=o['f_min']))
    o = simulate(3.0, NOM, dt_brake=0.01, dt_cool=0.25)
    res.append(dict(nr=28, nz=8, dt='half', Tmax=o['Tmax'], vm=o['vm_max'], wear=o['wear_um'], f=o['f_min']))
    return res


def monte_carlo(n=150, seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        P = {k: rng.uniform(*RANGE[k]) for k in RANGE}
        samples.append(P)
    out = {str(wt): [] for wt in WT}
    for P in samples:
        for wt in WT:
            o = simulate(wt, P)
            out[str(wt)].append({k: o[k] for k in ('Tmax', 'vm_max', 'SR', 'wear_um', 'f_min', 'k', 'E', 'alpha')})
    return samples, out


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'nominal':
        eff = efficiency_table()
        nom = []
        for wt in WT:
            t = time.time(); o = simulate(wt, NOM, nr=56, nz=16, want_field=(wt in (0.0, 3.0)))
            o['time'] = time.time() - t
            if 'hist' in o:
                np.savez(f'field_{wt}.npz', nodes=o.pop('nodes'), tris=o.pop('tris'), T=o.pop('T_peak'), vm=o.pop('vm'),
                         wear=o.pop('wear_field'), hist=np.array(o.pop('hist')))
            nom.append(o)
        json.dump(dict(eff=eff, nominal=nom, mesh=mesh_study()), open('nominal.json', 'w'), indent=1, default=float)
        for o in nom:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items() if k != 'freqs'}, o['freqs'])
        for r in eff:
            print(r)
    elif what == 'mc':
        n = int(sys.argv[2])
        S, out = monte_carlo(n, seed=int(sys.argv[3]) if len(sys.argv) > 3 else 0)
        json.dump(dict(samples=S, out=out), open(f'mc_{n}.json', 'w'), default=float)
