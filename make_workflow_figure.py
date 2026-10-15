import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family': 'Liberation Serif', 'font.size': 8, 'mathtext.fontset': 'stix', 'savefig.dpi': 400})

def box(ax, x, y, w, h, t, fill='white', fs=6.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.015', lw=0.8, ec='black', fc=fill))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, linespacing=1.15)

def arr(ax, p, q, dashed=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=7, lw=0.8, color='black', shrinkA=0, shrinkB=0,
                                 linestyle=(0, (4, 2.5)) if dashed else '-'))

fig = plt.figure(figsize=(3.35, 3.55)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1.06); ax.axis('off')
L, R = 0.06, 0.87; W = R - L; h = 0.105
box(ax, L, 0.93, W, h, 'Verified experiments: Al-2024 + 20 nm SiC, 0–4 wt%\nyield strength $\\sigma_y(w)$, Brinell hardness $H_B(w)$', fill='#f2f2f2')
box(ax, L, 0.77, W, h, 'Micromechanics: Mori–Tanaka $E,\\nu$; Maxwell $k$; Turner $\\alpha$\nstrengthening diagnostic: efficiency $\\eta_s(w)$')
g12, g = 0.055, 0.03
wb = (W - g12 - 2 * g) / 4
xs = [L, L + wb + g12, L + 2 * wb + g12 + g, L + 3 * wb + g12 + 2 * g]
dutyR = xs[2] + wb
box(ax, L, 0.61, dutyR - L, h, 'Brake duty: torque $\\rightarrow$ pad pressure $p$\nfrictional flux $q(r,t)$, five stops, convection $h$')
labels = ['Transient\nheat FE\n$T(r,z,t)$', 'Thermo-\nelastic FE\n$\\sigma_{vM}$, SR', 'Archard\nwear\n$\\delta(r)$', 'Plate\nmodal\n$f_n$']
for x, t in zip(xs, labels):
    box(ax, x, 0.40, wb, 0.14, t, fs=6.2)
box(ax, L, 0.20, W, h, 'Composite desirability $D(w)$ and Monte Carlo statistics', fill='#f2f2f2')
box(ax, L, 0.04, W, h, 'Selection of the reinforcement level $w^{\\ast}$', fill='#f2f2f2')
cx = L + W / 2
arr(ax, (cx, 0.93), (cx, 0.77 + h + 0.004))
arr(ax, ((L + dutyR) / 2, 0.77), ((L + dutyR) / 2, 0.61 + h + 0.004))
arr(ax, (xs[3] + wb / 2, 0.77), (xs[3] + wb / 2, 0.54 + 0.004))           # elastic properties -> modal
arr(ax, (xs[0] + wb / 2, 0.61), (xs[0] + wb / 2, 0.54 + 0.004))           # duty -> heat
arr(ax, (xs[2] + wb / 2, 0.61), (xs[2] + wb / 2, 0.54 + 0.004))           # duty -> wear
arr(ax, (xs[0] + wb, 0.47), (xs[1] - 0.004, 0.47))                        # temperature -> stress
for x in xs:
    arr(ax, (x + wb / 2, 0.40), (x + wb / 2, 0.20 + h + 0.004))
arr(ax, (cx, 0.20), (cx, 0.04 + h + 0.004))
xl = 0.935; d = (0, (4, 2.5))
ax.plot([R + 0.004, xl], [0.20 + h / 2] * 2, color='black', lw=0.8, ls=d)
ax.plot([xl, xl], [0.20 + h / 2, 0.77 + h / 2], color='black', lw=0.8, ls=d)
arr(ax, (xl, 0.77 + h / 2), (R + 0.004, 0.77 + h / 2), dashed=True)
ax.text(0.968, 0.535, 'resample uncertain inputs (150 draws)', rotation=90, fontsize=5.8, ha='center', va='center')
fig.savefig('fig1_workflow.png'); plt.close(fig)
