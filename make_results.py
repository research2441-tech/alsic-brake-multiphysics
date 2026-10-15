import json
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from study import WT, YS, BHN, RANGE, NOM, effective, vol_frac, GEOM

plt.rcParams.update({'font.family': 'Liberation Serif', 'font.size': 8, 'mathtext.fontset': 'stix', 'savefig.dpi': 400,
                     'axes.linewidth': 0.6, 'hatch.linewidth': 0.5})
nom = json.load(open('nominal.json')); mc = json.load(open('mc_150.json'))
S = mc['samples']; OUT = mc['out']


def desirability(row, ref):
    """composite desirability (geometric mean) of four duty indicators, each scaled against the unreinforced disc."""
    d_sr = np.clip(ref['SR'] / row['SR'], 0, None)             # lower stress ratio is better
    d_w = np.clip(ref['wear_um'] / row['wear_um'], 0, None)     # lower wear is better
    d_f = row['f_min'] / ref['f_min']                           # higher first frequency is better
    d_t = ref['Tmax'] / row['Tmax']                             # lower peak temperature is better
    return float((d_sr * d_w * d_f * d_t) ** 0.25)


def mc_summary():
    out = {}
    n = len(S)
    best = np.zeros(len(WT))
    for i in range(n):
        ref = OUT['0.0'][i]
        D = [desirability(OUT[str(w)][i], ref) for w in WT]
        best[int(np.argmax(D))] += 1
    for w in WT:
        rows = OUT[str(w)]
        out[str(w)] = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])),
                           float(np.percentile([r[k] for r in rows], 5)), float(np.percentile([r[k] for r in rows], 95)))
                       for k in ('Tmax', 'vm_max', 'SR', 'wear_um', 'f_min', 'k', 'E', 'alpha')}
        # probability of improvement against the unreinforced disc in the same sample
        out[str(w)]['P_SR_better'] = float(np.mean([OUT[str(w)][i]['SR'] < OUT['0.0'][i]['SR'] for i in range(n)]))
        out[str(w)]['P_T_worse_1K'] = float(np.mean([OUT[str(w)][i]['Tmax'] > OUT['0.0'][i]['Tmax'] + 1 for i in range(n)]))
        out[str(w)]['D_mean'] = float(np.mean([desirability(OUT[str(w)][i], OUT['0.0'][i]) for i in range(n)]))
    out['p_best'] = (best / n).tolist()
    # Spearman sensitivity at 3 wt%
    keys = list(RANGE.keys())
    X = np.array([[s[k] for k in keys] for s in S])
    sens = {}
    for outk in ('Tmax', 'SR', 'wear_um', 'f_min'):
        y = np.array([r[outk] for r in OUT['3.0']])
        sens[outk] = {k: float(spearmanr(X[:, j], y).correlation) for j, k in enumerate(keys)}
    out['sens'] = sens
    return out


def fig_fields():
    d0 = np.load('field_0.0.npz'); d3 = np.load('field_3.0.npz')
    fig = plt.figure(figsize=(3.35, 3.75))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.62, 0.62, 1.25], width_ratios=[1, 0.035], hspace=0.80, wspace=0.05)
    n = d3['nodes'] * 1e3; tri = mtri.Triangulation(n[:, 0], n[:, 1], d3['tris'])
    ax = fig.add_subplot(gs[0, 0]); cax = fig.add_subplot(gs[0, 1])
    cs = ax.tricontourf(tri, d3['T'], levels=14, cmap='Greys')
    ax.tricontour(tri, d3['T'], levels=7, colors='black', linewidths=0.3)
    cb = fig.colorbar(cs, cax=cax); cb.ax.tick_params(labelsize=5.3); cb.set_label('T (°C)', fontsize=5.8)
    cb.set_ticks(np.arange(np.ceil(d3['T'].min() / 20) * 20, d3['T'].max(), 20)); cb.ax.set_yticklabels([f'{int(t)}' for t in cb.get_ticks()])
    ax.set_title('(a) temperature at peak, 3 wt% SiC', fontsize=7, pad=2); ax.set_ylabel('z (mm)', fontsize=6.3)
    ax.tick_params(labelsize=5.8); ax.set_xticklabels([]); ax.set_xlim(60, 140)
    ax = fig.add_subplot(gs[1, 0]); cax = fig.add_subplot(gs[1, 1])
    vm = d3['vm'] / 1e6
    tp = ax.tripcolor(tri, facecolors=vm, cmap='Greys', edgecolors='none', rasterized=True)
    cb = fig.colorbar(tp, cax=cax); cb.ax.tick_params(labelsize=5.3); cb.set_label('σ$_{vM}$ (MPa)', fontsize=5.8)
    ax.set_title('(b) von Mises stress at peak, 3 wt% SiC', fontsize=7, pad=2); ax.set_xlabel('r (mm)', fontsize=6.3); ax.set_ylabel('z (mm)', fontsize=6.3)
    ax.tick_params(labelsize=5.8); ax.set_xlim(60, 140)
    ax = fig.add_subplot(gs[2, :])
    for d, ls, lab in [(d0, '--', '0 wt%'), (d3, '-', '3 wt% SiC')]:
        h = d['hist']
        ax.plot(h[:, 0], h[:, 1], ls, color='black', lw=0.8, label=lab + ', peak')
        ax.plot(h[:, 0], h[:, 2], ls, color='0.55', lw=0.8, label=lab + ', track mean')
    ax.set_xlabel('Time (s)', fontsize=6.3); ax.set_ylabel('T (°C)', fontsize=6.3); ax.tick_params(labelsize=5.8)
    ax.set_title('(c) five-stop duty cycle', fontsize=7, pad=2)
    ax.legend(fontsize=5.4, ncol=2, frameon=False, loc='upper left', handlelength=2.2)
    ax.set_ylim(20, 300)
    ax.grid(ls=':', lw=0.4, color='0.6'); ax.spines[['top', 'right']].set_visible(False)
    fig.subplots_adjust(left=0.11, right=0.90, top=0.95, bottom=0.10)
    fig.savefig('fig2_fields.png'); plt.close(fig)


def fig_mc(summ):
    fig, axes = plt.subplots(2, 2, figsize=(3.35, 3.0))
    items = [('SR', 'Stress-to-yield ratio'), ('wear_um', 'Track wear depth (µm)'), ('Tmax', 'Peak temperature (°C)'), ('f_min', 'First frequency (Hz)')]
    for ax, (k, lab) in zip(axes.ravel(), items):
        data = [[r[k] for r in OUT[str(w)]] for w in WT]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color='black', lw=0.8),
                        whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5), boxprops=dict(lw=0.5))
        for patch, w in zip(bp['boxes'], WT):
            patch.set_facecolor('0.3' if w == 3.0 else 'white')
        ax.set_xticks(range(1, 6)); ax.set_xticklabels([f'{int(w)}' for w in WT], fontsize=6)
        ax.set_ylabel(lab, fontsize=6.3); ax.tick_params(labelsize=5.8)
        ax.grid(axis='y', ls=':', lw=0.4, color='0.6'); ax.spines[['top', 'right']].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel('SiC content (wt%)', fontsize=6.5)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.12, wspace=0.42, hspace=0.25)
    fig.savefig('fig3_mc.png'); plt.close(fig)


def fig_sens_modes(summ):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.35, 1.9), gridspec_kw={'width_ratios': [1.25, 1]})
    names = {'E_p': '$E_p$', 'nu_p': r'$\nu_p$', 'k_p': '$k_p$', 'cp_p': '$c_{p,p}$', 'k_m': '$k_m$', 'cp_m': '$c_{p,m}$', 'h': '$h$', 'mu': r'$\mu$', 'Kred': r'$\kappa$'}
    keys = list(names)
    outs = [('Tmax', '', 'white'), ('SR', '////', 'white'), ('wear_um', '', '0.3')]
    y = np.arange(len(keys)); hgt = 0.27
    for i, (o, hatch, col) in enumerate(outs):
        vals = [summ['sens'][o][k] for k in keys]
        a1.barh(y + (i - 1) * hgt, vals, hgt * 0.92, color=col, edgecolor='black', lw=0.4, hatch=hatch,
                label={'Tmax': '$T_{\\max}$', 'SR': 'SR', 'wear_um': 'wear'}[o])
    a1.set_yticks(y); a1.set_yticklabels([names[k] for k in keys], fontsize=6.2); a1.axvline(0, color='black', lw=0.5)
    a1.set_xlim(-1, 1); a1.set_xlabel('Spearman ρ (3 wt%)', fontsize=6.3); a1.tick_params(labelsize=5.8)
    a1.legend(fontsize=5.4, frameon=False, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=3, handlelength=1.2, columnspacing=0.6)
    a1.invert_yaxis(); a1.spines[['top', 'right']].set_visible(False)
    base = next(r for r in nom['nominal'] if r['wt'] == 0.0)['freqs']
    for n_, ls, mk in [('0', '-', 'o'), ('5', '--', 's')]:
        inc = [100 * (next(r for r in nom['nominal'] if r['wt'] == w)['freqs'][n_] / base[n_] - 1) for w in WT]
        a2.plot(WT, inc, ls, marker=mk, ms=3, color='black', lw=0.8, mfc='white' if n_ == '0' else 'black',
                label=f'$n={n_}$ ({base[n_] / 1e3:.2f} kHz at 0 wt%)')
    a2.set_xlabel('SiC content (wt%)', fontsize=6.3); a2.set_ylabel('Frequency increase (%)', fontsize=6.3); a2.tick_params(labelsize=5.8)
    a2.legend(fontsize=5.2, frameon=False, loc='upper left'); a2.grid(ls=':', lw=0.4, color='0.6'); a2.spines[['top', 'right']].set_visible(False)
    a2.set_ylim(0, 3.0)
    a1.set_title('(a)', fontsize=7, pad=12); a2.set_title('(b)', fontsize=7, pad=12)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.82, bottom=0.2, wspace=0.42)
    fig.savefig('fig4_sens_modes.png'); plt.close(fig)


if __name__ == '__main__':
    summ = mc_summary()
    fig_fields(); fig_mc(summ); fig_sens_modes(summ)
    json.dump(summ, open('mc_summary.json', 'w'), indent=1)
    for w in WT:
        s = summ[str(w)]
        print(w, {k: (round(v[0], 3), round(v[1], 3)) if isinstance(v, tuple) or isinstance(v, list) else round(v, 3) for k, v in s.items()})
    print('p_best', summ['p_best'])
    for o, v in summ['sens'].items():
        print(o, {k: round(x, 2) for k, x in v.items()})
