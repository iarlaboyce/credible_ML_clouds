"""
Real-data delta-skill figure: (a) per-pixel correction scatter (inflated by
shared -re_mod06 retrieval noise), (b) segment-level within-granule
demeaned scatter, illustrating the collapse from corr~0.31 to rho~0.06
that motivates the segment-aggregated evaluation metric.

Usage: python make_segment_skill_figure.py [weights_name]
Output: figures/segment_skill_scatter.png
"""
import sys, glob, os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from real_data_inference import load_model, apply_granule_calibration
from plot_style import apply_style, panel_label, stats_box, SCATTER_KW, ONE_TO_ONE_KW, SAVE_KW

WEIGHTS_NAME = sys.argv[1] if len(sys.argv) > 1 else 'apivae_weights_v2b.pth'
AGE_CONST = 0.5

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, 'figures')

paths = glob.glob(os.path.join(BASE, 'data', 'modis_real', 'N_Pacific_collocated.parquet'))
df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
vae, scalers, device = load_model(weights_name=WEIGHTS_NAME)
print(f'{len(df):,} pixels loaded')

geom_min = scalers['X_min'][3:].astype('float32')
geom_max = scalers['X_max'][3:].astype('float32')
X_min2 = torch.tensor(scalers['X_min'][:2].astype('float32'), device=device)
X_max2 = torch.tensor(scalers['X_max'][:2].astype('float32'), device=device)
geom_s = (df[['solz', 'satz', 'raz']].values.astype('float32') - geom_min) / (geom_max - geom_min)
refl = df[['refl_213', 'refl_086']].values.astype('float32')
proximity = df['track'].astype(float).values
age_norm = np.where(df['track'].values, AGE_CONST, 0.0)
c_track = np.column_stack([proximity, age_norm]).astype('float32')

re_out = np.empty(len(df), dtype='float64')
bs = 65536
with torch.no_grad():
    for i in range(0, len(df), bs):
        sl = slice(i, i + bs)
        mu_x, _, _, _ = vae.encoder(torch.tensor(refl[sl], device=device),
                                    torch.tensor(geom_s[sl], device=device),
                                    torch.tensor(c_track[sl], device=device))
        zx = torch.sigmoid(mu_x)
        phys = zx * (X_max2 - X_min2) + X_min2
        re_out[sl] = phys[:, 0].cpu().numpy()
out = df.copy()
out['re_hat_raw'] = re_out
cal = apply_granule_calibration(out)

sub = cal[cal['track'] & cal['bg_valid'] & cal['re_hat_cal'].notna()].copy()
sub['delta_obs'] = sub['bg_re_mean'] - sub['re_mod06']
sub['delta_hat'] = sub['re_hat_cal'] - sub['re_mod06']
sub = sub[np.isfinite(sub['delta_obs']) & np.isfinite(sub['delta_hat'])]

corr_pixel = np.corrcoef(sub['delta_obs'], sub['delta_hat'])[0, 1]

sub['cell'] = (sub['granule_key'] + '_' + (sub['lat'] * 4).round().astype(int).astype(str)
               + '_' + (sub['lon'] * 4).round().astype(int).astype(str))
agg = sub.groupby('cell').agg(do=('delta_obs', 'mean'), dh=('delta_hat', 'mean'),
                              gm=('granule_key', 'first'), n=('delta_obs', 'size'))
agg = agg[agg['n'] >= 30]
do_c = agg['do'] - agg.groupby('gm')['do'].transform('mean')
dh_c = agg['dh'] - agg.groupby('gm')['dh'].transform('mean')
rho_seg = np.corrcoef(do_c, dh_c)[0, 1]
print(f'per-pixel corr = {corr_pixel:.3f}, segment within-granule rho = {rho_seg:.3f}')

apply_style()
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
rng = np.random.default_rng(0)

ax = axes[0]
idx = rng.choice(len(sub), size=min(8000, len(sub)), replace=False)
ax.scatter(sub['delta_obs'].values[idx], sub['delta_hat'].values[idx], **SCATTER_KW)
lim = (-8, 12)
ax.plot(lim, lim, **ONE_TO_ONE_KW)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel(r'Observed correction $r_e^{\mathrm{bg}} - r_e^{\mathrm{MOD06}}$ (µm)')
ax.set_ylabel(r'Retrieved correction $\hat{r}_e^{\mathrm{clean}} - r_e^{\mathrm{MOD06}}$ (µm)')
panel_label(ax, 0)
stats_box(ax, f'per-pixel corr = {corr_pixel:.2f}\n(inflated by shared retrieval noise)')

ax = axes[1]
idx2 = rng.choice(len(agg), size=min(8000, len(agg)), replace=False) if len(agg) > 8000 else np.arange(len(agg))
ax.scatter(do_c.values[idx2], dh_c.values[idx2], **SCATTER_KW)
lim2 = (-6, 6)
ax.plot(lim2, lim2, **ONE_TO_ONE_KW)
ax.set_xlim(lim2); ax.set_ylim(lim2)
ax.set_xlabel('Observed correction (µm)\nwithin-granule demeaned')
ax.set_ylabel('Retrieved correction (µm)\nwithin-granule demeaned')
panel_label(ax, 1)
stats_box(ax, f'segment $\\rho$ = {rho_seg:.3f}')

fig.subplots_adjust(wspace=0.35, bottom=0.18)
out_path = os.path.join(FIG_DIR, 'segment_skill_scatter.png')
fig.savefig(out_path, **SAVE_KW)
print(f'Saved: {out_path}')
