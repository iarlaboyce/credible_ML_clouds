"""
Granule-clustered bootstrap for the segment-level within-granule delta-skill
correlation on real MODIS data. Resamples granules rather than segments,
since segments within a granule are not independent.

Usage: python segment_skill_bootstrap.py [weights_name] [n_boot]
"""
import sys, glob, os
import numpy as np
import pandas as pd
import torch
from real_data_inference import load_model, apply_granule_calibration

WEIGHTS_NAME = sys.argv[1] if len(sys.argv) > 1 else 'apivae_weights_v2b.pth'
N_BOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
AGE_CONST = 0.5

paths = glob.glob(os.path.join('data', 'modis_real', 'N_Pacific_collocated.parquet'))
df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
vae, scalers, device = load_model(weights_name=WEIGHTS_NAME)
print(f'Model: {WEIGHTS_NAME}, {len(df):,} pixels loaded')

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
sub['cell'] = (sub['granule_key'] + '_' + (sub['lat'] * 4).round().astype(int).astype(str)
               + '_' + (sub['lon'] * 4).round().astype(int).astype(str))
agg = sub.groupby('cell').agg(do=('delta_obs', 'mean'), dh=('delta_hat', 'mean'),
                              gm=('granule_key', 'first'), n=('delta_obs', 'size'))
agg = agg[agg['n'] >= 30].reset_index(drop=True)
granules = agg['gm'].unique()
print(f'{len(agg):,} segments across {len(granules)} granules')


def within_granule_corr(a):
    do_c = a['do'] - a.groupby('gm')['do'].transform('mean')
    dh_c = a['dh'] - a.groupby('gm')['dh'].transform('mean')
    if do_c.std() == 0 or dh_c.std() == 0:
        return np.nan
    return np.corrcoef(do_c, dh_c)[0, 1]


rho_hat = within_granule_corr(agg)
print(f'Point estimate: segment within-granule corr = {rho_hat:+.4f}')

rng = np.random.default_rng(0)
boot = np.empty(N_BOOT)
by_granule = {g: idx.values for g, idx in agg.groupby('gm').groups.items()}
for b in range(N_BOOT):
    sampled = rng.choice(granules, size=len(granules), replace=True)
    idx = np.concatenate([by_granule[g] for g in sampled])
    # re-key granule labels so resampled duplicates of the same granule don't
    # get merged into one group in the within-granule demeaning step
    parts = []
    for k, g in enumerate(sampled):
        d = agg.loc[by_granule[g]].copy()
        d['gm'] = f'{g}__{k}'
        parts.append(d)
    resampled = pd.concat(parts, ignore_index=True)
    boot[b] = within_granule_corr(resampled)

boot = boot[np.isfinite(boot)]
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
p_le_zero = float((boot <= 0).mean())
print(f'Granule-clustered bootstrap (n_boot={len(boot)}):')
print(f'  95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]')
print(f'  P(rho <= 0 | resample distribution) = {p_le_zero:.4f}  '
      f'(one-sided p-value for H0: rho <= 0, i.e. no skill)')

out_path = 'results/segment_skill_bootstrap.json'
import json
with open(out_path, 'w') as f:
    json.dump({'weights': WEIGHTS_NAME, 'n_segments': int(len(agg)),
               'n_granules': int(len(granules)), 'n_boot': int(len(boot)),
               'rho_hat': float(rho_hat), 'ci_95': [float(ci_lo), float(ci_hi)],
               'p_le_zero': p_le_zero}, f, indent=2)
print(f'Saved: {out_path}')
