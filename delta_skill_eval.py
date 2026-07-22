"""
Delta-skill evaluation: does the model's per-pixel counterfactual CORRECTION
(re_hat_cal - re_mod06) match the observed correction (bg_re_mean - re_mod06)?

This isolates counterfactual skill from shared natural cloud variability,
which dominates the pooled Analysis-1 R2 and is inherited for free by the
do-nothing baseline (whose delta is identically zero -> zero delta skill by
construction).

CAUTION on interpreting per-pixel numbers: both deltas share -re_mod06, so
single-pixel MOD06 retrieval noise (~1.5um) manufactures spurious positive
per-pixel correlation (~0.2-0.3 expected analytically) that has nothing to
do with counterfactual skill. The noise-robust measure is the SEGMENT-level
(~25km cell) within-granule correlation, where shared pixel noise averages
out: real perturbation structure survives aggregation, shared-noise
correlation vanishes. On 16/07/2026 the v3-retrained synthetic-only model
scored per-pixel 0.31-0.37 but segment-level within-granule ~0.04 -- i.e.
the apparent per-pixel skill was almost entirely the shared-noise artifact,
and true per-segment counterfactual skill was ~zero. The one real signal:
the mean correction has the right sign at roughly half the observed
magnitude (+1.10um vs +2.07um observed, with age fixed at 0.5).

Usage: python delta_skill_eval.py [weights_name]
"""
import sys
import glob, os
import numpy as np
import pandas as pd
import torch
from real_data_inference import load_model, apply_granule_calibration, AGE_SATURATION_KM

WEIGHTS_NAME = sys.argv[1] if len(sys.argv) > 1 else 'apivae_weights_v2b.pth'
REGION_GLOB = sys.argv[2] if len(sys.argv) > 2 else '*_collocated.parquet'

paths = glob.glob(os.path.join('data', 'modis_real', REGION_GLOB))
df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
vae, scalers, device = load_model(weights_name=WEIGHTS_NAME)
print(f'Model: {WEIGHTS_NAME}')

geom_min = scalers['X_min'][3:].astype('float32')
geom_max = scalers['X_max'][3:].astype('float32')
X_min2 = torch.tensor(scalers['X_min'][:2].astype('float32'), device=device)
X_max2 = torch.tensor(scalers['X_max'][:2].astype('float32'), device=device)
geom_s = (df[['solz','satz','raz']].values.astype('float32') - geom_min) / (geom_max - geom_min)
refl = df[['refl_213','refl_086']].values.astype('float32')
proximity = df['track'].astype(float).values

def infer(age_mode):
    if age_mode == 'pipeline':
        age_km = df['dist_along_track_km'].fillna(0.0).values
        age_norm = np.clip(age_km / AGE_SATURATION_KM, 0.0, 1.0)
        age_norm = np.where(df['track'].values, age_norm, 0.0)
    else:
        age_norm = np.where(df['track'].values, float(age_mode), 0.0)
    c_track = np.column_stack([proximity, age_norm]).astype('float32')
    re_out = np.empty(len(df), dtype='float64')
    bs = 65536
    with torch.no_grad():
        for i in range(0, len(df), bs):
            sl = slice(i, i+bs)
            mu_x, _, _, _ = vae.encoder(torch.tensor(refl[sl], device=device),
                                        torch.tensor(geom_s[sl], device=device),
                                        torch.tensor(c_track[sl], device=device))
            zx = torch.sigmoid(mu_x)
            phys = zx * (X_max2 - X_min2) + X_min2
            re_out[sl] = phys[:,0].cpu().numpy()
    out = df.copy(); out['re_hat_raw'] = re_out
    return apply_granule_calibration(out)

def r2(y, yhat):
    return 1 - np.sum((y-yhat)**2)/np.sum((y-y.mean())**2)

def evaluate(cal, label):
    sub = cal[cal['track'] & cal['bg_valid'] & cal['re_hat_cal'].notna()].copy()
    sub['delta_obs'] = sub['bg_re_mean'] - sub['re_mod06']   # what pollution actually did (sign: clean minus polluted)
    sub['delta_hat'] = sub['re_hat_cal'] - sub['re_mod06']   # what the model says pollution did
    sub = sub[np.isfinite(sub['delta_obs']) & np.isfinite(sub['delta_hat'])]
    do, dh = sub['delta_obs'].values, sub['delta_hat'].values

    print(f"\n===== {label} (n={len(sub):,}) =====")
    print(f"delta_obs: mean={do.mean():+.3f} std={do.std():.3f} | delta_hat: mean={dh.mean():+.3f} std={dh.std():.3f}")

    # per-pixel skill
    print(f"PER-PIXEL:  corr={np.corrcoef(do,dh)[0,1]:+.4f}  "
          f"R2(model)={r2(do,dh):+.4f}  R2(zero)={r2(do,np.zeros_like(do)):+.4f}")
    # within-granule (removes between-scene shared variance)
    do_c = do - sub.groupby('granule_key')['delta_obs'].transform('mean').values
    dh_c = dh - sub.groupby('granule_key')['delta_hat'].transform('mean').values
    print(f"WITHIN-GRANULE per-pixel: corr={np.corrcoef(do_c,dh_c)[0,1]:+.4f}")
    # sign agreement (vs 50% chance), on pixels with a non-trivial observed effect
    m = np.abs(do) > 0.5
    print(f"SIGN AGREEMENT (|delta_obs|>0.5um, n={m.sum():,}): {np.mean(np.sign(do[m])==np.sign(dh[m])):.4f}")

    # observable baseline: per-granule constant correction = granule mean of delta_obs
    # (computable at inference time from background stats; the strongest trivial rival)
    g_mean = sub.groupby('granule_key')['delta_obs'].transform('mean').values
    print(f"BASELINE granule-mean-delta: corr={np.corrcoef(do,g_mean)[0,1]:+.4f}  R2={r2(do,g_mean):+.4f}")

    # segment aggregation: ~25km spatial cells within granule, min 30 px
    sub['cell'] = sub['granule_key'] + '_' + (sub['lat']*4).round().astype(int).astype(str) \
                  + '_' + (sub['lon']*4).round().astype(int).astype(str)
    agg = sub.groupby('cell').agg(do=('delta_obs','mean'), dh=('delta_hat','mean'),
                                  gm=('granule_key','first'), n=('delta_obs','size'))
    agg = agg[agg['n'] >= 30]
    print(f"SEGMENT (~25km cells, n={len(agg):,}): corr={np.corrcoef(agg['do'],agg['dh'])[0,1]:+.4f}  "
          f"R2(model)={r2(agg['do'].values,agg['dh'].values):+.4f}")
    gseg = agg.groupby('gm')['do'].transform('mean').values
    print(f"  segment baseline granule-mean-delta: corr={np.corrcoef(agg['do'],gseg)[0,1]:+.4f}  R2={r2(agg['do'].values,gseg):+.4f}")
    agg_do_c = agg['do'] - agg.groupby('gm')['do'].transform('mean')
    agg_dh_c = agg['dh'] - agg.groupby('gm')['dh'].transform('mean')
    print(f"  segment WITHIN-GRANULE: corr={np.corrcoef(agg_do_c,agg_dh_c)[0,1]:+.4f}")

evaluate(infer('pipeline'), "AGE = pipeline (noisy geometric proxy)")
evaluate(infer(0.5), "AGE = fixed 0.5")
