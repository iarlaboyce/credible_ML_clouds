"""
Held-out delta-skill gate for the real-data fine-tuning experiment.

Evaluates counterfactual delta skill on the TEST-split granules only
(results/realft_granule_split.json -- never seen during fine-tuning),
comparing the canonical synthetic-only weights against the fine-tuned
weights. The decision metric is the SEGMENT-level (~25km cell)
within-granule correlation between the model's correction
(re_hat_cal - re_mod06) and the observed correction (bg_re_mean -
re_mod06); per-pixel numbers are reported but are inflated by the shared
-re_mod06 noise artifact (see delta_skill_eval.py docstring).

Age conditioning is fixed at AGE_CONST=0.5, matching fine-tuning.

Usage: python finetune_eval.py [region_glob]
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
import torch
from real_data_inference import load_model, apply_granule_calibration

BASE = os.path.dirname(os.path.abspath(__file__))
AGE_CONST = 0.5
REGION_GLOB = sys.argv[1] if len(sys.argv) > 1 else 'N_Pacific_collocated.parquet'

with open(os.path.join(BASE, 'results', 'realft_granule_split.json')) as f:
    split = json.load(f)
test_granules = set(split['test'])

paths = glob.glob(os.path.join(BASE, 'data', 'modis_real', REGION_GLOB))
df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
df = df[df['granule_key'].isin(test_granules)].reset_index(drop=True)
print(f'{len(df):,} pixels from {df["granule_key"].nunique()} HELD-OUT granules')


def infer_and_calibrate(weights_name):
    vae, scalers, device = load_model(weights_name=weights_name)
    g_min = scalers['X_min'][3:].astype('float32')
    g_max = scalers['X_max'][3:].astype('float32')
    X_min2 = torch.tensor(scalers['X_min'][:2].astype('float32'), device=device)
    X_max2 = torch.tensor(scalers['X_max'][:2].astype('float32'), device=device)
    geom_s = (df[['solz', 'satz', 'raz']].values.astype('float32') - g_min) / (g_max - g_min)
    refl = df[['refl_213', 'refl_086']].values.astype('float32')
    age_norm = np.where(df['track'].values, AGE_CONST, 0.0)
    c_track = np.column_stack([df['track'].astype(float).values, age_norm]).astype('float32')
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
    return apply_granule_calibration(out)


def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def evaluate(cal, label):
    sub = cal[cal['track'] & cal['bg_valid'] & cal['re_hat_cal'].notna()].copy()
    sub['delta_obs'] = sub['bg_re_mean'] - sub['re_mod06']
    sub['delta_hat'] = sub['re_hat_cal'] - sub['re_mod06']
    sub = sub[np.isfinite(sub['delta_obs']) & np.isfinite(sub['delta_hat'])]
    do, dh = sub['delta_obs'].values, sub['delta_hat'].values

    print(f'\n===== {label} (n={len(sub):,}, held-out granules only) =====')
    print(f'delta_obs mean={do.mean():+.3f} | delta_hat mean={dh.mean():+.3f} '
          f'(magnitude ratio {dh.mean()/do.mean():.2f})')
    print(f'PER-PIXEL (noise-inflated): corr={np.corrcoef(do, dh)[0, 1]:+.4f}')

    sub['cell'] = (sub['granule_key'] + '_' + (sub['lat'] * 4).round().astype(int).astype(str)
                   + '_' + (sub['lon'] * 4).round().astype(int).astype(str))
    agg = sub.groupby('cell').agg(do=('delta_obs', 'mean'), dh=('delta_hat', 'mean'),
                                  gm=('granule_key', 'first'), n=('delta_obs', 'size'))
    agg = agg[agg['n'] >= 30]
    do_c = agg['do'] - agg.groupby('gm')['do'].transform('mean')
    dh_c = agg['dh'] - agg.groupby('gm')['dh'].transform('mean')
    seg_within = np.corrcoef(do_c, dh_c)[0, 1]
    print(f'SEGMENT (~25km, n={len(agg):,}): pooled corr={np.corrcoef(agg["do"], agg["dh"])[0, 1]:+.4f}  '
          f'R2={r2(agg["do"].values, agg["dh"].values):+.4f}')
    print(f'>>> SEGMENT WITHIN-GRANULE corr (THE GATE): {seg_within:+.4f}')
    gseg = agg.groupby('gm')['do'].transform('mean').values
    print(f'    reference: granule-mean-delta baseline pooled corr='
          f'{np.corrcoef(agg["do"], gseg)[0, 1]:+.4f}')
    return seg_within


base = evaluate(infer_and_calibrate('apivae_weights_v2b.pth'), 'BASELINE: canonical v3-synthetic weights')
ft = evaluate(infer_and_calibrate('apivae_weights_v2b_realft.pth'), 'FINE-TUNED on real pairs')
print(f'\n================ GATE ================')
print(f'segment within-granule delta corr: baseline {base:+.4f} -> fine-tuned {ft:+.4f}')
