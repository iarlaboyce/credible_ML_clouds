"""
Shared inference + calibration for the real-data study.

Applies the trained API-VAE encoder to collocated real MODIS pixels
(output of collocate_real_data.py) to obtain the inferred unperturbed
state (re_hat, tau_hat), then fits and applies a per-granule affine
radiometric calibration (slope + intercept, re_mod06 ~ f(re_hat_raw))
using background pixels only, so no track-pixel information enters the
calibration. An offset-only fit (slope=1) was tried first but per-granule
slopes vary substantially (~0.03-8x across granules), so a fixed slope
leaves accuracy well below the correlation-implied ceiling.

Age proxy: real pixels have no ground-truth plume age. Each track
component's own intrinsic along-track distance (dist_along_track_km, a
PCA axis per connected component oriented via the MOD06 Twomey signature)
is scaled by an assumed advection speed and clipped to the training
range. Background pixels get age = 0 (proximity = 0 already zeroes the
perturbation in the forward model, so age is a don't-care there, but the
encoder still expects a value).
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from model_exp import API_VAE
from neural_surrogate import SurrogateRTM

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, 'models')
DATA_DIR = os.path.join(BASE, 'data')

# Assumed ship advection speed for the age proxy (~18 knots), and the
# distance at which normalised age saturates to 1. Both are approximations
# pending a proper distance-to-age calibration.
ADVECTION_SPEED_KMH = 33.0
AGE_SATURATION_KM = 120.0


def load_model(weights_name='apivae_weights_v2b.pth', scalers_name='surrogate_scalers_retrained.npz',
              surrogate_name='surrogate_rtm_weights.pth', device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scalers = np.load(os.path.join(DATA_DIR, scalers_name))
    surrogate = SurrogateRTM().to(device)
    surrogate.load_state_dict(torch.load(os.path.join(MODELS_DIR, surrogate_name),
                                         map_location=device, weights_only=True))
    surrogate.eval()
    vae = API_VAE(surrogate, scalers, device).to(device)
    vae.load_state_dict(torch.load(os.path.join(MODELS_DIR, weights_name),
                                   map_location=device, weights_only=True))
    vae.eval()
    return vae, scalers, device


def infer_clean_state(df, vae, scalers, device, batch_size=65536):
    """Run the encoder on every pixel; returns re_hat, tau_hat (uncalibrated)."""
    geom_min = scalers['X_min'][3:].astype('float32')
    geom_max = scalers['X_max'][3:].astype('float32')
    X_min2 = torch.tensor(scalers['X_min'][:2].astype('float32'), device=device)
    X_max2 = torch.tensor(scalers['X_max'][:2].astype('float32'), device=device)

    age_km = df['dist_along_track_km'].fillna(0.0).values
    age_norm = np.clip(age_km / AGE_SATURATION_KM, 0.0, 1.0)
    age_norm = np.where(df['track'].values, age_norm, 0.0)
    proximity = df['track'].astype(float).values

    geom = df[['solz', 'satz', 'raz']].values.astype('float32')
    geom_s = (geom - geom_min) / (geom_max - geom_min)
    refl = df[['refl_213', 'refl_086']].values.astype('float32')  # (2.1um, 0.86um) order matches training
    c_track = np.column_stack([proximity, age_norm]).astype('float32')

    re_out, tau_out = np.empty(len(df), dtype='float64'), np.empty(len(df), dtype='float64')
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            sl = slice(i, i + batch_size)
            x_refl = torch.tensor(refl[sl], device=device)
            c_geom = torch.tensor(geom_s[sl], device=device)
            c_trk = torch.tensor(c_track[sl], device=device)
            mu_x, _, _, _ = vae.encoder(x_refl, c_geom, c_trk)
            zx = torch.sigmoid(mu_x)
            phys = zx * (X_max2 - X_min2) + X_min2
            re_out[sl] = phys[:, 0].cpu().numpy()
            tau_out[sl] = phys[:, 1].cpu().numpy()

    df = df.copy()
    df['re_hat_raw'] = re_out
    df['tau_hat_raw'] = tau_out
    return df


def apply_granule_calibration(df, min_bg_pixels=200, min_hat_std=1.5, slope_clip=(0.5, 2.0)):
    """
    Per granule: fit re_mod06 ~ slope * re_hat_raw + intercept by ordinary
    least squares over BACKGROUND, liquid-phase pixels only, then apply
    that same affine map to every pixel in the granule (track included),
    so track pixels never contribute to their own calibration.

    Falls back to an offset-only fit (slope=1) when the background
    re_hat_raw has too little spread (min_hat_std) to estimate a slope
    reliably, and clips the fitted slope to slope_clip otherwise. Both
    guards were added after finding that an unclipped per-granule slope
    fit made pooled Analysis-1 accuracy WORSE than offset-only (R^2 0.247
    vs 0.271): granules where the encoder's raw output is nearly flat on
    background pixels (bg_hat_std ~0.5-1, vs ~2-4 typically) got slopes up
    to ~5x, which amplifies noise in that granule's near-flat signal
    rather than recovering real information, and that noise then gets
    carried onto track pixels. Clipping recovers a small net gain over
    offset-only (R^2 0.273) instead of a regression.
    """
    df = df.copy()
    df['re_hat_cal'] = np.nan
    df['calib_slope'] = np.nan
    df['calib_intercept'] = np.nan
    df['calib_n_bg'] = 0

    for gkey, g in df.groupby('granule_key'):
        bg = g[~g['track']]
        n_bg = len(bg)
        if n_bg < min_bg_pixels:
            continue  # leave uncalibrated (NaN) -- excluded downstream
        if bg['re_hat_raw'].std() < min_hat_std:
            slope, intercept = 1.0, (bg['re_mod06'] - bg['re_hat_raw']).mean()
        else:
            slope, intercept = np.polyfit(bg['re_hat_raw'], bg['re_mod06'], 1)
            slope = np.clip(slope, *slope_clip)
            intercept = (bg['re_mod06'] - slope * bg['re_hat_raw']).mean()
        idx = g.index
        df.loc[idx, 're_hat_cal'] = slope * df.loc[idx, 're_hat_raw'] + intercept
        df.loc[idx, 'calib_slope'] = slope
        df.loc[idx, 'calib_intercept'] = intercept
        df.loc[idx, 'calib_n_bg'] = n_bg
    return df


def run_inference_pipeline(df, weights_name='apivae_weights_v2b.pth'):
    vae, scalers, device = load_model(weights_name=weights_name)
    df = infer_clean_state(df, vae, scalers, device)
    df = apply_granule_calibration(df)
    return df


if __name__ == '__main__':
    import glob
    paths = glob.glob(os.path.join(DATA_DIR, 'modis_real', '*_collocated.parquet'))
    print(f'Found {len(paths)} collocated region file(s): {paths}')
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    print(f'{len(df):,} total pixels')

    df = run_inference_pipeline(df)
    n_calibrated = df['re_hat_cal'].notna().sum()
    print(f'{n_calibrated:,} pixels calibrated ({100*n_calibrated/len(df):.1f}%)')
    print(f"mean calibration slope: {df['calib_slope'].mean():.3f} "
          f"(std across granules: {df.groupby('granule_key')['calib_slope'].first().std():.3f})")
    print(f"mean calibration intercept: {df['calib_intercept'].mean():.3f} um "
          f"(std across granules: {df.groupby('granule_key')['calib_intercept'].first().std():.3f})")

    out_path = os.path.join(DATA_DIR, 'modis_real', 'inferred_combined.parquet')
    df.to_parquet(out_path, index=False)
    print(f'Saved: {out_path}')
