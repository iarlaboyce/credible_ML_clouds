"""
Build (track pixel, nearby clean background pixel) pairs from the collocated
real MODIS data, for real-data pair-consistency fine-tuning.

This is the observationally-realizable analogue of the synthetic PairDataset:
- polluted member: a track pixel's own reflectances/geometry,
  conditioning (proximity=1, age=AGE_CONST). AGE_CONST is used because the
  geometric along-track age proxy carries no usable age signal and
  actively harms inference.
- clean member: the NEAREST valid background (non-track) pixel within
  MAX_PAIR_KM, its own reflectances/geometry, conditioning (0, AGE_CONST).
- physics target (applied to the clean member only, exactly as in synthetic
  training): the background pixel's own MOD06 (re, tau), normalised by the
  surrogate scalers. Pairs whose target falls outside the scaler (training
  prior) range are dropped -- the fraction retained is printed; this biases
  the fine-tuning sample toward the synthetic prior's re/tau range and is a
  known limitation.

Granule-level 70/30 train/test split (seed 42): test granules are NEVER to
be used in fine-tuning; they are the held-out set for the delta-skill gate.

Usage: python build_real_pairs.py [region_glob]
  region_glob default 'N_Pacific_collocated.parquet' -- the data dir also
  holds other regions' collocations (e.g. SE_Pacific, for the companion
  selection-bias paper); this paper's scope is North Pacific only.
Output: data/modis_real/real_pairs.parquet  (+ split in results/realft_granule_split.json)
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data', 'modis_real')
RES_DIR = os.path.join(BASE, 'results')

AGE_CONST = 0.5
MAX_PAIR_KM = 20.0
MAX_PAIRS_PER_GRANULE = 8000
SEED = 42
TEST_FRAC = 0.30
REGION_GLOB = sys.argv[1] if len(sys.argv) > 1 else 'N_Pacific_collocated.parquet'

paths = glob.glob(os.path.join(DATA_DIR, REGION_GLOB))
df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
scalers = np.load(os.path.join(BASE, 'data', 'surrogate_scalers_retrained.npz'))
p_min, p_max = scalers['X_min'][:2], scalers['X_max'][:2]
print(f'{len(df):,} pixels loaded; physics-target prior range: '
      f're [{p_min[0]:.1f},{p_max[0]:.1f}] tau [{p_min[1]:.1f},{p_max[1]:.1f}]')

rng = np.random.default_rng(SEED)
granules = sorted(df['granule_key'].unique())
shuffled = list(granules)
rng.shuffle(shuffled)
n_test = int(round(TEST_FRAC * len(shuffled)))
test_granules = sorted(shuffled[:n_test])
train_granules = sorted(shuffled[n_test:])
os.makedirs(RES_DIR, exist_ok=True)
with open(os.path.join(RES_DIR, 'realft_granule_split.json'), 'w') as f:
    json.dump({'seed': SEED, 'train': train_granules, 'test': test_granules}, f, indent=2)
print(f'granule split: {len(train_granules)} train / {len(test_granules)} test')

frames = []
n_dropped_range = 0
n_dropped_dist = 0
for gkey, g in df.groupby('granule_key'):
    trk = g[g['track']]
    bg = g[~g['track']]
    if len(trk) == 0 or len(bg) < 100:
        continue
    # local tangent-plane km coordinates for honest distances
    lat0 = g['lat'].mean()
    kx = 111.0 * np.cos(np.radians(lat0))
    tree = cKDTree(np.column_stack([bg['lon'].values * kx, bg['lat'].values * 111.0]))
    d_km, idx = tree.query(np.column_stack([trk['lon'].values * kx, trk['lat'].values * 111.0]),
                           workers=-1)
    ok = d_km <= MAX_PAIR_KM
    n_dropped_dist += int((~ok).sum())
    trk = trk[ok]
    twin = bg.iloc[idx[ok]]
    dist = d_km[ok]

    # physics target must lie inside the training prior range
    in_range = ((twin['re_mod06'].values >= p_min[0]) & (twin['re_mod06'].values <= p_max[0]) &
                (twin['tau_mod06'].values >= p_min[1]) & (twin['tau_mod06'].values <= p_max[1]))
    n_dropped_range += int((~in_range).sum())
    trk, twin, dist = trk[in_range], twin.iloc[in_range], dist[in_range]
    if len(trk) == 0:
        continue
    if len(trk) > MAX_PAIRS_PER_GRANULE:
        take = rng.choice(len(trk), size=MAX_PAIRS_PER_GRANULE, replace=False)
        trk, twin, dist = trk.iloc[take], twin.iloc[take], dist[take]

    frames.append(pd.DataFrame({
        'granule_key': gkey,
        'split': 'test' if gkey in test_granules else 'train',
        'lat': trk['lat'].values, 'lon': trk['lon'].values,
        'refl_213_pol': trk['refl_213'].values, 'refl_086_pol': trk['refl_086'].values,
        'solz_pol': trk['solz'].values, 'satz_pol': trk['satz'].values, 'raz_pol': trk['raz'].values,
        'refl_213_cln': twin['refl_213'].values, 'refl_086_cln': twin['refl_086'].values,
        'solz_cln': twin['solz'].values, 'satz_cln': twin['satz'].values, 'raz_cln': twin['raz'].values,
        're_tgt': twin['re_mod06'].values, 'tau_tgt': twin['tau_mod06'].values,
        'pair_dist_km': dist,
    }))

pairs = pd.concat(frames, ignore_index=True)
out = os.path.join(DATA_DIR, 'real_pairs.parquet')
pairs.to_parquet(out, index=False)
print(f'\n{len(pairs):,} pairs -> {out}')
print(f'  dropped {n_dropped_dist:,} (no background within {MAX_PAIR_KM} km), '
      f'{n_dropped_range:,} (target outside prior range)')
print(f'  train pairs: {(pairs["split"]=="train").sum():,}  test pairs: {(pairs["split"]=="test").sum():,}')
print(f'  median pair distance: {pairs["pair_dist_km"].median():.2f} km')
