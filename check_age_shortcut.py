"""
Fraction of perturbation-ratio variance explained by plume age, for the
deterministic (global-constant plume parameters) and stochastic (v3,
per-pair plume parameters) benchmarks.

Usage: python check_age_shortcut.py
"""
import os
import numpy as np
import xarray as xr

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data')


def age_explained_fraction(nc_path, age_col='age'):
    import pandas as pd
    ds = xr.open_dataset(nc_path)
    df = ds.to_dataframe().reset_index().dropna(subset=['refl_2.1um', 'refl_0.86um'])
    pol = df[df['type'] == 'polluted']
    cln = df[df['type'] == 'clean']
    p = pd.merge(pol, cln, on='pair_id', suffixes=('_pol', '_cln'))
    ratio = (p['re_true_pol'] / p['re_true_cln']).values
    age = p[f'{age_col}_pol'].values
    bins = np.digitize(age, np.linspace(0, 1, 21))
    within = np.mean([ratio[bins == b].std() for b in range(1, 21) if (bins == b).any()])
    return 1 - within ** 2 / ratio.var()


for name, path in [('deterministic (cloud_train_exp.nc)', 'cloud_train_exp.nc'),
                    ('stochastic v3 (cloud_training_data_v3.nc)', 'cloud_training_data_v3.nc')]:
    frac = age_explained_fraction(os.path.join(DATA_DIR, path))
    print(f'{name}: age-explained fraction of perturbation-ratio variance = {frac:.3f}')
