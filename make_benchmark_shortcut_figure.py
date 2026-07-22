"""
Benchmark-shortcut figure: perturbation ratio r_e^pol/r_e^clean versus
plume age, for the deterministic (global-constant plume parameters) and
stochastic (v3, per-pair plume parameters) benchmarks. Also fits a simple
linear regression of ratio ~ age alone for each benchmark, giving a direct
demonstration of how much of the perturbation age alone explains.

Usage: python make_benchmark_shortcut_figure.py
Output: figures/benchmark_shortcut.png
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from plot_style import apply_style, panel_label, stats_box, SCATTER_KW, SAVE_KW

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data')
FIG_DIR = os.path.join(BASE, 'figures')


def load_ratio_age(nc_path, age_col='age'):
    ds = xr.open_dataset(nc_path)
    df = ds.to_dataframe().reset_index().dropna(subset=['refl_2.1um', 'refl_0.86um'])
    pol = df[df['type'] == 'polluted']
    cln = df[df['type'] == 'clean']
    p = pd.merge(pol, cln, on='pair_id', suffixes=('_pol', '_cln'))
    ratio = (p['re_true_pol'] / p['re_true_cln']).values
    age = p[f'{age_col}_pol'].values
    return age, ratio


apply_style()
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for i, (ax, name, path) in enumerate([
        (axes[0], 'Deterministic benchmark', 'cloud_train_exp.nc'),
        (axes[1], 'Stochastic (v3) benchmark', 'cloud_training_data_v3.nc')]):
    age, ratio = load_ratio_age(os.path.join(DATA_DIR, path))
    reg = LinearRegression().fit(age.reshape(-1, 1), ratio)
    r2 = r2_score(ratio, reg.predict(age.reshape(-1, 1)))

    rng = np.random.default_rng(0)
    sub = rng.choice(len(age), size=min(5000, len(age)), replace=False)
    ax.scatter(age[sub], ratio[sub], **SCATTER_KW)
    age_line = np.linspace(0, 1, 100)
    ax.plot(age_line, reg.predict(age_line.reshape(-1, 1)), 'r-', lw=2.5,
            label='linear fit (age only)')
    ax.set_xlabel('Normalised plume age')
    ax.set_ylabel(r'Perturbation ratio $r_e^{\mathrm{pol}}/r_e^{\mathrm{clean}}$')
    panel_label(ax, i)
    stats_box(ax, f'$R^2$(ratio $\\sim$ age) = {r2:.3f}')
    ax.legend(loc='lower left', fontsize=13, frameon=False)
    print(f'{name}: linear R2(ratio ~ age) = {r2:.3f}')

fig.tight_layout()
out = os.path.join(FIG_DIR, 'benchmark_shortcut.png')
fig.savefig(out, **SAVE_KW)
print(f'Saved: {out}')
