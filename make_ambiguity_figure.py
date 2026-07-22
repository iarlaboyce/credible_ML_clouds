"""
Identifiability experiment figure: (a) recovery R^2 across the four
optimisation-constraint variants, (b) per-seed R^2 for the clean-only
(natural) objective versus the pair-consistency objective from the
multi-seed robustness check.

Reads results/canonical_results_{variant}.json, produced by
canonical_eval_v2.py / train_apivae_v2b.py.

Usage: python make_ambiguity_figure.py
Output: figures/ambiguity_multiseed.png
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_style import apply_style, panel_label, SAVE_KW

BASE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(BASE, 'results')
FIG_DIR = os.path.join(BASE, 'figures')


def r2_of(variant):
    with open(os.path.join(RES_DIR, f'canonical_results_{variant}.json')) as f:
        return json.load(f)['re_recovery']['apivae']['r2']


apply_style()
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# panel (a): the four canonical variants
ax = axes[0]
labels = ['Clean-only\n(default init)', 'Clean-only\n(informed init)',
          'Pair-\nconsistency', 'Oracle\nlabels']
values = [r2_of('v2c'), r2_of('v2d'), r2_of('v2b_fixedbenchmark'), r2_of('v2e')]
colors = ['#c0392b', '#e67e22', '#2980b9', '#27ae60']
bars = ax.bar(labels, values, color=colors, width=0.6)
ax.axhline(0, color='k', lw=1)
ax.set_ylabel(r'$R^2(r_e)$')
ax.set_ylim(-0.35, 1.05)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + (0.03 if v >= 0 else -0.03),
            f'{v:+.2f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=13)
panel_label(ax, 0)

# panel (b): per-seed spread, clean-only vs pair-consistency
ax = axes[1]
clean_r2 = [r2_of(f'v2c_seed{s}') for s in range(5)]
pair_r2 = [r2_of(f'v2b_seed{s}') for s in range(3)]
rng = np.random.default_rng(0)
x_clean = 0 + rng.uniform(-0.08, 0.08, len(clean_r2))
x_pair = 1 + rng.uniform(-0.08, 0.08, len(pair_r2))
ax.scatter(x_clean, clean_r2, s=90, color='#c0392b', zorder=3, label='clean-only (natural objective)')
ax.scatter(x_pair, pair_r2, s=90, color='#2980b9', zorder=3, label='pair-consistency')
ax.axhline(0, color='k', lw=1)
ax.set_xticks([0, 1])
ax.set_xticklabels([f'clean-only\n(n={len(clean_r2)} seeds)',
                     f'pair-consistency\n(n={len(pair_r2)} seeds)'])
ax.set_xlim(-0.5, 1.5)
ax.set_ylabel(r'$R^2(r_e)$')
panel_label(ax, 1)

fig.tight_layout()
out = os.path.join(FIG_DIR, 'ambiguity_multiseed.png')
fig.savefig(out, **SAVE_KW)
print(f'Saved: {out}')
