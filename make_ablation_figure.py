"""
Bar chart of the component-ablation recovery table (train_ablation_v3.py
results): R^2 on the held-out stochastic-plume benchmark for the full
model versus each ablated variant.

Usage: python make_ablation_figure.py
Output: figures/ablation_bars.png
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_style import apply_style, SAVE_KW

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

VARIANTS = ['Full\n(canonical)', 'No physics\nsupervision', 'No plume\nmodule',
            'Unconstrained\ndecoder']
R2 = [0.66, -0.31, -0.45, 0.63]
COLORS = ['#2980b9', '#c0392b', '#c0392b', '#2980b9']

apply_style()
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(VARIANTS, R2, color=COLORS, width=0.6)
ax.axhline(0, color='k', lw=1)
ax.set_ylabel(r'$R^2(r_e)$')
for b, v in zip(bars, R2):
    ax.text(b.get_x() + b.get_width() / 2, v + (0.03 if v >= 0 else -0.06),
            f'{v:+.2f}', ha='center',
            va='bottom' if v >= 0 else 'top', fontsize=14)
ax.set_ylim(-0.6, 0.85)

out = os.path.join(FIG_DIR, 'ablation_bars.png')
fig.savefig(out, **SAVE_KW)
print(f'Saved: {out}')
