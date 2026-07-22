"""
Shared matplotlib style (fonts, panel labelling, stats-box, save params)
for consistent figures across scripts.
"""
import matplotlib.pyplot as plt

PAPER_RCPARAMS = {
    'font.size':         17,
    'axes.labelsize':    19,
    'xtick.labelsize':   15,
    'ytick.labelsize':   15,
    'legend.fontsize':   15,
    'axes.linewidth':    1.3,
    'axes.spines.top':   False,
    'axes.spines.right': False,
}

SCATTER_KW = dict(s=8, alpha=0.45, rasterized=True)
ONE_TO_ONE_KW = dict(color='k', linestyle='--', lw=1.8)
SAVE_KW = dict(dpi=200, bbox_inches='tight')


def apply_style():
    plt.rcParams.update(PAPER_RCPARAMS)


PANEL_LETTERS = 'abcdefghijklmnopqrstuvwxyz'


def panel_label(ax, index, fontsize=18):
    """Bold (a)/(b)/(c)... panel tag, upper-left. `index` is 0-based.
    Descriptive content (what the panel shows) belongs in the caption,
    not on the panel itself."""
    ax.text(0.05, 0.96, f'({PANEL_LETTERS[index]})', transform=ax.transAxes,
            fontsize=fontsize, va='top', fontweight='bold')


def stats_box(ax, text, fontsize=15):
    """White rounded stats annotation, upper-left below the panel label."""
    ax.text(0.05, 0.87, text, transform=ax.transAxes, fontsize=fontsize,
            va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 edgecolor='#cccccc', alpha=0.85))
