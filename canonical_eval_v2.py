"""
Canonical evaluation for the v2 API-VAE system: recovery, susceptibility,
and aerosol indirect effect metrics against the held-out test set.

System under test:
  model:     src/model_exp.py  API_VAE
  weights:   models/apivae_weights_v2.pth
  surrogate: models/surrogate_rtm_weights.pth  (R2=0.9947/0.9924 with)
  scalers:   data/surrogate_scalers_retrained.npz
  train:     data/cloud_train_exp.nc
  test:      data/cloud_test_exp.nc

Outputs:
  results/canonical_results_v2.json
  figures/pres_four_estimator_v2.png
  figures/pres_aie_scatter_v2.png
"""
import sys, os, json, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

from model_exp import API_VAE
from neural_surrogate import SurrogateRTM

BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE, 'data')
MODELS_DIR = os.path.join(BASE, 'models')
FIG_DIR    = os.path.join(BASE, 'figures')
RES_DIR    = os.path.join(BASE, 'results')

VARIANT   = sys.argv[1] if len(sys.argv) > 1 else 'v2'
# Optional CLI overrides for the train/test .nc (defaults = old exp benchmark).
# For the v3 evaluation: canonical_eval_v2.py v2b cloud_training_data_v3.nc cloud_test_v3.nc
TRAIN_NAME = sys.argv[2] if len(sys.argv) > 2 else 'cloud_train_exp.nc'
TEST_NAME  = sys.argv[3] if len(sys.argv) > 3 else 'cloud_test_exp.nc'
WEIGHTS   = os.path.join(MODELS_DIR, f'apivae_weights_{VARIANT}.pth')
SURROGATE = os.path.join(MODELS_DIR, 'surrogate_rtm_weights.pth')
SCALERS   = os.path.join(DATA_DIR, 'surrogate_scalers_retrained.npz')
TRAIN_NC  = os.path.join(DATA_DIR, TRAIN_NAME)
TEST_NC   = os.path.join(DATA_DIR, TEST_NAME)

N_BOOT = 5000
SEED   = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rng    = np.random.default_rng(SEED)

def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

# load models
scalers  = np.load(SCALERS)
geom_min = scalers['X_min'][3:].astype('float32')
geom_max = scalers['X_max'][3:].astype('float32')
X_min_t  = torch.tensor(scalers['X_min'][:2].astype('float32'), device=device)
X_max_t  = torch.tensor(scalers['X_max'][:2].astype('float32'), device=device)
Y_min_t  = torch.tensor(scalers['Y_min'].astype('float32'), device=device)
Y_max_t  = torch.tensor(scalers['Y_max'].astype('float32'), device=device)

surrogate = SurrogateRTM().to(device)
surrogate.load_state_dict(torch.load(SURROGATE, map_location=device, weights_only=True))
surrogate.eval()

vae = API_VAE(surrogate, scalers, device).to(device)
vae.load_state_dict(torch.load(WEIGHTS, map_location=device, weights_only=True))
vae.eval()

# data
def load_pairs(nc_path):
    ds  = xr.open_dataset(nc_path)
    df  = ds.to_dataframe().reset_index().dropna(subset=['refl_2.1um', 'refl_0.86um'])
    pol = df[df['type'] == 'polluted']
    cln = df[df['type'] == 'clean']
    return pd.merge(pol, cln, on='pair_id', suffixes=('_pol', '_cln')).reset_index(drop=True)

train_pairs = load_pairs(TRAIN_NC)
test_pairs  = load_pairs(TEST_NC)
n = len(test_pairs)
print(f'Train pairs: {len(train_pairs):,}  Test pairs: {n:,}')

re_true  = test_pairs['re_true_cln'].values.astype(np.float64)
tau_true = test_pairs['tau_true_cln'].values.astype(np.float64)
re_pol   = test_pairs['re_true_pol'].values.astype(np.float64)
tau_pol  = test_pairs['tau_true_pol'].values.astype(np.float64)

# estimator 1: naive
re_naive = re_pol.copy()

# estimator 2: mean correction (offset from TRAINING set)
S_mean_train = ((train_pairs['re_true_cln'] - train_pairs['re_true_pol'])
                / train_pairs['re_true_cln']).mean()
re_mean  = re_pol / (1.0 - S_mean_train)
tau_mean = tau_pol * (re_pol / re_mean)   # LWP-conserving

# estimator 3: MLP baseline (same inputs & labels as VAE)
def make_features(pairs, age_max):
    refl   = pairs[['refl_2.1um_pol', 'refl_0.86um_pol']].values.astype(np.float32)
    geom   = pairs[['solz_pol', 'satz_pol', 'raz_pol']].values.astype(np.float32)
    geom_s = ((geom - geom_min) / (geom_max - geom_min)).astype(np.float32)
    age    = (pairs[['age_pol']].values / age_max).astype(np.float32)
    return np.hstack([refl, geom_s, age])

age_max  = float(train_pairs['age_pol'].max())
X_tr     = make_features(train_pairs, age_max)
X_te     = make_features(test_pairs,  age_max)
y_tr     = train_pairs['re_true_cln'].values.astype(np.float32)
mu_f, sd_f = X_tr.mean(0), X_tr.std(0) + 1e-8

class MLP(nn.Module):
    def __init__(self, in_dim=6, hidden=256, depth=4):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(1)

torch.manual_seed(SEED)
mlp    = MLP().to(device)
loader = DataLoader(TensorDataset(torch.tensor((X_tr - mu_f) / sd_f, device=device),
                                  torch.tensor(y_tr, device=device)),
                    batch_size=512, shuffle=True)
opt   = optim.Adam(mlp.parameters(), lr=1e-3)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
print('Training MLP baseline...')
for ep in range(100):
    mlp.train()
    for xb, yb in loader:
        loss = nn.functional.mse_loss(mlp(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
mlp.eval()
with torch.no_grad():
    re_mlp = mlp(torch.tensor((X_te - mu_f) / sd_f, device=device)).cpu().numpy().astype(np.float64)

# estimator 4: API-VAE v2 (posterior mean)
x_refl  = torch.tensor(test_pairs[['refl_2.1um_pol', 'refl_0.86um_pol']].values,
                       dtype=torch.float32, device=device)
geom_raw = test_pairs[['solz_pol', 'satz_pol', 'raz_pol']].values.astype('float32')
# c_geom: scaled to [0,1] for the VAE ENCODER only (its geometry-conditioning
# convention). The surrogate (R086 below) needs the RAW-degree geometry, since
# it applies its own single normalisation via the full 6-dim X_min_s/X_max_s --
# feeding it c_geom would double-normalise and collapse geometry near 0.
c_geom  = torch.tensor((geom_raw - geom_min) / (geom_max - geom_min),
                       dtype=torch.float32, device=device)
c_geom_raw = torch.tensor(geom_raw, dtype=torch.float32, device=device)
c_track = torch.tensor(np.column_stack((np.ones(n), test_pairs['age_pol'].values)),
                       dtype=torch.float32, device=device)
with torch.no_grad():
    mu_x, _, _, _ = vae.encoder(x_refl, c_geom, c_track)
    zx      = torch.sigmoid(mu_x)
    re_vae  = (zx * (X_max_t - X_min_t) + X_min_t)[:, 0].cpu().numpy().astype(np.float64)
    tau_vae = (zx * (X_max_t - X_min_t) + X_min_t)[:, 1].cpu().numpy().astype(np.float64)

# plume parameters (interpretability)
import torch.nn.functional as F
plume = {
    'k_eff':        float(F.softplus(vae.plume_network.k_eff).item()),
    'nd_boost':     float(F.softplus(vae.plume_network.nd_boost).item()),
    'max_dilution': float(torch.sigmoid(vae.plume_network.max_dilution).item()),
}

# metrics: re recovery
estimators = {
    'naive':   re_naive,
    'mean':    re_mean,
    'mlp':     re_mlp,
    'apivae':  re_vae,
}
re_metrics = {k: {'r2': r2_score(re_true, v),
                  'mae': mean_absolute_error(re_true, v)}
              for k, v in estimators.items()}

# metrics: susceptibility
S_true = (re_true - re_pol) / re_true
S_ests = {
    'mean':   (re_mean - re_pol) / re_mean,
    'mlp':    (re_mlp  - re_pol) / re_mlp,
    'apivae': (re_vae  - re_pol) / re_vae,
}
S_metrics = {k: {'r2': r2_score(S_true, v),
                 'mae': mean_absolute_error(S_true, v)}
             for k, v in S_ests.items()}

# AIE: both truth definitions
def R086(re, tau, batch=4096):
    X_min_s = torch.tensor(scalers['X_min'].astype('float32'), device=device)
    X_max_s = torch.tensor(scalers['X_max'].astype('float32'), device=device)
    re_t  = torch.tensor(re,  dtype=torch.float32, device=device).unsqueeze(1)
    tau_t = torch.tensor(tau, dtype=torch.float32, device=device).unsqueeze(1)
    veff  = torch.full_like(re_t, 0.12)
    out = []
    for i in range(0, len(re), batch):
        sl  = slice(i, i + batch)
        inp = torch.cat([re_t[sl], tau_t[sl], veff[sl], c_geom_raw[sl]], dim=1)
        with torch.no_grad():
            r = surrogate((inp - X_min_s) / (X_max_s - X_min_s)) * (Y_max_t - Y_min_t) + Y_min_t
        out.append(r[:, 1].cpu().numpy())
    return np.concatenate(out).astype(np.float64)

R_pol_s  = R086(re_pol,  tau_pol)
R_true_s = R086(re_true, tau_true)
R_vae_s  = R086(re_vae,  tau_vae)
R_mean_s = R086(re_mean, tau_mean)

dR = {
    'surr': {  # consistent-forward-model definition
        'true': R_pol_s - R_true_s,
        'vae':  R_pol_s - R_vae_s,
        'mean': R_pol_s - R_mean_s,
    },
    'label': {  # non-circular: DISORT-labelled truth
        'true': (test_pairs['refl_0.86um_pol'].values
                 - test_pairs['refl_0.86um_cln'].values).astype(np.float64),
        'vae':  test_pairs['refl_0.86um_pol'].values.astype(np.float64) - R_vae_s,
        'mean': test_pairs['refl_0.86um_pol'].values.astype(np.float64) - R_mean_s,
    },
}
aie_metrics = {}
for defn, d in dR.items():
    aie_metrics[defn] = {}
    for est in ['vae', 'mean']:
        aie_metrics[defn][est] = {
            'r2':   r2_score(d['true'], d[est]),
            'corr': float(np.corrcoef(d['true'], d[est])[0, 1]),
            'bias': float((d[est] - d['true']).mean()),
        }

# paired bootstrap on key comparisons
def paired_bootstrap_r2(y, a, b, n_boot=N_BOOT):
    """CIs for R2(a), R2(b) and R2(a)-R2(b) under paired resampling."""
    idx_all = np.arange(len(y))
    r2a, r2b = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(idx_all, size=len(y), replace=True)
        r2a[i] = r2_score(y[idx], a[idx])
        r2b[i] = r2_score(y[idx], b[idx])
    diff = r2a - r2b
    pct = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return {'a_ci': pct(r2a), 'b_ci': pct(r2b), 'diff_ci': pct(diff),
            'p_one_sided': float((diff <= 0).mean())}

print('Bootstrapping (this takes a minute)...')
boot = {
    're_apivae_vs_mlp':  paired_bootstrap_r2(re_true, re_vae, re_mlp),
    're_apivae_vs_mean': paired_bootstrap_r2(re_true, re_vae, re_mean),
    'aie_surr_apivae_vs_mean': paired_bootstrap_r2(
        dR['surr']['true'], dR['surr']['vae'], dR['surr']['mean']),
}

# figures
plt.rcParams.update({
    'font.size': 17, 'axes.labelsize': 19, 'xtick.labelsize': 15,
    'ytick.labelsize': 15, 'axes.linewidth': 1.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})
sub = rng.choice(n, size=min(3000, n), replace=False)

fig, axes = plt.subplots(2, 2, figsize=(11, 10))
axes = axes.flatten()
lims = (float(min(re_true.min(), 8)), float(re_true.max() + 0.5))
names = [('Naive', re_naive), ('Mean correction', re_mean),
         ('MLP', re_mlp), ('API-VAE', re_vae)]
for ax, (name, pred) in zip(axes, names):
    sc = ax.scatter(re_true[sub], pred[sub], c=tau_pol[sub], cmap='viridis',
                    s=8, alpha=0.45, rasterized=True)
    ax.plot(lims, lims, 'k--', lw=1.8)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(r'True $r_e^{\mathrm{clean}}$ (µm)')
    ax.set_ylabel(r'Predicted $r_e^{\mathrm{clean}}$ (µm)')
    m = re_metrics[{'Naive': 'naive', 'Mean correction': 'mean',
                    'MLP': 'mlp', 'API-VAE': 'apivae'}[name]]
    ax.text(0.05, 0.96, name, transform=ax.transAxes, fontsize=18,
            va='top', fontweight='bold')
    ax.text(0.05, 0.87, f"MAE = {m['mae']:.2f} µm\n$R^2$ = {m['r2']:.3f}",
            transform=ax.transAxes, fontsize=15, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#cccccc', alpha=0.85))
fig.subplots_adjust(left=0.09, right=0.88, hspace=0.35, wspace=0.32)
cax = fig.add_axes([0.91, 0.12, 0.025, 0.76])
cb  = fig.colorbar(sc, cax=cax)
cb.set_label('Optical depth τ', fontsize=16)
out1 = os.path.join(FIG_DIR, f'pres_four_estimator_{VARIANT}.png')
plt.savefig(out1, dpi=200, bbox_inches='tight'); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
d = dR['surr']
lim = (float(d['true'].min() - 0.01), float(d['true'].max() + 0.01))
for ax, (name, key) in zip(axes, [('Mean correction', 'mean'), ('API-VAE', 'vae')]):
    sc = ax.scatter(d['true'][sub], d[key][sub], c=S_true[sub], cmap='RdYlBu_r',
                    s=8, alpha=0.45, rasterized=True)
    ax.plot(lim, lim, 'k--', lw=1.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('True aerosol indirect effect ΔR')
    ax.set_ylabel('Estimated ΔR')
    ax.text(0.05, 0.96, name, transform=ax.transAxes, fontsize=18,
            va='top', fontweight='bold')
    ax.text(0.05, 0.87, f"$R^2$ = {aie_metrics['surr'][key]['r2']:.3f}",
            transform=ax.transAxes, fontsize=15, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#cccccc', alpha=0.85))
fig.subplots_adjust(left=0.09, right=0.88, wspace=0.32)
cax = fig.add_axes([0.91, 0.12, 0.025, 0.76])
cb  = fig.colorbar(sc, cax=cax)
cb.set_label('Cloud susceptibility $S$', fontsize=16)
out2 = os.path.join(FIG_DIR, f'pres_aie_scatter_{VARIANT}.png')
plt.savefig(out2, dpi=200, bbox_inches='tight'); plt.close()

# manifest + save
results = {
    'provenance': {
        'weights':   {'path': os.path.relpath(WEIGHTS, BASE),   'md5': md5(WEIGHTS)},
        'surrogate': {'path': os.path.relpath(SURROGATE, BASE), 'md5': md5(SURROGATE)},
        'scalers':   {'path': os.path.relpath(SCALERS, BASE),   'md5': md5(SCALERS)},
        'train_nc':  {'path': os.path.relpath(TRAIN_NC, BASE),  'md5': md5(TRAIN_NC)},
        'test_nc':   {'path': os.path.relpath(TEST_NC, BASE),   'md5': md5(TEST_NC)},
        'n_test_pairs': int(n), 'seed': SEED, 'n_boot': N_BOOT,
        'mean_correction_offset_source': 'train',
        'vae_inference': 'posterior mean (sigmoid(mu_x))',
    },
    'plume_parameters': plume,
    're_recovery': re_metrics,
    'susceptibility': S_metrics,
    'aie': aie_metrics,
    'bootstrap': boot,
}
out_json = os.path.join(RES_DIR, f'canonical_results_{VARIANT}.json')
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps({k: v for k, v in results.items() if k != 'provenance'}, indent=2))
print(f'\nSaved: {out_json}\nSaved: {out1}\nSaved: {out2}')
