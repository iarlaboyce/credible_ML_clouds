"""
Real-data pair-consistency fine-tuning of the API-VAE.

Starts from the canonical (v3-synthetic-trained) weights and fine-tunes on
real MODIS (track pixel, nearby background pixel) pairs built by
build_real_pairs.py, using the SAME loss structure as train_apivae_v2b.py
'pair' mode -- which needs only observationally realizable signals:
  - reconstruction on both members
  - physics L1 on the CLEAN member vs its own MOD06 (re, tau)
  - pair consistency: track-pixel zx chases stop-grad(background zx)

Trains ONLY on 'train'-split granules (the 21 held-out test granules from
results/realft_granule_split.json are reserved for finetune_eval.py's
delta-skill gate). Saves to NEW filenames -- the canonical weights are not
touched.

Usage: python finetune_real_pairs.py
Output: models/apivae_weights_v2b_realft.pth
        results/apivae_v2b_realft_history.csv
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.stdout.reconfigure(line_buffering=True)
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, 'src'))
from model_exp import API_VAE
from neural_surrogate import SurrogateRTM

EPOCHS      = 40
BATCH_PAIRS = 512
LR          = 1e-4          # fine-tune: 10x below scratch-training LR
WARMUP      = 5
LAMBDA_RECON = 100.0
LAMBDA_PHYS  = 5.0
LAMBDA_PAIR  = 5.0
LAMBDA_AUX   = 2.0
BETA_MAX     = 0.0005
GAMMA_MAX    = 0.001
NOISE_LEVEL  = 0.01
EARLY_STOP   = 8
AGE_CONST    = 0.5
MEAN_JACOBIAN = 21.13
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
SUFFIX = f'_seed{SEED}' if len(sys.argv) > 1 else ''

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

PAIRS_PATH   = os.path.join(BASE, 'data', 'modis_real', 'real_pairs.parquet')
SCALERS_PATH = os.path.join(BASE, 'data', 'surrogate_scalers_retrained.npz')
INIT_WEIGHTS = os.path.join(BASE, 'models', 'apivae_weights_v2b.pth')
WEIGHTS_OUT  = os.path.join(BASE, 'models', f'apivae_weights_v2b_realft{SUFFIX}.pth')
HISTORY_OUT  = os.path.join(BASE, 'results', f'apivae_v2b_realft_history{SUFFIX}.csv')

device = torch.device('cuda')
print(f'Fine-tuning {INIT_WEIGHTS} -> {WEIGHTS_OUT}')
print(f'Training on: {torch.cuda.get_device_name(0)}')

scalers = np.load(SCALERS_PATH)
g_min = scalers['X_min'][3:].astype('float32')
g_max = scalers['X_max'][3:].astype('float32')
p_min = scalers['X_min'][:2].astype('float32')
p_max = scalers['X_max'][:2].astype('float32')

p = pd.read_parquet(PAIRS_PATH)
p = p[p['split'] == 'train'].reset_index(drop=True)
print(f'Train-split pairs: {len(p):,} from {p["granule_key"].nunique()} granules')

x_pol = p[['refl_213_pol', 'refl_086_pol']].values.astype('float32')
x_cln = p[['refl_213_cln', 'refl_086_cln']].values.astype('float32')
t_pol = np.column_stack([np.ones(len(p)), np.full(len(p), AGE_CONST)]).astype('float32')
t_cln = np.column_stack([np.zeros(len(p)), np.full(len(p), AGE_CONST)]).astype('float32')
g_pol = ((p[['solz_pol', 'satz_pol', 'raz_pol']].values.astype('float32') - g_min) / (g_max - g_min))
g_cln = ((p[['solz_cln', 'satz_cln', 'raz_cln']].values.astype('float32') - g_min) / (g_max - g_min))
tgt = ((p[['re_tgt', 'tau_tgt']].values.astype('float32') - p_min) / (p_max - p_min))

dataset = TensorDataset(*[torch.tensor(a) for a in
                          [x_pol, x_cln, t_pol, t_cln, g_pol, g_cln, tgt]])
n = len(dataset)
train_n = int(0.9 * n)
train_ds, val_ds = random_split(dataset, [train_n, n - train_n],
                                generator=torch.Generator().manual_seed(SEED))
train_loader = DataLoader(train_ds, batch_size=BATCH_PAIRS, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_PAIRS, shuffle=False)

surrogate = SurrogateRTM().to(device)
surrogate.load_state_dict(torch.load(os.path.join(BASE, 'models', 'surrogate_rtm_weights.pth'),
                                     map_location=device, weights_only=True))
surrogate.eval()
model = API_VAE(surrogate, scalers, device).to(device)
model.load_state_dict(torch.load(INIT_WEIGHTS, map_location=device, weights_only=True))

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

X_min_2d = torch.tensor(p_min, device=device)
X_max_2d = torch.tensor(p_max, device=device)
veff_norm = float((0.10 - scalers['X_min'][2]) / (scalers['X_max'][2] - scalers['X_min'][2]))


def batch_trust(model, x, c_geom, c_track):
    with torch.set_grad_enabled(True):
        mu_x, _, _, _ = model.encoder(x, c_geom, c_track)
        zx = torch.sigmoid(mu_x).detach().requires_grad_(True)
        veff = torch.full_like(zx[:, 0:1], veff_norm)
        p_refl = model.surrogate(torch.cat([zx[:, 0:1], zx[:, 1:2], veff, c_geom], dim=1))
        jac = torch.autograd.grad(p_refl, zx, grad_outputs=torch.ones_like(p_refl))[0]
        j_norm = torch.norm(jac, dim=1, keepdim=True).detach()
    return 1.0 / (1.0 + (j_norm / MEAN_JACOBIAN).pow(2))


best_val = float('inf')
patience = 0
history = []

for epoch in range(EPOCHS):
    model.train()
    anneal = min(1.0, (epoch + 1) / WARMUP)
    beta, gamma = BETA_MAX * anneal, GAMMA_MAX * anneal
    tot = {'recon': 0.0, 'phys': 0.0, 'pair': 0.0}

    for xb_p, xb_c, tb_p, tb_c, gb_p, gb_c, tg in train_loader:
        xb_p, xb_c = xb_p.to(device), xb_c.to(device)
        tb_p, tb_c = tb_p.to(device), tb_c.to(device)
        gb_p, gb_c = gb_p.to(device), gb_c.to(device)
        tg = tg.to(device)

        w_p = batch_trust(model, xb_p, gb_p, tb_p)
        w_c = batch_trust(model, xb_c, gb_c, tb_c)

        optimizer.zero_grad()
        out_p = model(xb_p + torch.randn_like(xb_p) * NOISE_LEVEL, tb_p, gb_p)
        out_c = model(xb_c + torch.randn_like(xb_c) * NOISE_LEVEL, tb_c, gb_c)
        (rec_p, mux_p, lvx_p, muy_p, lvy_p, zx_p, adv_p, aux_p) = out_p
        (rec_c, mux_c, lvx_c, muy_c, lvy_c, zx_c, adv_c, aux_c) = out_c

        recon_loss = (torch.mean(w_p * F.mse_loss(rec_p, xb_p, reduction='none'))
                      + torch.mean(w_c * F.mse_loss(rec_c, xb_c, reduction='none')))
        phys_pred = zx_c * (X_max_2d - X_min_2d) + X_min_2d
        phys_true = tg * (X_max_2d - X_min_2d) + X_min_2d
        phys_loss = F.l1_loss(phys_pred, phys_true)
        pair_loss = F.l1_loss(torch.sigmoid(mux_p), torch.sigmoid(mux_c).detach())
        kl = sum(-0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum()
                 for mu, lv in [(mux_p, lvx_p), (muy_p, lvy_p),
                                (mux_c, lvx_c), (muy_c, lvy_c)]) / (2 * xb_p.size(0))
        adv_loss = (F.mse_loss(adv_p, zx_p.detach()) + F.mse_loss(adv_c, zx_c.detach()))
        aux_loss = (F.mse_loss(aux_p, gb_p) + F.mse_loss(aux_c, gb_c))

        loss = (LAMBDA_RECON * recon_loss + beta * kl + LAMBDA_PHYS * phys_loss
                + LAMBDA_PAIR * pair_loss + LAMBDA_AUX * aux_loss + gamma * adv_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        tot['recon'] += recon_loss.item()
        tot['phys'] += phys_loss.item()
        tot['pair'] += pair_loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb_p, xb_c, tb_p, tb_c, gb_p, gb_c, _ in val_loader:
            xb_p, tb_p, gb_p = xb_p.to(device), tb_p.to(device), gb_p.to(device)
            xb_c, tb_c, gb_c = xb_c.to(device), tb_c.to(device), gb_c.to(device)
            rec_p, *_ = model(xb_p, tb_p, gb_p)
            rec_c, *_ = model(xb_c, tb_c, gb_c)
            val_loss += (F.mse_loss(rec_p, xb_p).item() + F.mse_loss(rec_c, xb_c).item()) * len(xb_p)
    val_loss /= (n - train_n)
    scheduler.step(val_loss)

    nb = len(train_loader)
    k_eff = F.softplus(model.plume_network.k_eff).item()
    history.append({'epoch': epoch + 1, **{k: v / nb for k, v in tot.items()},
                    'val': val_loss, 'k_eff': k_eff})
    print(f"Ep {epoch+1:03d} | recon={tot['recon']/nb:.5f} | phys={tot['phys']/nb:.4f} | "
          f"pair={tot['pair']/nb:.4f} | val={val_loss:.6f} | k={k_eff:.3f}")

    if val_loss < best_val:
        best_val = val_loss
        patience = 0
        torch.save(model.state_dict(), WEIGHTS_OUT)
    else:
        patience += 1
        if patience >= EARLY_STOP:
            print(f'Early stop at epoch {epoch + 1}')
            break

pd.DataFrame(history).to_csv(HISTORY_OUT, index=False)
print(f'Done. Best val: {best_val:.6f} -> {WEIGHTS_OUT}')
