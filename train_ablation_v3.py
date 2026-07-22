"""
Component ablations of the API-VAE architecture, retrained on v3
(stochastic-plume) data with the SAME pair-consistency training objective
as the canonical model (train_apivae_v2b.py), so the comparison is fair.

The paper's ablation claim ("no plume module / no physics supervision /
unconstrained decoder collapse recovery to negative R^2") was previously
demonstrated only with an older, DIFFERENT training objective (the legacy
no_plume_vae_train.py / no_physics_vae_train.py / black_box_vae_train.py
scripts supervise re_true/tau_true directly on BOTH pair members --
including the polluted member's own true state, which is oracle
information the canonical model never sees). That is not a fair ablation
of the current architecture; this script is.

Variants (--variant):
  full                   canonical architecture (model_exp.API_VAE) --
                         reproduces the main result as a sanity check.
  no_physics             same architecture, LAMBDA_PHYS=0 (no clean-member
                         physics supervision).
  no_plume               no_plume_vae.API_VAE_Ablation_B: PlumeNetwork
                         removed, encoder blind to track/age (2-arg,
                         matching that class's own design).
  unconstrained_decoder  no_physics_vae.API_VAE_Ablation_C: BlackBoxDecoder
                         replaces the surrogate+plume physics decoder
                         entirely (no RTM call at all, so no trust-weighting
                         is applied -- there is no Jacobian to weight by).

Usage: python train_ablation_v3.py <variant> [nc_filename] [test_nc_filename]
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

sys.stdout.reconfigure(line_buffering=True)

VARIANT = sys.argv[1] if len(sys.argv) > 1 else 'full'
assert VARIANT in ('full', 'no_physics', 'no_plume', 'unconstrained_decoder')
NC_NAME = sys.argv[2] if len(sys.argv) > 2 else 'cloud_training_data_v3.nc'
TEST_NC_NAME = sys.argv[3] if len(sys.argv) > 3 else 'cloud_test_v3.nc'

EPOCHS = 200
BATCH_PAIRS = 256
LR = 1e-3
WARMUP = 10
MEAN_JACOBIAN = 21.13
LAMBDA_RECON = 100.0
LAMBDA_PHYS = 0.0 if VARIANT == 'no_physics' else 5.0
LAMBDA_PAIR = 5.0
LAMBDA_AUX = 2.0
BETA_MAX = 0.0005
GAMMA_MAX = 0.001
NOISE_LEVEL = 0.01
EARLY_STOP = 20
HAS_TRUST_WEIGHT = VARIANT != 'unconstrained_decoder'  # no surrogate call -> no Jacobian to weight by
ENCODER_TAKES_TRACK = VARIANT != 'no_plume'             # Ablation B encoder is (x, c_geom) only

BASE = os.path.dirname(os.path.abspath(__file__))
NC_PATH = os.path.join(BASE, 'data', NC_NAME)
TEST_NC_PATH = os.path.join(BASE, 'data', TEST_NC_NAME)
SCALERS_PATH = os.path.join(BASE, 'data', 'surrogate_scalers_retrained.npz')
WEIGHTS_OUT = os.path.join(BASE, 'models', f'apivae_ablation_{VARIANT}_v3.pth')
HISTORY_OUT = os.path.join(BASE, 'results', f'apivae_ablation_{VARIANT}_v3_history.csv')
RESULT_OUT = os.path.join(BASE, 'results', f'apivae_ablation_{VARIANT}_v3_result.json')

from neural_surrogate import SurrogateRTM
if VARIANT in ('full', 'no_physics'):
    from model_exp import API_VAE as ModelClass
elif VARIANT == 'no_plume':
    from no_plume_vae import API_VAE_Ablation_B as ModelClass
elif VARIANT == 'unconstrained_decoder':
    from no_physics_vae import API_VAE_Ablation_C as ModelClass


class PairDataset(Dataset):
    def __init__(self, nc_path, scalers_path):
        ds = xr.open_dataset(nc_path)
        df = ds.to_dataframe().reset_index().dropna(subset=["refl_2.1um", "refl_0.86um"])
        pol = df[df["type"] == "polluted"]
        cln = df[df["type"] == "clean"]
        p = pd.merge(pol, cln, on="pair_id", suffixes=("_pol", "_cln")).reset_index(drop=True)

        scalers = np.load(scalers_path)
        g_min = scalers["X_min"][3:].astype("float32")
        g_max = scalers["X_max"][3:].astype("float32")
        p_min = scalers["X_min"][:2].astype("float32")
        p_max = scalers["X_max"][:2].astype("float32")

        self.x_pol = p[["refl_2.1um_pol", "refl_0.86um_pol"]].values.astype("float32")
        self.x_cln = p[["refl_2.1um_cln", "refl_0.86um_cln"]].values.astype("float32")
        self.t_pol = np.column_stack((np.ones(len(p)), p["age_pol"].values)).astype("float32")
        self.t_cln = np.column_stack((np.zeros(len(p)), p["age_cln"].values)).astype("float32")
        self.g_pol = ((p[["solz_pol", "satz_pol", "raz_pol"]].values.astype("float32") - g_min)
                      / (g_max - g_min))
        self.g_cln = ((p[["solz_cln", "satz_cln", "raz_cln"]].values.astype("float32") - g_min)
                      / (g_max - g_min))
        tgt = p[["re_true_cln", "tau_true_cln"]].values.astype("float32")
        self.zx_tgt = (tgt - p_min) / (p_max - p_min)
        print(f"Pairs: {len(p)}", flush=True)

    def __len__(self):
        return len(self.x_pol)

    def __getitem__(self, i):
        return (torch.tensor(self.x_pol[i]), torch.tensor(self.x_cln[i]),
                torch.tensor(self.t_pol[i]), torch.tensor(self.t_cln[i]),
                torch.tensor(self.g_pol[i]), torch.tensor(self.g_cln[i]),
                torch.tensor(self.zx_tgt[i]))


def call_encoder(model, x, c_geom, c_track):
    if ENCODER_TAKES_TRACK:
        return model.encoder(x, c_geom, c_track)
    return model.encoder(x, c_geom)


def batch_trust(model, x, c_geom, c_track, veff_norm):
    if not HAS_TRUST_WEIGHT:
        return torch.ones(x.size(0), 1, device=x.device)
    with torch.set_grad_enabled(True):
        mu_x, _, _, _ = call_encoder(model, x, c_geom, c_track)
        zx = torch.sigmoid(mu_x).detach().requires_grad_(True)
        veff = torch.full_like(zx[:, 0:1], veff_norm)
        p_refl = model.surrogate(torch.cat([zx[:, 0:1], zx[:, 1:2], veff, c_geom], dim=1))
        jac = torch.autograd.grad(p_refl, zx, grad_outputs=torch.ones_like(p_refl))[0]
        j_norm = torch.norm(jac, dim=1, keepdim=True).detach()
    return 1.0 / (1.0 + (j_norm / MEAN_JACOBIAN).pow(2))


if __name__ == "__main__":
    device = torch.device("cuda")
    print(f"Variant: {VARIANT}  ->  {WEIGHTS_OUT}", flush=True)
    print(f"LAMBDA_PHYS={LAMBDA_PHYS}  HAS_TRUST_WEIGHT={HAS_TRUST_WEIGHT}  "
          f"ENCODER_TAKES_TRACK={ENCODER_TAKES_TRACK}", flush=True)

    scalers = np.load(SCALERS_PATH)
    surrogate = SurrogateRTM().to(device)
    surrogate.load_state_dict(torch.load(os.path.join(BASE, 'models', 'surrogate_rtm_weights.pth'),
                                         map_location=device, weights_only=True))
    surrogate.eval()

    if VARIANT == 'unconstrained_decoder':
        model = ModelClass(scalers, device).to(device)
    else:
        model = ModelClass(surrogate, scalers, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)

    dataset = PairDataset(NC_PATH, SCALERS_PATH)
    n = len(dataset)
    train_n = int(0.85 * n)
    train_ds, val_ds = random_split(dataset, [train_n, n - train_n],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=BATCH_PAIRS, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_PAIRS, shuffle=False)

    X_min_2d = torch.tensor(scalers["X_min"][:2].astype("float32"), device=device)
    X_max_2d = torch.tensor(scalers["X_max"][:2].astype("float32"), device=device)
    veff_norm = float((0.10 - scalers["X_min"][2]) / (scalers["X_max"][2] - scalers["X_min"][2]))

    best_val = float("inf")
    patience = 0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        anneal = min(1.0, (epoch + 1) / WARMUP)
        beta, gamma = BETA_MAX * anneal, GAMMA_MAX * anneal
        tot = {"recon": 0.0, "phys": 0.0, "pair": 0.0}

        for xb_p, xb_c, tb_p, tb_c, gb_p, gb_c, tgt in train_loader:
            xb_p, xb_c = xb_p.to(device), xb_c.to(device)
            tb_p, tb_c = tb_p.to(device), tb_c.to(device)
            gb_p, gb_c = gb_p.to(device), gb_c.to(device)
            tgt = tgt.to(device)

            w_p = batch_trust(model, xb_p, gb_p, tb_p, veff_norm)
            w_c = batch_trust(model, xb_c, gb_c, tb_c, veff_norm)

            optimizer.zero_grad()
            out_p = model(xb_p + torch.randn_like(xb_p) * NOISE_LEVEL, tb_p, gb_p)
            out_c = model(xb_c + torch.randn_like(xb_c) * NOISE_LEVEL, tb_c, gb_c)
            (rec_p, mux_p, lvx_p, muy_p, lvy_p, zx_p, adv_p, aux_p) = out_p
            (rec_c, mux_c, lvx_c, muy_c, lvy_c, zx_c, adv_c, aux_c) = out_c

            recon_loss = (torch.mean(w_p * F.mse_loss(rec_p, xb_p, reduction="none"))
                          + torch.mean(w_c * F.mse_loss(rec_c, xb_c, reduction="none")))

            phys_pred = zx_c * (X_max_2d - X_min_2d) + X_min_2d
            phys_true = tgt * (X_max_2d - X_min_2d) + X_min_2d
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

            tot["recon"] += recon_loss.item()
            tot["phys"] += phys_loss.item()
            tot["pair"] += pair_loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb_p, xb_c, tb_p, tb_c, gb_p, gb_c, _ in val_loader:
                xb_p, tb_p, gb_p = xb_p.to(device), tb_p.to(device), gb_p.to(device)
                xb_c, tb_c, gb_c = xb_c.to(device), tb_c.to(device), gb_c.to(device)
                rec_p, *_ = model(xb_p, tb_p, gb_p)
                rec_c, *_ = model(xb_c, tb_c, gb_c)
                val_loss += (F.mse_loss(rec_p, xb_p).item()
                             + F.mse_loss(rec_c, xb_c).item()) * len(xb_p)
        val_loss /= (n - train_n)
        scheduler.step(val_loss)

        nb = len(train_loader)
        history.append({"epoch": epoch + 1, **{k: v / nb for k, v in tot.items()}, "val": val_loss})
        print(f"Ep {epoch+1:03d} | recon={tot['recon']/nb:.5f} | phys={tot['phys']/nb:.4f} | "
              f"pair={tot['pair']/nb:.4f} | val={val_loss:.6f}", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), WEIGHTS_OUT)
        else:
            patience += 1
            if patience >= EARLY_STOP:
                print(f"Early stop at epoch {epoch + 1}", flush=True)
                break

    pd.DataFrame(history).to_csv(HISTORY_OUT, index=False)
    print(f"Done. Best val: {best_val:.6f}  ->  {WEIGHTS_OUT}", flush=True)

    # held-out re-recovery evaluation on the v3 test set
    print(f"\nEvaluating on {TEST_NC_NAME}...", flush=True)
    model.load_state_dict(torch.load(WEIGHTS_OUT, map_location=device, weights_only=True))
    model.eval()
    test_pairs = PairDataset(TEST_NC_PATH, SCALERS_PATH)
    x_pol_t = torch.tensor(test_pairs.x_pol, device=device)
    g_pol_t = torch.tensor(test_pairs.g_pol, device=device)
    t_pol_t = torch.tensor(test_pairs.t_pol, device=device)
    with torch.no_grad():
        mu_x, _, _, _ = call_encoder(model, x_pol_t, g_pol_t, t_pol_t)
        zx = torch.sigmoid(mu_x)
        phys = zx * (X_max_2d - X_min_2d) + X_min_2d
        re_vae = phys[:, 0].cpu().numpy().astype(np.float64)

    re_true = (test_pairs.zx_tgt[:, 0] * (scalers['X_max'][0] - scalers['X_min'][0])
              + scalers['X_min'][0]).astype(np.float64)
    r2 = r2_score(re_true, re_vae)
    mae = mean_absolute_error(re_true, re_vae)
    print(f"re recovery on held-out v3 test set: R2={r2:.4f}  MAE={mae:.4f} um")

    with open(RESULT_OUT, 'w') as f:
        json.dump({'variant': VARIANT, 'n_test': len(test_pairs), 're_r2': float(r2),
                   're_mae': float(mae), 'best_val_loss': float(best_val)}, f, indent=2)
    print(f"Saved: {RESULT_OUT}")
