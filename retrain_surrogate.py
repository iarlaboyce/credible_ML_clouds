"""Retrain SurrogateRTM and save consistent weights + scalers.

Usage: python retrain_surrogate.py [nc_filename] [scalers_out_filename]
  nc_filename:           default 'cloud_training_data_exp.nc' (in data/)
  scalers_out_filename:  default 'surrogate_scalers_final.npz' (in data/)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import numpy as np
import xarray as xr
from sklearn.metrics import r2_score, mean_absolute_error

from neural_surrogate import SurrogateRTM

BASE       = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE, 'data')
MODELS_DIR = os.path.join(BASE, 'models')
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

NC_FILENAME      = sys.argv[1] if len(sys.argv) > 1 else 'cloud_training_data_exp.nc'
SCALERS_FILENAME = sys.argv[2] if len(sys.argv) > 2 else 'surrogate_scalers_final.npz'

# load data
ds  = xr.open_dataset(os.path.join(DATA_DIR, NC_FILENAME))
df  = ds.to_dataframe().reset_index().dropna(subset=['refl_2.1um', 'refl_0.86um'])
print(f'Dataset ({NC_FILENAME}): {len(df):,} samples')

X_raw = df[['re_true', 'tau_true', 'veff_true', 'solz', 'satz', 'raz']].values.astype(np.float32)
Y_raw = df[['refl_2.1um', 'refl_0.86um']].values.astype(np.float32)

X_min = X_raw.min(axis=0);  X_max = X_raw.max(axis=0)
Y_min = Y_raw.min(axis=0);  Y_max = Y_raw.max(axis=0)

X_norm = (X_raw - X_min) / (X_max - X_min)
Y_norm = (Y_raw - Y_min) / (Y_max - Y_min)

X_t = torch.tensor(X_norm, dtype=torch.float32)
Y_t = torch.tensor(Y_norm, dtype=torch.float32)

dataset    = TensorDataset(X_t, Y_t)
n          = len(dataset)
train_size = int(0.8 * n)
val_size   = int(0.1 * n)
test_size  = n - train_size - val_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size],
                                          generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=512)
test_loader  = DataLoader(test_ds,  batch_size=512)

# train
model     = SurrogateRTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
criterion = nn.MSELoss()

best_val  = float('inf')
best_state = None

for epoch in range(150):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        criterion(model(xb), yb).backward()
        optimizer.step()
    scheduler.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_ds)
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f'Epoch {epoch+1:3d}  val MSE: {val_loss:.6f}')

# evaluate on test set
model.load_state_dict(best_state)
model.eval()

all_pred, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        all_pred.append(model(xb.to(device)).cpu().numpy())
        all_true.append(yb.numpy())

P_norm = np.concatenate(all_pred)
T_norm = np.concatenate(all_true)
P_phys = P_norm * (Y_max - Y_min) + Y_min
T_phys = T_norm * (Y_max - Y_min) + Y_min

print(f'\n{"Band":<20} {"R² (norm)":>10} {"R² (phys)":>10} {"RMSE (phys)":>12} {"Rel RMSE":>10}')
print('-' * 66)
for i, band in enumerate(['R_2.1µm', 'R_0.86µm']):
    r2n   = r2_score(T_norm[:, i], P_norm[:, i])
    r2p   = r2_score(T_phys[:, i], P_phys[:, i])
    rmse  = np.sqrt(np.mean((T_phys[:, i] - P_phys[:, i])**2))
    rel   = rmse / T_phys[:, i].mean() * 100
    print(f'{band:<20} {r2n:>10.5f} {r2p:>10.5f} {rmse:>12.5f} {rel:>9.2f}%')

# save
weights_out = os.path.join(MODELS_DIR, 'surrogate_rtm_weights.pth')
scalers_out = os.path.join(DATA_DIR,   SCALERS_FILENAME)

torch.save(best_state, weights_out)
np.savez(scalers_out, X_min=X_min, X_max=X_max, Y_min=Y_min, Y_max=Y_max)
print(f'\nSaved weights: {weights_out}')
print(f'Saved scalers: {scalers_out}')
