import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error

# ==========================================
# 1. DATASET WITH BI-DIRECTIONAL SCALING
# ==========================================
class SurrogateDataset(Dataset):
    def __init__(self, nc_file_path):
        print(f"Loading 1D DISORT data from {nc_file_path}...")
        ds = xr.open_dataset(nc_file_path)
        df = ds.to_dataframe().reset_index().dropna(subset=['refl_2.1um', 'refl_0.86um'])
        
        # We KEEP extreme reflectances (up to 1.947) as they are physically valid 
        # in specific scattering geometries[cite: 36].
        self.X_raw = df[['re_true', 'tau_true', 'veff_true', 'solz', 'satz', 'raz']].values
        self.Y_raw = df[['refl_2.1um', 'refl_0.86um']].values
        
        # Calculate Scalers for both Inputs and Targets
        self.X_min, self.X_max = self.X_raw.min(axis=0), self.X_raw.max(axis=0)
        self.Y_min, self.Y_max = self.Y_raw.min(axis=0), self.Y_raw.max(axis=0)
        
        # Scale to [0, 1] for stable neural network convergence
        self.X_norm = (self.X_raw - self.X_min) / (self.X_max - self.X_min)
        self.Y_norm = (self.Y_raw - self.Y_min) / (self.Y_max - self.Y_min)
        
        self.X = torch.tensor(self.X_norm, dtype=torch.float32)
        self.Y = torch.tensor(self.Y_norm, dtype=torch.float32)
        
        # Save scalers for the API-VAE to use during training [cite: 37]
        np.savez('surrogate_scalers.npz', 
                 X_min=self.X_min, X_max=self.X_max,
                 Y_min=self.Y_min, Y_max=self.Y_max)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

# ==========================================
# 2. ARCHITECTURE WITHOUT THE CEILING
# ==========================================
class SurrogateRTM(nn.Module):
    def __init__(self):
        super(SurrogateRTM, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 2),
            # ReLU ensures reflectance is non-negative but allows 
            # values to exceed 1.0 (unbounded upward)[cite: 69, 72].
            nn.ReLU() 
        )

    def forward(self, x): return self.net(x)

# ==========================================
# 3. PLOTTING FUNCTION
# ==========================================
def generate_plots(Y_true, Y_pred, tau_values, solz_values):
    """Generates parity and residual plots for model validation[cite: 73]."""
    sns.set_theme(style="whitegrid")
    bands = ['2.1µm (SWIR)', '0.86µm (Visible)']
    
    # Parity Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, ax in enumerate(axes):
        r2 = r2_score(Y_true[:, i], Y_pred[:, i])
        mae = mean_absolute_error(Y_true[:, i], Y_pred[:, i])
        
        sns.regplot(x=Y_true[:, i], y=Y_pred[:, i], ax=ax, 
                    scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':'red', 'ls':'--'})
        
        ax.set_title(f"Fidelity: {bands[i]}\n$R^2$: {r2:.5f} | MAE: {mae:.4f}")
        ax.set_xlabel("True DISORT Reflectance")
        ax.set_ylabel("Neural Surrogate Prediction")
        
        # 1:1 Reference Line
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, 'k-', alpha=0.5, zorder=0)

    plt.tight_layout()
    plt.savefig("rtm_fidelity_parity.png", dpi=300)
    plt.show()

    # Residual Analysis (Focusing on 0.86um Visible Band)
    residuals = Y_true[:, 1] - Y_pred[:, 1]
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tau_values, residuals, alpha=0.3, s=8, c=solz_values, cmap='plasma')
    plt.axhline(0, color='red', linestyle='--')
    plt.colorbar(scatter, label='Solar Zenith Angle (deg)')
    
    plt.title("Residual Analysis: Error vs. Optical Depth (τ)")
    plt.xlabel("True Optical Depth (τ)")
    plt.ylabel("Reflectance Residual (True - Pred)")
    
    # Highlight non-linear regime (tau < 10) for Trust Mask context 
    plt.axvspan(0, 10, color='gray', alpha=0.1, label='Non-linear Regime (τ < 10)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("rtm_residual_analysis.png", dpi=300)
    plt.show()

# ==========================================
# 4. TRAINING & VALIDATION SCRIPT
# ==========================================
def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SurrogateDataset("cloud_training_data_exp.nc")
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)
    test_loader = DataLoader(test_ds, batch_size=256)

    model = SurrogateRTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(100):
        model.train()
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad(); loss = criterion(model(batch_X), batch_Y)
            loss.backward(); optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1} Complete")

    # Evaluation and Plotting Preparation
    model.eval()
    all_preds, all_true = [], []
    all_tau, all_solz = [], []
    
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            preds = model(batch_X)
            
            # Inverse scale back to reflectance units
            Y_min = torch.tensor(dataset.Y_min, device=device)
            Y_max = torch.tensor(dataset.Y_max, device=device)
            unscaled_preds = preds * (Y_max - Y_min) + Y_min
            unscaled_true = batch_Y * (Y_max - Y_min) + Y_min
            
            all_preds.append(unscaled_preds.cpu())
            all_true.append(unscaled_true.cpu())
            
            # Unscale inputs to get original tau and solz for residual analysis
            X_min = torch.tensor(dataset.X_min, device=device)
            X_max = torch.tensor(dataset.X_max, device=device)
            unscaled_X = batch_X * (X_max - X_min) + X_min
            all_tau.append(unscaled_X[:, 1].cpu()) # tau_true index
            all_solz.append(unscaled_X[:, 3].cpu()) # solz index

    Y_p = torch.cat(all_preds).numpy()
    Y_t = torch.cat(all_true).numpy()
    tau_vals = torch.cat(all_tau).numpy()
    solz_vals = torch.cat(all_solz).numpy()
    
    # Output metrics
    for i, band in enumerate(['2.1µm', '0.86µm']):
        print(f"Band {band} | R2: {r2_score(Y_t[:,i], Y_p[:,i]):.5f} | MAE: {mean_absolute_error(Y_t[:,i], Y_p[:,i]):.5f}")

    # Generate Plots
    generate_plots(Y_t, Y_p, tau_vals, solz_vals)

    torch.save(model.state_dict(), "surrogate_rtm_weights.pth")

if __name__ == "__main__":
    run_training()