import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# 1. THE ADVERSARIAL COMPONENTS (GRL)
# ==========================================
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_weight):
        ctx.lambda_weight = lambda_weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_weight, None

class GRL(nn.Module):
    def __init__(self, lambda_weight=1.0):
        super(GRL, self).__init__()
        self.lambda_weight = lambda_weight

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.lambda_weight)

class Discriminator(nn.Module):
    def __init__(self, latent_dim_y=4, latent_dim_x=2, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim_y, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim_x),
            nn.Sigmoid()  # <-- Added to match zx_scaled bounds [0, 1]
        )

    def forward(self, zy):
        return self.net(zy)

# ==========================================
# 2. VAE ENCODER & DECODER COMPONENTS
# ==========================================
class PartitionedEncoder(nn.Module):
    def __init__(self, input_dim=2, condition_dim=3, track_dim=2, latent_dim_x=2, latent_dim_y=4, hidden_dim=128):
        super(PartitionedEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim + condition_dim + track_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mu_x     = nn.Linear(hidden_dim, latent_dim_x)
        self.logvar_x = nn.Linear(hidden_dim, latent_dim_x)
        self.mu_y     = nn.Linear(hidden_dim, latent_dim_y)
        self.logvar_y = nn.Linear(hidden_dim, latent_dim_y)
        self.activation = nn.SiLU()

    def forward(self, x_obs, c_geom, c_track):
        h = self.activation(self.fc1(torch.cat([x_obs, c_geom, c_track], dim=1)))
        h = self.activation(self.fc2(h))
        logvar_x = torch.clamp(self.logvar_x(h), -4.0, 4.0)
        logvar_y = torch.clamp(self.logvar_y(h), -4.0, 4.0)
        return self.mu_x(h), logvar_x, self.mu_y(h), logvar_y

class ResidualDecoder(nn.Module):
    def __init__(self, latent_dim_y=4, condition_dim=3, output_dim=2, hidden_dim=64):
        super(ResidualDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim_y + condition_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, output_dim) 
        )
    def forward(self, zy, c_geom):
        return 0.05 * torch.tanh(self.net(torch.cat([zy, c_geom], dim=1)))

class AuxPredictor(nn.Module):
    def __init__(self, latent_dim_y=4, target_dim=3, hidden_dim=64):
        super(AuxPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim_y, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, target_dim)
        )
    def forward(self, zy):
        return self.net(zy)

# ==========================================
# 3. PHYSICAL PLUME COMPONENT (Hardcoded Bias)
# ==========================================
class ExponentialTwomeyPlume(nn.Module):
    def __init__(self):
        super(ExponentialTwomeyPlume, self).__init__()
        # Initializing near data generation parameters
        self.k_eff = nn.Parameter(torch.tensor([-0.71]))     # softplus(-0.71) ≈ 0.4 h^-1, matching K_DECAY in data gen
        self.nd_boost = nn.Parameter(torch.tensor([5.0]))    
        self.max_dilution = nn.Parameter(torch.tensor([0.1]))

    def forward(self, age):
        # Enforce positive constraints for physical constants
        k = F.softplus(self.k_eff)
        boost = F.softplus(self.nd_boost)
        # Sigmoid ensures dilution fraction stays between 0 and 1
        dil_limit = torch.sigmoid(self.max_dilution) 

        # Exponential decay of aerosol concentration
        c_t = torch.exp(-k * age)
        return c_t, boost, dil_limit

# ==========================================
# 4. THE MASTER API-VAE CLASS
# ==========================================
class API_VAE(nn.Module):
    def __init__(self, surrogate_model, scalers, device):
        super(API_VAE, self).__init__()
        self.device = device
        
        # Scaling parameters (Physics Branch)
        self.X_min_2d = torch.tensor(scalers['X_min'][:2], dtype=torch.float32).to(device) 
        self.X_max_2d = torch.tensor(scalers['X_max'][:2], dtype=torch.float32).to(device)
        self.v_min = float(scalers['X_min'][2])
        self.v_max = float(scalers['X_max'][2])

        # Reflectance Bounds (Output Un-scaling)
        self.Y_min = torch.tensor(scalers['Y_min'], dtype=torch.float32).to(device)
        self.Y_max = torch.tensor(scalers['Y_max'], dtype=torch.float32).to(device)

        # Networks
        self.encoder = PartitionedEncoder()
        self.plume_network = ExponentialTwomeyPlume()
        self.residual_decoder = ResidualDecoder()
        self.discriminator = Discriminator()
        self.grl = GRL(lambda_weight=1.0)
        self.aux_weather = AuxPredictor(latent_dim_y=4, target_dim=3) 
        
        self.surrogate = surrogate_model
        for param in self.surrogate.parameters():
            param.requires_grad = False 

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x_refl, c_track, c_geom_scaled):
        # 1. Latent Encoding
        mu_x, logvar_x, mu_y, logvar_y = self.encoder(x_refl, c_geom_scaled, c_track)
        zx_raw = self.reparameterize(mu_x, logvar_x)
        zy = self.reparameterize(mu_y, logvar_y)
        
        # 2. Physics Recovery (Estimating Clean Background State)
        zx_scaled = torch.sigmoid(zx_raw)
        physics_real = zx_scaled * (self.X_max_2d - self.X_min_2d) + self.X_min_2d
        re_clean, tau_clean = physics_real[:, 0:1], physics_real[:, 1:2]
        veff_clean = torch.full_like(re_clean, 0.12) # Fixed background prior

        # 3. Causal Transformation (Aerosol-Cloud Interactions)
        proximity, age = c_track[:, 0:1], c_track[:, 1:2]
        c_t, boost, dil_limit = self.plume_network(age)
        
        # A. LWP fractional change due to entrainment (proximity-weighted)
        lwp_factor = 1.0 - (dil_limit * (1.0 - c_t) * proximity)

        # B. Twomey Effect: re ∝ (LWP/N_d)^(1/3) — jointly accounts for Nd increase and LWP change
        nd_ratio = 1.0 + (boost * c_t * proximity)
        re_final = re_clean * (lwp_factor ** (1.0 / 3.0)) * (nd_ratio ** (-1.0 / 3.0))

        # C. Tau from LWP and re: tau = (3/2)*LWP/(rho_w*re), so tau ∝ LWP/re
        tau_final = tau_clean * lwp_factor * (re_clean / re_final)

        # D. Dispersion Effect (Veff Transition) — interpolates toward 0.05 only when on-track
        veff_final = veff_clean + (0.05 - veff_clean) * c_t * proximity
        
        # 4. Surrogate Pass (Rescaling to RTM domain)
        re_s = (re_final - self.X_min_2d[0]) / (self.X_max_2d[0] - self.X_min_2d[0])
        tau_s = (tau_final - self.X_min_2d[1]) / (self.X_max_2d[1] - self.X_min_2d[1])
        veff_s = (veff_final - self.v_min) / (self.v_max - self.v_min)
        
        norm_refl = self.surrogate(torch.cat([re_s, tau_s, veff_s, c_geom_scaled], dim=1))
        
        # Un-scale to Physical Units
        base_refl_phys = norm_refl * (self.Y_max - self.Y_min) + self.Y_min
        
        # 5. Final Reconstruction & Adversarial Predictions
        recon_refl = base_refl_phys + self.residual_decoder(zy, c_geom_scaled)
        
        pred_zx_from_zy = self.discriminator(self.grl(zy))
        pred_weather = self.aux_weather(zy)
        
        return (recon_refl, mu_x, logvar_x, mu_y, logvar_y, 
                zx_scaled, pred_zx_from_zy, pred_weather)