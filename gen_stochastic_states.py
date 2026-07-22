"""
Generate the stochastic-plume state table for DISORT regeneration.

Plume parameters (k, B, delta) are drawn per-pair from distributions
instead of global constants, so the perturbation is not deterministic
given plume age alone. Dilution is applied in both the re and tau
channels, matching the model_exp forward equations exactly.

Output: data/v3_states.csv -- one row per cloud state (2 per pair), with all
DISORT inputs (re, tau, veff, solz, satz, raz) plus provenance columns
(pair_id, type, age, k, B, delta). Run each row through libRadtran to get
refl_0.86um / refl_2.1um.
"""
import numpy as np
import pandas as pd
import os, sys

# Defaults generate the TRAINING states. For the held-out TEST split use a
# DIFFERENT seed and output name, e.g.:
#   python gen_stochastic_states.py 271828 10000 v3_test_states.csv
SEED    = int(sys.argv[1]) if len(sys.argv) > 1 else 314159
N_PAIRS = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
OUT_NAME = sys.argv[3] if len(sys.argv) > 3 else 'v3_states.csv'
OUT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', OUT_NAME)

RE_MIN_LUT = 5.5          # keep polluted re inside the Mie LUT range used previously

rng = np.random.default_rng(SEED)

# clean-state priors (unchanged from v2 dataset)
re_cln   = rng.uniform(10.0, 18.0, N_PAIRS)
tau_cln  = rng.uniform(8.0, 30.0, N_PAIRS)
veff_cln = rng.uniform(0.10, 0.15, N_PAIRS)
age      = rng.uniform(0.0, 1.0, N_PAIRS)

# geometry per pair (shared by both members, as before)
solz = rng.uniform(10.0, 80.0, N_PAIRS)
satz = rng.uniform(0.0, 60.0, N_PAIRS)
raz  = rng.uniform(0.0, 180.0, N_PAIRS)

# per-pair stochastic plume parameters
# centred on the v2 generating values (k=4.8, B=5.0, delta=0.10) with
# meaningful spread; resample any pair whose polluted re would fall below
# the Mie LUT floor.
def draw_plume(n):
    k = rng.lognormal(mean=np.log(4.8), sigma=0.45, size=n)          # ~[2.0, 11.6] 95%
    B = rng.lognormal(mean=np.log(5.0), sigma=0.55, size=n)          # ~[1.7, 14.7] 95%
    B = np.clip(B, 1.0, 12.0)
    d = rng.beta(2.0, 3.0, size=n) * 0.25                            # mean 0.10, in [0, 0.25]
    return k, B, d

k, B, delta = draw_plume(N_PAIRS)

def forward(re, tau, veff, age, k, B, d):
    """model_exp forward equations exactly (dilution in BOTH channels)."""
    c   = np.exp(-k * age)
    lam = 1.0 - d * (1.0 - c)          # LWP dilution factor
    nd  = 1.0 + B * c                  # Nd enhancement factor
    re_p   = re * lam**(1/3) * nd**(-1/3)
    tau_p  = tau * lam * (re / re_p)
    veff_p = veff + (0.05 - veff) * c
    return re_p, tau_p, veff_p

re_pol, tau_pol, veff_pol = forward(re_cln, tau_cln, veff_cln, age, k, B, delta)

# rejection: redraw plume params where polluted re leaves the LUT range
bad = re_pol < RE_MIN_LUT
it = 0
while bad.any():
    it += 1
    nk, nB, nd_ = draw_plume(int(bad.sum()))
    k[bad], B[bad], delta[bad] = nk, nB, nd_
    re_pol[bad], tau_pol[bad], veff_pol[bad] = forward(
        re_cln[bad], tau_cln[bad], veff_cln[bad], age[bad], k[bad], B[bad], delta[bad])
    bad = re_pol < RE_MIN_LUT
    if it > 50:
        raise RuntimeError('rejection sampling failed to converge')
print(f'rejection iterations: {it}')

# sanity check: perturbation should not be deterministic given age alone
ratio = re_pol / re_cln
bins  = np.digitize(age, np.linspace(0, 1, 21))
within = np.mean([ratio[bins == b].std() for b in range(1, 21)])
print(f'age-explained fraction of ratio variance: {1 - within**2 / ratio.var():.3f}')

# assemble long-format table
pair_id = np.arange(N_PAIRS)
def rows(kind, re, tau, veff):
    return pd.DataFrame({
        'pair_id': pair_id, 'type': kind, 'age': age,
        're_true': re, 'tau_true': tau, 'veff_true': veff,
        'solz': solz, 'satz': satz, 'raz': raz,
        'k_true': k, 'B_true': B, 'delta_true': delta,
    })

df = pd.concat([rows('clean', re_cln, tau_cln, veff_cln),
                rows('polluted', re_pol, tau_pol, veff_pol)],
               ignore_index=True)
df.to_csv(OUT, index=False, float_format='%.6f')

print(f'\n{len(df):,} states -> {OUT}')
for c in ['re_true', 'tau_true', 'veff_true']:
    print(f'  {c:10s} [{df[c].min():.3f}, {df[c].max():.3f}]')
print(f'  plume k [{k.min():.2f},{k.max():.2f}]  B [{B.min():.2f},{B.max():.2f}]  '
      f'delta [{delta.min():.3f},{delta.max():.3f}]')
