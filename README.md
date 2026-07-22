# credible_ML_clouds

Code for "Toward credible machine-learning counterfactual retrievals of unperturbed cloud properties."

Each script maps to a specific result in the paper:

| Script | Produces |
|---|---|
| `train_apivae_v2b.py` | Identifiability experiment (Table: counterfactual decomposition under different optimisation constraints); modes `cleanonly`, `trueinit`, `pair`, `oracle` |
| `canonical_eval_v2.py` | Recovery metrics (R2, MAE) and plume-parameter recovery for any trained variant |
| `gen_stochastic_states.py` | Stochastic-plume benchmark generation |
| `check_age_shortcut.py` | Age-explained variance fraction (deterministic vs. stochastic benchmark) |
| `train_ablation_v3.py` | Component ablation table |
| `retrain_surrogate.py` | Radiative-transfer surrogate training |
| `collocate_real_data.py` | MODIS granule collocation (mask + L1B + MYD03 + MYD06) |
| `real_data_inference.py` | Shared inference + per-granule calibration for the real-data study |
| `delta_skill_eval.py` | Real-data delta-skill diagnostics (per-pixel and within-granule correlation) |
| `segment_skill_bootstrap.py` | Segment-level skill metric with granule-clustered bootstrap CI |
| `build_real_pairs.py` | Real (track, background) pair construction for fine-tuning |
| `finetune_real_pairs.py` | Real-pair fine-tuning of the pair-consistency objective |
| `finetune_eval.py` | Held-out delta-skill evaluation of the fine-tuned model |
| `make_ambiguity_figure.py` | Figure: recovery R2 across optimisation-constraint variants and per-seed spread |
| `make_benchmark_shortcut_figure.py` | Figure: perturbation ratio vs. plume age, with a linear age-only fit, for both benchmarks |
| `make_segment_skill_figure.py` | Figure: real-data per-pixel vs. segment-level within-granule delta-skill scatter |
| `make_ablation_figure.py` | Figure: component-ablation recovery bar chart |

`plot_style.py` holds shared matplotlib styling used by the figure scripts.

`src/model_exp.py` and `src/neural_surrogate.py` are the model and surrogate radiative-transfer emulator these scripts train and evaluate.

Data (MODIS L1B/MYD03/MYD06, ship-track masks, synthetic DISORT states) is not included.
