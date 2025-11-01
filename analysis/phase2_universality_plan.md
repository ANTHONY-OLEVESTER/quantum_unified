# Phase II Universality Scaling Plan

This document captures the pre-registered Phase-II battery implemented in `Curvature–Information-Universality.py`.

## Grids
- **Unitary models** `random2body`, `pswap`, `dephasing`
  - Sizes `(n_S,n_E) ∈ {(1,1),(1,2),(2,1),(2,2),(3,1),(1,3)}`
  - Couplings `κ ∈ {0.40, 0.60, 0.80}`
  - Trials per grid point: `400`
- **Depolarizing channels** `d ∈ {2,3,5,7}`, `λ ∈ {0.70, 0.90}`, trials `2000`
- **Amplitude damping** `p ∈ {0.10,0.30,0.50,0.80}`, twirl depth `m ∈ {0,1,3,5}`, trials `3000`

## Reported statistics
For every grid point and pooled model:
- OLS slope `α` on `(log10(d_eff-1), log10 Y)`
- Bootstrap mean and 95% CI (B=2000)
- Kendall `τ`, Spearman `ρ`
- Theil–Sen slope
- Finite-size exponent `γ` (unitary models)
- Pass/fail flag per criteria (`|α|≤0.08`, CI includes 0, `|τ|≤0.10`, `γ≥0.8` with CI ≥0.5)

## Artifacts
- `universality_sweep.csv`
- `universality_pooled.csv`
- `finite_size_gamma.csv`

## Deterministic seeding
`seed = hash32(kind, model, size_label, param, extra)` ensures reproducibility.
