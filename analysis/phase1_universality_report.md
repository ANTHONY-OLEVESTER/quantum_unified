# Quantum Formula Universality Study (2025-10-27)

## Repository housekeeping
- Created dedicated directories: `docs/` for LaTeX assets, `tests/` for numerical experiments, `proofs/` for derivation sketches, and `analysis/` for generated data.
- Updated `main.tex` graphic path and `tools/build_figures.py` so TeX and figure builders respect the new layout.
- Added package initialisers to `tests/` and `proofs/` to allow clean module imports by the tooling.

## Simulation highlights
- Ran `tests/universality_suite.py` (new) to benchmark the collapse invariant across contrasting dynamics. Results stored in `analysis/universality_results.json` and summarised below.
- Chaotic random two-body unitary (n_S=n_E=2, κ=0.60): slope α≈−0.129, coefficient of variation for Y ≈0.31. Collapse invariant remains flat within numerical noise.
- Integrable XXZ chain (Δ=1.2): slope α≈−0.406 with broader spread (CV≈0.39), signalling systematic drift away from the universal √ scaling.
- Amplitude damping (γ=0.35): Y contracts sharply (⟨Y⟩≈0.067) and fluctuations dominate (CV≈0.77), illustrating breakdown under strong dissipation.
- Measurement-induced dephasing (p=0.40): positive slope α≈+0.206 with high R², confirming collapse failure when repeated measurements dominate.

## Figure regeneration
- Verified the figure pipeline with `python tools/build_figures.py --only beta-vs-dimension`, `--only collapse-k0p6`, `--only information-fidelity`, and `--only fourqubit-summary`. Outputs saved under `figs/`.

## Outstanding items & recommended next steps
1. Analytic programme: formalise Haar/Weingarten derivation of Y-invariance and connect to Fisher/Bures curvature (sections §II–§VI of `docs/main.tex` outline the roadmap).
2. Additional counterexamples: extend `tests/universality_suite.py` with integrable XXZ at different Δ, open-system Lindblad chains, and topological or measurement-induced transition models (e.g., toric code snapshots).
3. Finite-size collapse: automate dimension sweeps (up to 2¹²) and gather scaling fits (γ) for inclusion in the manuscript.
4. Experimental anchoring: draft protocols for trapped-ion or superconducting implementations—identify observables needed (local purity, mutual information, interferometric phase).
5. Cross-domain links: relate the invariant numerics to fluctuation–dissipation and ETH limits; document classical Fisher curvature correspondence in proofs.
6. Robustness tests: add noise injection, Kraus-rank scans, thermal initial states, and time-series tracking Y(t).

## Data artefacts
- Numerical summary: `analysis/universality_results.json`
- Detailed notebook/script: `tests/universality_suite.py`
- Regenerated figures: `figs/fig_*.pdf` / `.png`

These assets can be cited directly in the LaTeX source (update `docs/main.tex`) for the next manuscript draft.
