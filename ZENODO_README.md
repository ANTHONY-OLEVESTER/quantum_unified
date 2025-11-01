# Universal Curvature-Information Principle: Simulation Data and Code

## Overview

This repository contains the complete simulation data, code, and analysis for the research paper:

**"A Universal Curvature-Information Principle: Flatness and D^(-1) Concentration under 2-Designs"**

Author: Anthony Olevester (olevester.joram123@gmail.com)

## Abstract

This work studies the quantum information-geometric invariant:

```
Y = sqrt(d_eff - 1) * A^2 / I
```

where:
- **A** = Bures/Uhlmann angle (quantum geometric distance)
- **I** = mutual information (in bits)
- **d_eff** = effective dimension of the quantum state

**Key Findings:**
- Flat signed-α at large dimension D
- Universal variance law: Var(Y) ∝ D^(-1)
- 2-design theorem explains concentration rates: E[Y] = Y_0 + O(D^(-1)), Var(Y) = Θ(D^(-1))
- Universality across chaotic, structured, and twirled quantum dynamics

## Repository Structure

### Core Code
- **`Quantum_unification.py`** - Main unification test script demonstrating the curvature-information-dimension law
- **`quantum_unified/`** - Python package with core utilities for computing the invariant
  - `bures_angle()` - Computes Bures/Uhlmann angle between quantum states
  - `effective_dimension()` - Computes effective dimension from density matrix
  - `mutual_information_bits()` - Computes mutual information
  - `compute_Y()` - Computes the Y invariant

### Simulation Phases

#### Phase 0: Foundation
- **`phase0_proofs/`** - Analytical derivations and information geometry proofs
- **`phase0_tests/`** - Initial numerical experiments:
  - `beta_vs_dimension.py` - Scaling exponent vs system dimension
  - `Collapse_test.py` - Invariant collapse verification
  - `COLLISION_MODEL.py` - Collision-model dynamics
  - `four_qubit_experiment.py` - 4-qubit entanglement dynamics
  - `universality_suite.py` - Tests across different dynamical models

#### Phase 1: Universality Studies
- **Scripts:** `scripts/phase1_universality.py`
- **Outputs:** `analysis/phase1_universality_report.md`, `analysis/universality_results.json`
- **Key Results:** Demonstrated collapse invariant remains flat under chaotic random unitaries (α ≈ -0.129), while integrable systems (XXZ chain) show drift (α ≈ -0.406)

#### Phase 2: Collapse Panels and Finite-Size Scaling
- **Scripts:** `src/phase2.py`, `scripts/phase2_generate_plots.py`
- **Data:** `data/finite_size_alpha_per_size.csv`, `data/finite_size_gamma.csv`
- **Figures:**
  - `figures/phase2_collapse_panels.png` - Multi-model collapse comparison
  - `figures/phase2_finite_size_gamma.png` - Finite-size corrections
  - `figures/phase2_finite_size_scaling.png` - Scaling behavior
  - `figures/phase2_twirl_restoration.png` - Twirling restores universality

#### Phase 3: Alpha vs Inverse Dimension
- **Outputs:** `phase3-out/`
- **Data:** `phase3_varY_by_D.csv` (10M data points), `finite_size_alpha_per_size.csv`
- **Figures:**
  - `phase3_alpha_vs_invD.png` - |α| vs 1/D showing finite-size drift
  - `phase3_varY_scaling.png` - Variance scaling with dimension
- **Key Results:** Variance of Y scales as D^(-1) over wide dimension range

#### Phase 4: Asymptotic Analysis
- **Scripts:** `src/phase4_asymptotics.py`
- **Outputs:** `phase4-out/`
- **Data:** `phase4_alpha_vs_invD.csv`, `phase4_varY_by_D.csv`
- **Figures:** Alpha vs 1/D extrapolation to infinite dimension limit

#### Phase 5: Design Concentration
- **Scripts:** `src/phase5_design_concentration.py`
- **Data:** `data/phase5_alpha_vs_invD.csv`, `data/phase5_robustness_eps.csv`
- **Figures:**
  - `phase5_alpha_vs_invD.png`
  - `phase5_robustness.png` - Robustness under perturbations
  - `phase5_varY_scaling.png`

#### Phase 6: Fast Isotropic Asymptotics
- **Scripts:** `src/phase6_fast.py`, `src/phase6_theorem_harness.py`
- **Outputs:** `phase6-out/`
- **Data:** `phase6_alpha_vs_invD.csv`, `phase6_theorem_perD.csv`, `phase6_varY_by_D.csv`
- **Summary:** `phase6_summary.txt`
  - Alpha intercept at 1/D→0: +0.2599 (CI: [+0.0768, +0.4483])
  - Var(Y) slope: -1.0004 (CI: [-1.0104, -0.9862])

#### Phase 7: Stinespring 2-Design
- **Scripts:** `src/phase7_stinespring_2design.py`
- **Outputs:** `phase7-out/`
- **Summary:** `phase7_summary.txt`
  - |α| vs 1/D intercept: +0.5714 (CI: [+0.3947, +0.7477])
  - Var(Y) vs D slope: -0.9990 (CI: [-1.0079, -0.9897])
- **Key Results:** Haar isometry implementation confirms 2-design predictions

#### Phase 8: Diagnostics and Variance Scaling
- **Scripts:** `src/phase8_diagnostics.py`
- **Outputs:** `phase8-out/`
- **Data:** `phase8_alpha_intercepts.csv`, `phase8_per_size.csv`, `phase8_var_scaling.csv`
- **Figures:**
  - `phase8_var_scaling.png` - log Var(Y) vs log D with slope ≈ -1
  - `phase8_alpha_intercepts.png`

#### Phase 9: Signed Alpha with Confidence Intervals
- **Scripts:** `src/phase9_signed_alpha_CI.py`, `src/phase9_plus.py`
- **Outputs:** `phase9-out/`, `phase9-out-wide/`, `phase9-plus-haar/`, `phase9-plus-haar-extend/`
- **Data:** `phase9_alpha_perD.csv`, `phase9_intercept_summary.csv`
- **Figures:**
  - `phase9_alpha_vs_invD_CI.png` - Signed α with confidence intervals
  - `phase9_alpha_hist_Dmax.png` - Distribution at maximum dimension
  - `phase9_varY_scaling.png` - Final variance scaling confirmation
- **Key Results:** Confidence intervals include 0 at largest D, confirming flatness

### Paper and Documentation
- **`paper/`** - LaTeX source for the academic paper
  - `main.tex` - Main manuscript
  - `main.pdf` - Compiled paper (Curvature-Information-Principle.pdf)
  - `appendix_theorem.tex` - Formal 2-design theorem proof
  - `references.bib` - Bibliography
  - `sections.tex` - Paper sections
- **`Curvature-Information-Principle.pdf`** - Final compiled paper
- **`arxiv_bundle/`** - arXiv submission package
- **`arxiv.tar.gz`**, **`curvature_information_arxiv.tgz`** - Compressed arXiv bundles

### Analysis and Reports
- **`analysis/`**
  - `phase1_universality_report.md` - Detailed Phase 1 results
  - `phase2_universality_plan.md` - Phase 2 research plan
  - `phase3_results_note.md` - Phase 3 findings
  - `universality_results.json` - Machine-readable results

### Figures
- **`figures/`** - Publication-quality figures (PNG format)
- **`figs/`** - Additional figures (PDF + PNG)
  - `fig_beta-vs-dimension` - Scaling exponent analysis
  - `fig_collapse-k0p6` - Collapse at coupling κ=0.6
  - `fig_fourqubit-summary` - 4-qubit system overview
  - `fig_information-fidelity` - Information-fidelity relationship
  - `fig_projection-arrow` - Geometric projection visualization
  - `fig_unification-summary` - Overall unification results

### Data Files
- **`data/`** - CSV data files from all simulation phases
  - Raw simulation outputs
  - Statistical summaries
  - Scaling analysis results

### Scripts and Utilities
- **`scripts/`**
  - `ingest_metrics.py` - Data processing for paper
  - `make_tables.py` - Table generation
  - `phase*_to_tex.py` - TeX conversion utilities
  - `build_paper.ps1` - Paper build automation

### Tools
- **`tools/`** - Build and analysis utilities

### Archive
- **`archive/`** - Historical versions and backups

## Software Requirements

### Python Dependencies
See `requirement.txt` for complete list. Key packages:
- numpy >= 2.2.6
- scipy >= 1.15.3
- matplotlib >= 3.10.7
- pandas >= 2.3.3
- seaborn >= 0.13.2
- sympy >= 1.14.0
- numba >= 0.62.1 (for performance)
- tqdm >= 4.67.1 (progress bars)

Optional:
- cupy-cuda11x >= 13.6.0 (GPU acceleration)

### Installation

```bash
# Install the quantum_unified package
pip install -e .

# Or manually install dependencies
pip install -r requirement.txt
```

## Usage Examples

### Computing the Invariant

```python
import numpy as np
from quantum_unified import bures_angle, effective_dimension, compute_Y, mutual_information_bits

# Example: Two qubit states
rho0 = np.array([[1, 0], [0, 0]], dtype=complex)  # Pure |0⟩
rho1 = np.array([[0.9, 0.1], [0.1, 0.1]], dtype=complex)  # Mixed state

# Compute geometric distance
A = bures_angle(rho0, rho1)

# Compute effective dimension
d_eff = effective_dimension(rho1)

# Compute Y invariant (requires mutual information from joint state)
I_bits = 0.5  # Example value
Y = compute_Y(rho0, rho1, I_bits)

print(f"Bures angle: {A:.4f}")
print(f"Effective dimension: {d_eff:.4f}")
print(f"Y invariant: {Y:.4f}")
```

### Running the Main Unification Test

```bash
python Quantum_unification.py
```

This will:
1. Test across different coupling types (dephasing, partial swap, random)
2. Sweep system sizes (2-qubit and 4-qubit)
3. Vary coupling strengths κ
4. Generate log-log scatter plots showing A²/I vs (d_eff - 1)
5. Fit slopes to verify α ≈ -0.5 relationship

### Running Individual Phase Simulations

```bash
# Phase 2: Collapse panels
python src/phase2.py

# Phase 6: Fast asymptotics
python src/phase6_fast.py

# Phase 9: Signed alpha analysis
python src/phase9_signed_alpha_CI.py
```

### Building the Paper

```bash
# From repository root
make

# Or from paper directory
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Key Results Summary

### Universality
- Chaotic dynamics (random 2-body unitaries): α ≈ -0.129, CV(Y) ≈ 0.31
- Integrable systems (XXZ chain, Δ=1.2): α ≈ -0.406, CV ≈ 0.39 (drift from universal)
- Amplitude damping (γ=0.35): Y contracts, CV ≈ 0.77 (dissipation breakdown)
- Dephasing (p=0.40): α ≈ +0.206 (measurement-induced failure)

### Scaling Laws
- **Variance scaling:** Var(Y) ∝ D^(-1) with slope β ≈ -1.00 (CI: [-1.01, -0.99])
- **Alpha flatness:** |α| → 0 as D → ∞, with O(D^(-1)) corrections
- **2-design theorem:** E[Y] = Y_0 + O(D^(-1)), confirmed across phases

### Concentration Rates
- Phase 6: Var(Y) slope = -1.0004 (bootstrap CI: [-1.0104, -0.9862])
- Phase 7: Var(Y) slope = -0.9990 (bootstrap CI: [-1.0079, -0.9897])
- Phase 9: Signed α confidence intervals include 0 at largest D

## Theoretical Framework

The invariant Y couples:
1. **Bures/Uhlmann geometry** - Monotone Riemannian metric on quantum state space
2. **Mutual information** - Entropic measure of correlations
3. **Effective dimension** - Purity-based measure of state mixedness

Under unitary 2-designs (approximating Haar-random evolutions):
- Weingarten calculus yields asymptotic flatness
- Concentration of measure gives D^(-1) variance scaling
- Connects to Eigenstate Thermalization Hypothesis (ETH) in chaotic systems

## Related Work

This research builds on:
- Uhlmann fidelity and Bures metric (Uhlmann 1976, Hübner 1992)
- Unitary designs and random circuits (Dankert 2009, Brandão 2016)
- Quantum thermalization and ETH (D'Alessio 2016)
- Page curves and entanglement (Page 1993)
- Randomized benchmarking (Magesan 2012)

## Citation

If you use this code or data, please cite:

```
Olevester, A. (2024). A Universal Curvature-Information Principle:
Flatness and D^(-1) Concentration under 2-Designs.
arXiv preprint [quant-ph].
```

## License

[Specify license - typically MIT or GPL for research code, CC-BY for data]

## Contact

Anthony Olevester
Email: olevester.joram123@gmail.com

## Acknowledgments

This work uses standard scientific Python libraries (NumPy, SciPy, Matplotlib)
and draws on quantum information theory frameworks developed by the community.

## Files Not Included

The following directories are excluded from this Zenodo archive:
- `.venv/` - Python virtual environment (user-specific)
- `quantum_Git_Repo/` - Separate git repository (redundant)
- `__pycache__/` - Python bytecode cache (auto-generated)
- `.claude/` - Development tools configuration

## Version Information

- Archive Date: November 1, 2025
- Code Version: As of final Phase 9+ simulations
- Paper Version: Draft for arXiv submission

---

**Note:** This is a computational research archive. All simulations can be
reproduced by running the provided Python scripts with the dependencies listed
in `requirement.txt`.
