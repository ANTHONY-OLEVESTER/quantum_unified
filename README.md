# 🧠 Quantum Unified  
*A computational proof-chain for the Curvature–Information Principle*  
> _“Flatness and D⁻¹ concentration under 2-designs.”_ — A. Olevester (2025)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17497059.svg)](https://doi.org/10.5281/zenodo.17497059)
[![PyPI version](https://badge.fury.io/py/quantum-unified.svg)](https://pypi.org/project/quantum-unified/)

---

## 🌌 Overview  
This repository contains the complete numerical and theoretical workflow supporting the paper  
**“A Universal Curvature–Information Principle: Flatness and D⁻¹ Concentration under 2-Designs.”**

It reconstructs how the invariant:

> **Y = sqrt(d_eff - 1) · (A² / I)**

emerges as a universal coupling between **quantum curvature** (Bures/Uhlmann geometry) and **mutual information**.

Each *Phase (0 → 9)* builds on the previous, converging to the theorem:

```
E[Y] = Y0 + O(D⁻¹)
Var(Y) = Θ(D⁻¹)
E[α]  = O(D⁻¹)
```

---

## 📁 Repository Structure

| Path | Description |
|------|--------------|
| `src/` | All simulation scripts (`phase0_baseline.py` → `phase9_plus.py`) |
| `data/` | CSV datasets generated from simulations |
| `figures/` | Auto-generated plots |
| `paper/` | LaTeX sources |
| `scripts/` | Build + bundling utilities |
| `Makefile` | Run phases, build figures, generate arXiv bundle |

---

## 🧩 Phase-Wise Roadmap

| Phase | Goal | Script | Output / What it Proves |
|-------|------|--------|--------------------------|
| **0 – Baseline Geometry** | Validate Bures/Uhlmann curvature and purity logic | `phase0_baseline.py` | Initial sanity check |
| **1 – Random State Test** | Haar random density matrices | `phase1_random_state.py` | Y stabilizes |
| **2 – Universality Sweep** | Chaotic vs structured vs twirled channels | `phase2_universality_sweep.py` | Twirl → flatness restored |
| **3 – Variance Scaling** | Var(Y) vs D | `phase3_varY_by_D.py` | Raw D⁻¹ slope data |
| **4 – α Regression** | Fit α vs 1/D | `phase4_alpha_vs_invD.py` | α → 0 intercept |
| **5 – Stinespring Extension** | Open quantum system evolution | `phase5_stinespring.py` | Same scaling under channels |
| **6 – Theorem Validation** | 2-design Weingarten sampler | `phase6_theorem_perD.py` | First convergence proof |
| **7 – WLS Refinement** | Weighted regression + bootstraps | `phase7_wls.py` | β ≈ –1.000 |
| **8 – Bootstrap Audit** | Resampling confidence check | `phase8_bootstrap.py` | Robustness proven |
| **9 – Haar Final Proof (GPU)** | High precision sampling | `phase9_plus.py` | Final numbers used in the paper |

---

## 🧪 Running Simulations

### Install dependencies

```
pip install -r requirement.txt
```

### Minimal run (sanity)

```
python src/phase0_baseline.py --trials 100
```

## Simulation Phases and How to Reproduce Them
| Phase | Focus | Key Outputs | How to Run |
| --- | --- | --- | --- |
| 0 - Foundation | Analytical checks, small-system experiments, regression guards | `phase0_proofs/`, `phase0_tests/` scripts, CSV snapshots | Run individual scripts such as `python phase0_tests/universality_suite.py`; add `pytest` wrappers as desired. |
| 1 - Universality Sweep | Chaotic vs structured dynamics, channel noise, bootstrap CIs | `phase3-out/universality_sweep.csv` | `python scripts/phase1_universality.py --seed 42` (writes CSV in place). |
| 2 - Collapse Panels & Finite-Size Scaling | Extended grids, gamma fits, pooled flatness metrics | `phase3-out/phase2_*`, `phase3-out/finite_size_gamma.csv` | `python src/phase2.py --sweep --output-dir phase3-out` then rerun with `--plots-only` to regenerate figures. |
| 3 - Alpha vs 1/D | 10M-point scaling confirming drift to zero | `data/phase3_varY_by_D.csv`, `figures/phase3_alpha_vs_invD.png` | Produced during the Phase 2 sweep; reuse `phase3-out` cache or repeat the `--sweep`. |
| 4 - Asymptotics | Large-D extrapolation and concentration slopes | `phase4-out/phase4_alpha_vs_invD.csv`, PNGs | `python src/phase4_asymptotics.py --output-dir phase4-out --max-qubits 9`. |
| 5 - Design Concentration | Robustness under perturbations, 2-design verification | `phase5_design_concentration.py` logs, tables ingested into the paper | `python src/phase5_design_concentration.py --output-dir phase5-out`. |
| 6 - Fast Isotropic Asymptotics | Accelerated sampler benchmarking the D^-1 law | `phase6-out/data/phase6_theorem_perD.csv`, `phase6-out/figures/phase6_*` | `python src/phase6_fast.py --output-dir phase6-out`. |
| 7 - Stinespring 2-Design | Haar isometries, variance slope confirmation | `phase7-out/data/*.csv`, `phase7-out/figures/*.png` | `python src/phase7_stinespring_2design.py --output-dir phase7-out`. |
| 8 - Diagnostics | Log-log variance audits, regression plots | `phase8-out/figures/phase8_var_scaling.png`, `phase8-out/phase8_summary.txt` | `python src/phase8_diagnostics.py --output-dir phase8-out`. |
| 9 - Signed Alpha CIs | Bootstrap confidence intervals, Haar extensions | `phase9-out/data/*.csv`, `phase9-plus-haar*/figures/*.png` | `python src/phase9_signed_alpha_CI.py --output-dir phase9-out` and `python src/phase9_plus.py --output-dir phase9-plus-haar`. |

Each phase script exposes `--help`; adjust qubit counts, trial numbers, or output directories there when scaling experiments.

### Full reproduction (Phase IX, GPU)

```
python src/phase9_plus.py --sampler haar --device gpu --nE 7-14 --trials 3000 --seeds-per-D 10 --boot-B-point 12000 --boot-B-intercept 12000 --wls --debias lodo --workers 8 --outdir phase9-plus-haar-extend
```

Expected output:

```
α@Dmax: mean=+0.2868, CI=[-0.5377,+1.1088]  -> PASS
Var(Y) slope β = −0.999 [−1.004, −0.995]
```

---

## 📊 Regenerate all figures

```
make figures
```

---

## 🧮 Theory Snapshot

Y couples three quantities of an open evolution:
- `A²` — Bures/Uhlmann curvature
- `I` — Mutual information
- `d_eff` — Effective Hilbert-space dimension (inverse purity)

Under a unitary **2-design**:
```
E[Y]   = Y0 + O(D⁻¹)
Var(Y) = Θ(D⁻¹)
|α|    ~ O(D⁻¹/²)
```

---

## 🧠 Conclusions

- α converges → 0  ✅ flatness
- Var(Y) follows D⁻¹ ✅ universal variance law
- Twirling restores isotropy ✅ proof of universality

---

## 📚 Citation
If you use this repository or data:

```
@article{Olevester2025CurvatureInformation,
  author  = {Anthony Olevester},
  title   = {A Universal Curvature–Information Principle: Flatness and D⁻¹ Concentration under 2-Designs},
  year    = {2025},
  doi     = {10.5281/zenodo.17497059},
  note    = {https://pypi.org/project/quantum-unified/}
}
```

---

## 👤 Author

**Anthony Olevester**  
📧 olevester.joram123@gmail.com  
🌐 [https://anthony-olevester.github.io/quantum_unified](https://anthony-olevester.github.io/quantum_unified)

---
> “Flatness → Universality. Variance → D⁻¹. Civilization → Accelerated.”
