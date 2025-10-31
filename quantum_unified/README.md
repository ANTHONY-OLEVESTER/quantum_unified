Quantum Unified Formula — Python utilities
=========================================

This package contains small, dependency‑light helpers for computing the
curvature–information invariant

  Y = sqrt(d_eff - 1) * A^2 / I

where A is the Bures (Uhlmann) angle and I is mutual information (bits).

Functions

- bures_angle(rho, sigma) → float
- effective_dimension(rho) → float
- von_neumann_entropy_bits(rho) → float
- mutual_information_bits(rhoS, rhoE, rhoSE) → float
- compute_Y(rhoS, rhoS_prime, Ibits=None) → float
- compute_Y_from_SE(rhoS, rhoS_prime, rhoSE_post, dims=(dS,dE)) → float

Notes

- All matrices should be NumPy complex arrays; functions internally
  hermitize and clamp small negatives for numerical stability.
- Mutual information requires the post‑channel joint state; if you only
  have reduced states, pass Ibits directly to compute_Y.

Quick example

```python
import numpy as np
from quantum_unified import bures_angle, effective_dimension, compute_Y, mutual_information_bits

# toy qubit states
rho0 = np.array([[1,0],[0,0]], dtype=complex)
rho1 = np.array([[0.9, 0.1],[0.1, 0.1]], dtype=complex)

A = bures_angle(rho0, rho1)
deff = effective_dimension(rho1)
Ibits = 0.5  # example MI in bits (from dilation)
Y = compute_Y(rho0, rho1, Ibits)
print(A, deff, Y)
```

