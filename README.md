# quantum-unified

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17497059.svg)](https://doi.org/10.5281/zenodo.17497059)
[![PyPI](https://img.shields.io/pypi/v/quantum-unified.svg)](https://pypi.org/project/quantum-unified/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

Curvature–Information utilities: compute and analyze the invariant

  Y = sqrt(d_eff - 1) * A^2 / I

where
- A is the Bures (Uhlmann) angle between two density matrices,
- I is mutual information in bits from a Stinespring dilation, and
- d_eff = 1 / Tr(rho^2) is the effective dimension.

The package provides stable, dependency‑light helpers for fidelity/angle, effective dimension,
entropies in bits, mutual information, and convenience functions to assemble Y from reduced
or joint states.

## Install

`ash
pip install quantum-unified
`

## Quick start

`python
import numpy as np
from quantum_unified import (
    bures_angle,
    effective_dimension,
    von_neumann_entropy_bits,
    mutual_information_bits,
    compute_Y,
)

# Two qubit states (example)
rho0 = np.array([[1,0],[0,0]], dtype=complex)
rho1 = np.array([[0.9, 0.1],[0.1, 0.1]], dtype=complex)

A = bures_angle(rho0, rho1)                 # Uhlmann/Bures angle (radians)
De = effective_dimension(rho1)              # 1 / Tr(rho^2)
Y  = compute_Y(rho0, rho1, Ibits=0.5)      # needs mutual information in bits
print(A, De, Y)
`

## API

- ures_angle(rho, sigma) -> float: Bures angle A.
- effective_dimension(rho) -> float: Effective dimension d_eff.
- on_neumann_entropy_bits(rho) -> float: Entropy in bits.
- mutual_information_bits(rhoS, rhoE, rhoSE) -> float: I = S(S)+S(E)-S(SE) in bits.
- compute_Y(rhoS, rhoS_prime, Ibits) -> float: Y from reduced states and Ibits.
- partial_trace_SE(rhoSE, dims, subsystem) / compute_Y_from_SE(...): helpers with joint state.

## Background: the curvature–information invariant

Y couples the Bures geometry of quantum states (via the fidelity angle) to informational change (mutual
information). In isotropic/2‑design regimes, Y concentrates with mean Y0 + O(D^{-1}) and variance Θ(D^{-1}).
This package exposes the building blocks to evaluate Y in your own models and data.

## License

BSD 3‑Clause. See [LICENSE](LICENSE).

## Author

Anthony Olevester (olevester.joram123@gmail.com)
