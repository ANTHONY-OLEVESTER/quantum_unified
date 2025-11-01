"""
quantum_unified: Core utilities for the curvature–information invariant

Exports
 - bures_angle(rho, sigma) -> float
 - effective_dimension(rho) -> float
 - mutual_information_bits(rhoS, rhoE, rhoSE) -> float
 - compute_Y(rhoS, rhoS_prime, Ibits=None) -> float
 - compute_Y_from_SE(rhoS, rhoS_prime, rhoSE, dims) -> float
"""

from __future__ import annotations
import numpy as np

EPS = 1e-12

def _is_hermitian(x: np.ndarray, tol: float = 1e-8) -> bool:
    return np.allclose(x, x.conj().T, atol=tol, rtol=0)

def _psd_sqrt(mat: np.ndarray) -> np.ndarray:
    """Hermitian PSD matrix square root via eigen-decomposition."""
    if not _is_hermitian(mat):
        mat = 0.5 * (mat + mat.conj().T)
    w, v = np.linalg.eigh(mat)
    w = np.clip(w, 0.0, None)
    return (v * np.sqrt(w)) @ v.conj().T

def _psd_trace_sqrt(mat: np.ndarray) -> float:
    """Return Tr(sqrt(mat)) for Hermitian PSD mat."""
    if not _is_hermitian(mat):
        mat = 0.5 * (mat + mat.conj().T)
    w = np.linalg.eigvalsh(mat)
    w = np.clip(w, 0.0, None)
    return float(np.sum(np.sqrt(w)))

def bures_angle(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Bures (Uhlmann) angle A(rho, sigma) = arccos(sqrt(F)).

    Fidelity F = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2
    """
    rho = 0.5 * (rho + rho.conj().T)
    sigma = 0.5 * (sigma + sigma.conj().T)
    sr = _psd_sqrt(rho)
    x = sr @ sigma @ sr
    tr_sqrt = _psd_trace_sqrt(x)
    F = max(min(tr_sqrt**2, 1.0), 0.0)
    return float(np.arccos(np.sqrt(F)))

def effective_dimension(rho: np.ndarray) -> float:
    rho = 0.5 * (rho + rho.conj().T)
    tr_rho2 = float(np.real(np.trace(rho @ rho)))
    tr_rho2 = max(tr_rho2, EPS)
    return 1.0 / tr_rho2

def _entropy_bits_from_eigs(eigs: np.ndarray) -> float:
    p = np.clip(np.real(eigs), 0.0, 1.0)
    s = float(-np.sum(np.where(p > EPS, p * (np.log(p) / np.log(2)), 0.0)))
    return s

def von_neumann_entropy_bits(rho: np.ndarray) -> float:
    rho = 0.5 * (rho + rho.conj().T)
    w = np.linalg.eigvalsh(rho)
    return _entropy_bits_from_eigs(w)

def mutual_information_bits(rhoS: np.ndarray, rhoE: np.ndarray, rhoSE: np.ndarray) -> float:
    return von_neumann_entropy_bits(rhoS) + von_neumann_entropy_bits(rhoE) - von_neumann_entropy_bits(rhoSE)

def compute_Y(rhoS: np.ndarray, rhoS_prime: np.ndarray, Ibits: float | None = None) -> float:
    """Compute Y = sqrt(deff-1) * A^2 / I.

    Requires Ibits (mutual information in bits). If None, returns np.nan.
    """
    if Ibits is None or Ibits <= EPS:
        return float("nan")
    deff = effective_dimension(rhoS_prime)
    x = max(deff - 1.0, EPS)
    A = bures_angle(rhoS, rhoS_prime)
    return float(np.sqrt(x) * (A**2) / Ibits)

def partial_trace_SE(rhoSE: np.ndarray, dims: tuple[int, int], subsystem: str) -> np.ndarray:
    """Partial trace over S or E from rhoSE with dims=(dS,dE)."""
    dS, dE = dims
    rho = rhoSE.reshape(dS, dE, dS, dE)
    if subsystem.upper() == 'E':
        return np.einsum('aebi->ab', rho)
    elif subsystem.upper() == 'S':
        return np.einsum('aebi->ei', rho)
    else:
        raise ValueError("subsystem must be 'S' or 'E'")

def compute_Y_from_SE(rhoS: np.ndarray, rhoS_prime: np.ndarray, rhoSE_post: np.ndarray, dims: tuple[int, int]) -> float:
    """Compute Y using post-dilation joint state rhoSE_post to get Ibits."""
    dS, dE = dims
    rhoS_post = partial_trace_SE(rhoSE_post, dims, 'E')
    rhoE_post = partial_trace_SE(rhoSE_post, dims, 'S')
    Ibits = mutual_information_bits(rhoS_post, rhoE_post, rhoSE_post)
    return compute_Y(rhoS, rhoS_prime, Ibits)

