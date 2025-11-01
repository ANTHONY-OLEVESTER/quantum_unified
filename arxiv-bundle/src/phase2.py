# ============================================================================
# Phase-II Universality Scaling Battery
# ============================================================================
# Implements the agreed test plan:
#   * Expanded grids over unitary and CPTP dynamics
#   * Resampled slope statistics (OLS, bootstrap, Theil-Sen)
#   * Rank flatness metrics (Kendall tau, Spearman rho)
#   * Finite-size scaling exponent gamma per model
#   * Twirl-depth restoration study for amplitude damping
# Outputs CSV artifacts:
#   - universality_sweep.csv       (one row per grid point)
#   - universality_pooled.csv      (per-model pooled summary + pass/fail flags)
#   - finite_size_gamma.csv        (gamma fits per model)
# All randomness is deterministically seeded via SHA256 hashes of grid labels.
# ============================================================================

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import pickle
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from collections import defaultdict


import numpy as np

try:  # numpy >=1.20
    from numpy.core._exceptions import _ArrayMemoryError as _NPArrayMemoryError  # type: ignore
except Exception:  # pragma: no cover - fallback for older numpy
    class _NPArrayMemoryError(Exception):
        pass

from phase2_plot_utils import build_plot_cache, plot_alpha_vs_invD, plot_gamma_from_alpha, plot_phase2_results, plot_variance_scaling

try:  # optional SciPy speed-ups
    from scipy.stats import kendalltau as _scipy_kendalltau  # type: ignore
except Exception:  # pragma: no cover - optional
    _scipy_kendalltau = None

try:  # optional SciPy Theil-Sen implementation
    from scipy.stats import theilslopes as _scipy_theilslopes  # type: ignore
except Exception:  # pragma: no cover - optional
    _scipy_theilslopes = None

try:  # optional SciPy Spearman implementation
    from scipy.stats import spearmanr as _scipy_spearmanr  # type: ignore
except Exception:  # pragma: no cover - optional
    _scipy_spearmanr = None


EPS = 1e-12
PI = math.pi
GAMMA_BOOTSTRAP_SAMPLES = 5000
VARIANCE_BOOTSTRAP_SAMPLES = 3000

ALPHA_PER_SIZE_FIELDS = [
    'model', 'size_label', 'nS', 'nE', 'D', 'alpha_signed', 'abs_alpha',
]
POOLED_FIELDS = [
    'model', 'kind', 'n', 'alpha', 'intercept', 'alpha_lo', 'alpha_hi', 'R2',
    'kendall_tau', 'spearman_rho', 'slope_theil_sen', 'num_sizes',
    'gamma', 'gamma_lo', 'gamma_hi', 'log_c', 'pass_universality',
]
GAMMA_FIELDS = [
    'model', 'num_sizes', 'gamma', 'gamma_lo', 'gamma_hi', 'log_c', 'note',
]

# -------------------------- Linear-algebra helpers --------------------------

def dm(psi: np.ndarray) -> np.ndarray:
    psi = psi.reshape(-1, 1)
    return psi @ psi.conj().T


def dagger(A: np.ndarray) -> np.ndarray:
    return A.conj().T


def eigh_psd(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H = (M + dagger(M)) / 2
    w, V = np.linalg.eigh(H)
    w = np.clip(np.real(w), 0.0, None)
    return w, V


def sqrtm_psd(M: np.ndarray) -> np.ndarray:
    w, V = eigh_psd(M)
    return V @ np.diag(np.sqrt(w)) @ dagger(V)


def fidelity_uhlmann(rho: np.ndarray, sigma: np.ndarray) -> float:
    s = sqrtm_psd(rho) @ sigma @ sqrtm_psd(rho)
    return float(np.trace(sqrtm_psd(s)).real ** 2)


def vn_entropy_bits(rho: np.ndarray, tol: float = 1e-12) -> float:
    w, _ = eigh_psd(rho)
    w = w[w > tol]
    return float((-w * np.log2(w)).sum()) if w.size else 0.0


def purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))


def partial_trace(rho: np.ndarray, keep: Iterable[int], dims: List[int]) -> np.ndarray:
    dims = list(dims)
    keep = sorted(keep)
    n = len(dims)
    trace = [i for i in range(n) if i not in keep]
    perm = keep + trace + [i + n for i in keep] + [i + n for i in trace]
    resh = rho.reshape(*(dims + dims)).transpose(perm)
    d_keep = int(np.prod([dims[i] for i in keep])) if keep else 1
    d_trace = int(np.prod([dims[i] for i in trace])) if trace else 1
    resh = resh.reshape(d_keep, d_trace, d_keep, d_trace)
    return np.einsum("aibi->ab", resh)


# -------------------------- Random states & ops -----------------------------

def kron(*ops: np.ndarray) -> np.ndarray:
    out = np.array(1.0 + 0.0j)
    for op in ops:
        out = np.kron(out, op)
    return out


def haar_state(n_qudits: int, d: int = 2, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    dim = d ** n_qudits
    vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    vec /= np.linalg.norm(vec)
    return vec


def haar_unitary(dim: int, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    X = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    Q, R = np.linalg.qr(X)
    phase = np.diag(R) / np.abs(np.diag(R))
    return Q @ np.diag(phase.conj())


I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = (sx, sy, sz)


def embed_one_qubit(op: np.ndarray, idx: int, total: int) -> np.ndarray:
    ops = [I2] * total
    ops[idx] = op
    return kron(*ops)


def embed_pair(op_a: np.ndarray, idx_a: int, op_b: np.ndarray, idx_b: int, total: int) -> np.ndarray:
    if idx_a == idx_b:
        raise ValueError("pair embedding requires distinct indices")
    ops = [I2] * total
    ops[idx_a] = op_a
    ops[idx_b] = op_b
    return kron(*ops)


# -------------------------- Hamiltonians & unitaries ------------------------

def random_2body_H(nS: int, nE: int, rng: np.random.Generator) -> np.ndarray:
    total = nS + nE
    dim = 2 ** total
    H = np.zeros((dim, dim), dtype=complex)
    scale = 1.0 / math.sqrt(max(1, nS * nE))
    for i in range(nS):
        for j in range(nE):
            idx_s = i
            idx_e = nS + j
            for a in PAULIS:
                for b in PAULIS:
                    coeff = scale * rng.normal()
                    H += coeff * embed_pair(a, idx_s, b, idx_e, total)
    return H


def structured_H(model: str, nS: int, nE: int) -> np.ndarray:
    total = nS + nE
    dim = 2 ** total
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(min(nS, nE)):
        idx_s = i
        idx_e = nS + i
        if model == "dephasing":
            H += embed_pair(sz, idx_s, sz, idx_e, total)
        elif model == "pswap":
            H += 0.25 * (
                embed_pair(sx, idx_s, sx, idx_e, total)
                + embed_pair(sy, idx_s, sy, idx_e, total)
                + embed_pair(sz, idx_s, sz, idx_e, total)
            )
        else:
            raise ValueError(f"unknown structured model {model}")
    return H


def unitary_from_H(H: np.ndarray, kappa: float) -> np.ndarray:
    w, V = np.linalg.eigh((H + dagger(H)) / 2)
    phases = np.exp(-1j * kappa * w)
    return V @ np.diag(phases) @ dagger(V)


# -------------------------- Channels (CPTP) ---------------------------------

def weyl_operators(d: int) -> List[np.ndarray]:
    """Return the d^2 Weyl operators with standard unitary normalization."""
    ops: List[np.ndarray] = []
    omega = np.exp(2j * np.pi / d)
    X = np.roll(np.eye(d, dtype=complex), 1, axis=1)
    Z = np.diag([omega ** k for k in range(d)])
    for a in range(d):
        Xa = np.linalg.matrix_power(X, a)
        for b in range(d):
            Zb = np.linalg.matrix_power(Z, b)
            ops.append(Xa @ Zb)
    return ops


def kraus_depolarizing_qudit(d: int, lam: float) -> List[np.ndarray]:
    """Return Kraus operators for Φ(ρ) = λρ + (1-λ) I/d."""
    if not (0.0 <= lam <= 1.0):
        raise ValueError("lambda must be in [0,1]")
    W = weyl_operators(d)
    if len(W) != d * d:
        raise RuntimeError("Unexpected number of Weyl operators")
    K: List[np.ndarray] = [math.sqrt(lam) * np.eye(d, dtype=complex)]
    if len(W) > 1:
        scale = math.sqrt((1.0 - lam) / (len(W) - 1))
        for op in W[1:]:
            K.append(scale * op)
    return K


def kraus_amplitude_damping(p: float) -> List[np.ndarray]:
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1]")
    K0 = np.array([[1, 0], [0, math.sqrt(1 - p)]], dtype=complex)
    K1 = np.array([[0, math.sqrt(p)], [0, 0]], dtype=complex)
    return [K0, K1]


def stinespring_isometry(K: List[np.ndarray]) -> Tuple[np.ndarray, int, int]:
    dS = K[0].shape[0]
    m = len(K)
    V = np.zeros((dS * m, dS), dtype=complex)
    for idx, A in enumerate(K):
        V[idx * dS : (idx + 1) * dS, :] = A
    return V, dS, m


def apply_channel_kraus(K: List[np.ndarray], rho: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rho)
    for A in K:
        out += A @ rho @ dagger(A)
    return out


def twirl_channel_sample(K: List[np.ndarray], rho: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    d = rho.shape[0]
    U = haar_unitary(d, rng)
    rho_in = dagger(U) @ rho @ U
    rho_out = apply_channel_kraus(K, rho_in)
    return U @ rho_out @ dagger(U)


def apply_channel_kraus_twirl_depth(K: List[np.ndarray], rho: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    rho_cur = rho.copy()
    depth = max(1, m)
    for _ in range(depth):
        rho_cur = twirl_channel_sample(K, rho_cur, rng)
    return rho_cur


# -------------------------- Metrics layer ----------------------------------

def metrics_after_unitary(U: np.ndarray, psiS: np.ndarray, psiE: np.ndarray, nS: int, nE: int) -> Tuple[float, float, float]:
    dims = [2] * (nS + nE)
    psiSE0 = kron(psiS, psiE).reshape(-1)
    rhoSE0 = dm(psiSE0)
    rhoSE1 = dm(U @ psiSE0)
    rhoS0 = partial_trace(rhoSE0, keep=list(range(nS)), dims=dims)
    rhoS1 = partial_trace(rhoSE1, keep=list(range(nS)), dims=dims)
    F = fidelity_uhlmann(rhoS0, rhoS1)
    A2 = float(np.arccos(np.sqrt(np.clip(F, 0.0, 1.0))) ** 2)
    Sbits = vn_entropy_bits(rhoS1)
    Ibits = 2.0 * Sbits
    deff = 1.0 / max(purity(rhoS1), EPS)
    return Ibits, A2, deff


def metrics_after_kraus(K: List[np.ndarray], rhoS0: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    V, dS, m = stinespring_isometry(K)
    rhoSE1 = V @ rhoS0 @ dagger(V)
    rhoS1 = partial_trace(rhoSE1, keep=[0], dims=[dS, m])
    rhoE1 = partial_trace(rhoSE1, keep=[1], dims=[dS, m])
    F = fidelity_uhlmann(rhoS0, rhoS1)
    A2 = float(np.arccos(np.sqrt(np.clip(F, 0.0, 1.0))) ** 2)
    Ibits = vn_entropy_bits(rhoS1) + vn_entropy_bits(rhoE1) - vn_entropy_bits(rhoSE1)
    deff = 1.0 / max(purity(rhoS1), EPS)
    return Ibits, A2, deff, rhoS1


# -------------------------- Statistics helpers -----------------------------

def stable_hash32(*parts: object) -> int:
    data = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:4], "little")


def fit_slope_loglog(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float, float, int, np.ndarray, np.ndarray]:
    mask = (X > 0) & (Y > 0)
    X = X[mask]
    Y = Y[mask]
    if X.size < 2:
        return float("nan"), float("nan"), float("nan"), int(X.size), np.array([]), np.array([])
    lx = np.log10(X)
    ly = np.log10(Y)
    A = np.column_stack((lx, np.ones_like(lx)))
    m, b = np.linalg.lstsq(A, ly, rcond=None)[0]
    yhat = m * lx + b
    ss_res = np.sum((ly - yhat) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2) + EPS
    R2 = 1.0 - ss_res / ss_tot
    return float(m), float(b), float(R2), int(lx.size), lx, ly


def bootstrap_slope(lx: np.ndarray, ly: np.ndarray, B: int = 2000, rng: np.random.Generator | None = None) -> Tuple[float, float, float]:
    if lx.size < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng() if rng is None else rng
    n = lx.size
    slopes = np.empty(B)
    ones = np.ones(n)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        A = np.column_stack((lx[idx], ones))
        slopes[b], _ = np.linalg.lstsq(A, ly[idx], rcond=None)[0]
    mean = float(slopes.mean())
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return mean, float(lo), float(hi)


def kendall_tau(lx: np.ndarray, ly: np.ndarray) -> float:
    if lx.size < 2:
        return 0.0
    if _scipy_kendalltau is not None:
        return float(_scipy_kendalltau(lx, ly, variant="b")[0])
    n = lx.size
    concordant = discordant = 0
    for i in range(n):
        dx = lx[i + 1:] - lx[i]
        dy = ly[i + 1:] - ly[i]
        prod = dx * dy
        concordant += int(np.sum(prod > 0))
        discordant += int(np.sum(prod < 0))
    denom = concordant + discordant
    return 0.0 if denom == 0 else float((concordant - discordant) / denom)


def spearman_rho(lx: np.ndarray, ly: np.ndarray) -> float:
    if lx.size < 2:
        return 0.0
    if _scipy_spearmanr is not None:
        return float(_scipy_spearmanr(lx, ly)[0])
    rx = np.argsort(np.argsort(lx)).astype(float)
    ry = np.argsort(np.argsort(ly)).astype(float)
    rx = (rx - rx.mean()) / (rx.std() + EPS)
    ry = (ry - ry.mean()) / (ry.std() + EPS)
    return float(np.mean(rx * ry))


def theil_sen_slope(lx: np.ndarray, ly: np.ndarray) -> float:
    if lx.size < 2:
        return 0.0
    if _scipy_theilslopes is not None:
        try:
            slope, _, _, _ = _scipy_theilslopes(ly, lx)
            return float(slope)
        except (MemoryError, _NPArrayMemoryError):
            pass  # fall back to manual computation for large samples
    slopes: List[float] = []
    n = lx.size
    for i in range(n):
        dx = lx[i + 1:] - lx[i]
        dy = ly[i + 1:] - ly[i]
        mask = np.abs(dx) > 1e-15
        slopes.extend((dy[mask] / dx[mask]).tolist())
    return float(np.median(slopes)) if slopes else 0.0


# -------------------------- Data collection ---------------------------------

def build_unitary(model: str, nS: int, nE: int, kappa: float, rng: np.random.Generator) -> np.ndarray:
    if model == "random2body":
        H = random_2body_H(nS, nE, rng)
    elif model in {"pswap", "dephasing"}:
        H = structured_H(model, nS, nE)
    else:
        raise ValueError(f"unknown unitary model {model}")
    return unitary_from_H(H, kappa)


def collect_unitary(model: str, nS: int, nE: int, kappa: float, n_trials: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    U = build_unitary(model, nS, nE, kappa, rng)
    X: List[float] = []
    Y: List[float] = []
    for _ in range(n_trials):
        psiS = haar_state(nS, d=2, rng=rng)
        psiE = haar_state(nE, d=2, rng=rng)
        Ibits, A2, deff = metrics_after_unitary(U, psiS, psiE, nS, nE)
        if Ibits <= EPS:
            continue
        x = max(deff - 1.0, EPS)
        y = math.sqrt(x) * A2 / Ibits
        X.append(x)
        Y.append(y)
    return np.asarray(X), np.asarray(Y)


def collect_depolarizing(d: int, lam: float, n_trials: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    K = kraus_depolarizing_qudit(d, lam)
    X: List[float] = []
    Y: List[float] = []
    for _ in range(n_trials):
        psi = haar_state(1, d=d, rng=rng)
        rho0 = dm(psi)
        Ibits, A2, deff, _ = metrics_after_kraus(K, rho0)
        if Ibits <= EPS:
            continue
        x = max(deff - 1.0, EPS)
        y = math.sqrt(x) * A2 / Ibits
        X.append(x)
        Y.append(y)
    return np.asarray(X), np.asarray(Y)


def collect_amp_damp(p: float, twirl_depth: int, n_trials: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    K = kraus_amplitude_damping(p)
    X: List[float] = []
    Y: List[float] = []
    for _ in range(n_trials):
        psi = haar_state(1, d=2, rng=rng)
        rho0 = dm(psi)
        Ibits, _, _, _ = metrics_after_kraus(K, rho0)
        if Ibits <= EPS:
            continue
        if twirl_depth > 0:
            rho1 = apply_channel_kraus_twirl_depth(K, rho0, twirl_depth, rng)
            F = fidelity_uhlmann(rho0, rho1)
            A2 = float(np.arccos(np.sqrt(np.clip(F, 0.0, 1.0))) ** 2)
            deff = 1.0 / max(purity(rho1), EPS)
        else:
            _, A2, deff, _ = metrics_after_kraus(K, rho0)
        x = max(deff - 1.0, EPS)
        y = math.sqrt(x) * A2 / Ibits
        X.append(x)
        Y.append(y)
    return np.asarray(X), np.asarray(Y)


# -------------------------- Finite-size scaling -----------------------------

def fit_gamma(records: List[Tuple[float, float]]):
    filtered = [(float(D), float(val)) for D, val in records if D > 0 and val > 0 and np.isfinite(D) and np.isfinite(val)]
    if len(filtered) < 2:
        return float("nan"), float("nan")
    Ds = np.array([pair[0] for pair in filtered], dtype=float)
    As = np.maximum(np.array([pair[1] for pair in filtered], dtype=float), EPS)
    mask = (Ds > 0) & (As > 0)
    Ds = Ds[mask]
    As = As[mask]
    if np.unique(Ds).size < 2:
        return float("nan"), float("nan")
    Lx = np.log10(Ds)
    Ly = np.log10(As)
    A = np.column_stack((-Lx, np.ones_like(Lx)))
    gamma, logc = np.linalg.lstsq(A, Ly, rcond=None)[0]
    return float(gamma), float(logc)


def bootstrap_gamma(records: List[Tuple[float, float]], B: int, rng: np.random.Generator) -> Tuple[float, float, float, float]:
    filtered = [(float(D), float(val)) for D, val in records if D > 0 and val > 0 and np.isfinite(D) and np.isfinite(val)]
    if len(filtered) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    Ds = np.array([pair[0] for pair in filtered], dtype=float)
    As = np.maximum(np.array([pair[1] for pair in filtered], dtype=float), EPS)
    mask = (Ds > 0) & (As > 0)
    Ds = Ds[mask]
    As = As[mask]
    if np.unique(Ds).size < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    Lx = np.log10(Ds)
    Ly = np.log10(As)
    idx_range = np.arange(len(Lx))
    gammas: List[float] = []
    logcs: List[float] = []
    for _ in range(B):
        idx = rng.choice(idx_range, size=len(idx_range), replace=True)
        sample_x = Lx[idx]
        sample_y = Ly[idx]
        if np.unique(sample_x).size < 2:
            continue
        A = np.column_stack((-sample_x, np.ones_like(sample_x)))
        gamma, logc = np.linalg.lstsq(A, sample_y, rcond=None)[0]
        gammas.append(float(gamma))
        logcs.append(float(logc))
    if not gammas:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(gammas))
    lo, hi = np.percentile(gammas, [2.5, 97.5])
    logc_mean = float(np.mean(logcs)) if logcs else float("nan")
    return mean, float(lo), float(hi), logc_mean




def concat_samples(arrays: Iterable[np.ndarray]) -> np.ndarray:
    gathered: List[np.ndarray] = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        if arr.size:
            gathered.append(arr)
    if not gathered:
        return np.array([], dtype=float)
    return np.concatenate(gathered)


def parse_size_label(size_label: str) -> Tuple[int, int]:
    try:
        parts = size_label.split(',')
        nS = int(parts[0].split('=')[1])
        nE = int(parts[1].split('=')[1])
    except (IndexError, ValueError):
        return -1, -1
    return nS, nE


def aggregate_alpha_per_size(size_data: Mapping[str, Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model, sizes in size_data.items():
        for size_label, payload in sizes.items():
            X = concat_samples(payload.get('X', []))
            Y = concat_samples(payload.get('Y', []))
            if X.size < 2 or Y.size < 2:
                continue
            alpha, intercept, R2, n, _, _ = fit_slope_loglog(X, Y)
            if not np.isfinite(alpha):
                continue
            D_value = float(payload.get('D', float('nan')))
            if not np.isfinite(D_value):
                continue
            nS = payload.get('nS')
            nE = payload.get('nE')
            if nS is None or nE is None:
                nS, nE = parse_size_label(size_label)
            row = {
                'model': model,
                'size_label': size_label,
                'nS': int(nS) if nS is not None else -1,
                'nE': int(nE) if nE is not None else -1,
                'D': D_value,
                'alpha_signed': float(alpha),
                'abs_alpha': float(abs(alpha)),
            }
            rows.append(row)
    rows.sort(key=lambda item: (item['model'], item['D']))
    return rows


def compute_gamma_statistics(
    per_model_records: Mapping[str, List[Tuple[float, float]]],
    seed_tag: str,
) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for model, records in per_model_records.items():
        unique_D = {float(pair[0]) for pair in records if float(pair[0]) > 0}
        num_sizes = len(unique_D)
        gamma_point, logc_point = fit_gamma(records)
        boot_rng = np.random.default_rng(stable_hash32(model, seed_tag))
        gamma_mean, gamma_lo, gamma_hi, logc_mean = bootstrap_gamma(records, GAMMA_BOOTSTRAP_SAMPLES, boot_rng)
        if np.isfinite(gamma_mean):
            gamma_point = gamma_mean
            if np.isfinite(logc_mean):
                logc_point = logc_mean
        summary[model] = {
            'num_sizes': num_sizes,
            'gamma': gamma_point,
            'gamma_lo': gamma_lo,
            'gamma_hi': gamma_hi,
            'log_c': logc_point,
        }
    return summary


def bootstrap_variance_slope(
    d_map: Mapping[float, Sequence[float]],
    Ds_sorted: np.ndarray,
    B: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Bootstrap the slope of log10 Var(Y) vs log10 D."""
    if Ds_sorted.size < 2:
        return float("nan"), float("nan"), float("nan")
    slopes: List[float] = []
    for _ in range(B):
        var_boot: List[float] = []
        valid = True
        for D in Ds_sorted:
            samples = np.asarray(d_map[float(D)], dtype=float)
            if samples.size < 2:
                valid = False
                break
            idx = rng.integers(0, samples.size, size=samples.size)
            boot_samples = samples[idx]
            variance = np.var(boot_samples, ddof=1)
            if variance <= 0 or not np.isfinite(variance):
                valid = False
                break
            var_boot.append(float(max(variance, EPS)))
        if not valid:
            continue
        slope, _, _, _, _, _ = fit_slope_loglog(Ds_sorted, np.array(var_boot, dtype=float))
        if np.isfinite(slope):
            slopes.append(float(slope))
    if not slopes:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(slopes))
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return mean, float(lo), float(hi)


def compute_variance_summary(var_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in var_rows:
        try:
            model = row.get('model', '')
            D = float(row.get('D', float('nan')))
            y_val = float(row.get('y_value', float('nan')))
        except (TypeError, ValueError):
            continue
        if not (model and np.isfinite(D) and np.isfinite(y_val) and D > 0):
            continue
        grouped[model][D].append(float(y_val))

    summary: Dict[str, Dict[str, Any]] = {}
    for model, d_map in grouped.items():
        Ds: List[float] = []
        variances: List[float] = []
        for D, values in d_map.items():
            arr = np.asarray(values, dtype=float)
            if arr.size < 2:
                continue
            Ds.append(float(D))
            variances.append(float(np.var(arr, ddof=1)))
        if len(Ds) < 2:
            continue
        Ds_array = np.array(Ds, dtype=float)
        var_array = np.maximum(np.array(variances, dtype=float), 1e-12)
        slope, _, _, _, _, _ = fit_slope_loglog(Ds_array, var_array)
        if not np.isfinite(slope):
            continue
        boot_rng = np.random.default_rng(stable_hash32(model, 'var_slope'))
        slope_mean, slope_lo, slope_hi = bootstrap_variance_slope(
            grouped[model],
            Ds_array,
            VARIANCE_BOOTSTRAP_SAMPLES,
            boot_rng,
        )
        slope_point = slope_mean if np.isfinite(slope_mean) else slope
        summary[model] = {
            'num_sizes': len(Ds),
            'slope': float(slope_point),
            'slope_lo': float(slope_lo),
            'slope_hi': float(slope_hi),
        }
    return summary


def passes_finite_size(stats: Mapping[str, Any]) -> bool:
    num_sizes = int(stats.get('num_sizes', 0))
    gamma = float(stats.get('gamma', float('nan')))
    gamma_lo = float(stats.get('gamma_lo', float('nan')))
    return (
        num_sizes >= 6
        and np.isfinite(gamma)
        and np.isfinite(gamma_lo)
        and gamma >= 0.8
        and gamma_lo >= 0.5
    )
@dataclass
class GridResult:
    kind: str
    model: str
    dim: str
    param: str
    seed_tag: str
    n: int
    alpha: float
    intercept: float
    alpha_lo: float
    alpha_hi: float
    R2: float
    tau: float
    rho: float
    theil_sen: float
    grid_label: str
    size_label: str


# -------------------------- Main sweep --------------------------------------

def run_phase2(
    output_dir: Path | str = ".",
    cache_path: Path | str | None = None,
    max_qubits: int = 6,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_qubits = max(2, int(max_qubits))
    if cache_path is None:
        cache_path = output_dir / "phase2_plot_cache.pkl"
    else:
        cache_path = Path(cache_path)

    unitary_models = ['random2body', 'pswap', 'dephasing']
    base_sizes = [
        (1, 1), (1, 2), (2, 1), (2, 2),
        (3, 1), (1, 3), (3, 2), (2, 3),
        (3, 3), (4, 2), (2, 4),
        (4, 3), (3, 4),
        (4, 4), (5, 3), (3, 5),
        (5, 4), (4, 5),
    ]
    unitary_sizes = sorted(
        {size for size in base_sizes if sum(size) <= max_qubits},
        key=lambda pair: (pair[0] + pair[1], pair[0], pair[1]),
    )
    unitary_kappas = [0.40, 0.60, 0.80]
    unitary_trials = 600

    dep_dims = [2, 3, 5, 7]
    dep_lams = [0.70, 0.90]
    dep_trials = 2000

    amp_ps = [0.10, 0.30, 0.50, 0.80]
    amp_twirl_depths = [0, 1, 3, 5]
    amp_trials = 3000

    rows: List[GridResult] = []
    pooled_data: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(lambda: {'X': [], 'Y': [], 'kind': ''})
    size_data: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(dict)
    var_y_rows: List[Dict[str, Any]] = []

    def record(
        model_key: str,
        kind: str,
        X: np.ndarray,
        Y: np.ndarray,
        size_label: str,
        D_value: float | None,
        *,
        nS: int | None = None,
        nE: int | None = None,
    ) -> None:
        if model_key not in pooled_data:
            pooled_data[model_key]['kind'] = kind
        pooled_data[model_key]['X'].append(X)
        pooled_data[model_key]['Y'].append(Y)
        if kind == 'unitary' and D_value is not None:
            slot = size_data[model_key].setdefault(
                size_label,
                {'X': [], 'Y': [], 'D': D_value, 'nS': nS, 'nE': nE},
            )
            slot['X'].append(X)
            slot['Y'].append(Y)
            if nS is not None:
                slot['nS'] = nS
            if nE is not None:
                slot['nE'] = nE

    # Unitary grid
    for model in unitary_models:
        for (nS, nE) in unitary_sizes:
            size_label = f'nS={nS},nE={nE}'
            D_value = float(2 ** (nS + nE))
            for kappa in unitary_kappas:
                seed = stable_hash32('unitary', model, nS, nE, kappa)
                X, Y = collect_unitary(model, nS, nE, kappa, unitary_trials, seed)
                for y_val in Y:
                    var_y_rows.append({'kind': 'unitary', 'model': model, 'size_label': size_label, 'nS': nS, 'nE': nE, 'D': D_value, 'y_value': float(y_val)})
                a, b, R2, n, lx, ly = fit_slope_loglog(X, Y)
                boot_rng = np.random.default_rng(seed ^ 0xA5A5A5A5)
                amean, alo, ahi = bootstrap_slope(lx, ly, B=2000, rng=boot_rng)
                tau = kendall_tau(lx, ly)
                rho = spearman_rho(lx, ly)
                ts = theil_sen_slope(lx, ly)
                grid_label = f'kappa={kappa:.2f}'
                row = GridResult(
                    kind='unitary',
                    model=model,
                    dim='d=2',
                    param=f'kappa={kappa:.2f}',
                    seed_tag=f'0x{seed:08x}',
                    n=n,
                    alpha=a,
                    intercept=b,
                    alpha_lo=alo,
                    alpha_hi=ahi,
                    R2=R2,
                    tau=tau,
                    rho=rho,
                    theil_sen=ts,
                    grid_label=grid_label,
                    size_label=size_label,
                )
                rows.append(row)
                record(model, 'unitary', X, Y, size_label, D_value, nS=nS, nE=nE)

    # Depolarizing grid
    for d in dep_dims:
        for lam in dep_lams:
            seed = stable_hash32('depolarizing', d, lam)
            X, Y = collect_depolarizing(d, lam, dep_trials, seed)
            for y_val in Y:
                var_y_rows.append({'kind': 'channel', 'model': 'depolarizing', 'size_label': f'd={d}', 'nS': -1, 'nE': -1, 'D': float(d), 'y_value': float(y_val)})
            a, b, R2, n, lx, ly = fit_slope_loglog(X, Y)
            boot_rng = np.random.default_rng(seed ^ 0x5C5C5C5C)
            amean, alo, ahi = bootstrap_slope(lx, ly, B=2000, rng=boot_rng)
            tau = kendall_tau(lx, ly)
            rho = spearman_rho(lx, ly)
            ts = theil_sen_slope(lx, ly)
            row = GridResult(
                kind='channel',
                model='depolarizing',
                dim=f'd={d}',
                param=f'lambda={lam:.2f}',
                seed_tag=f'0x{seed:08x}',
                n=n,
                alpha=a,
                intercept=b,
                alpha_lo=alo,
                alpha_hi=ahi,
                R2=R2,
                tau=tau,
                rho=rho,
                theil_sen=ts,
                grid_label=f'lambda={lam:.2f}',
                size_label=f'd={d}',
            )
            rows.append(row)
            record('depolarizing', 'channel', X, Y, size_label=f'd={d}', D_value=float(d))

    # Amplitude damping grid
    for p in amp_ps:
        for depth in amp_twirl_depths:
            seed = stable_hash32('amp_damp', p, depth)
            model_key = f'amp_damp_twirl{depth}'
            X, Y = collect_amp_damp(p, depth, amp_trials, seed)
            for y_val in Y:
                var_y_rows.append({
                    'kind': 'channel',
                    'model': model_key,
                    'size_label': f'p={p:.2f}|twirl={depth}',
                    'nS': -1,
                    'nE': -1,
                    'D': float(2),
                    'y_value': float(y_val),
                })
            a, b, R2, n, lx, ly = fit_slope_loglog(X, Y)
            boot_rng = np.random.default_rng(seed ^ 0x3F3F3F3F)
            amean, alo, ahi = bootstrap_slope(lx, ly, B=2000, rng=boot_rng)
            tau = kendall_tau(lx, ly)
            rho = spearman_rho(lx, ly)
            ts = theil_sen_slope(lx, ly)
            row = GridResult(
                kind='channel',
                model=model_key,
                dim='d=2',
                param=f'p={p:.2f}',
                seed_tag=f'0x{seed:08x}',
                n=n,
                alpha=a,
                intercept=b,
                alpha_lo=alo,
                alpha_hi=ahi,
                R2=R2,
                tau=tau,
                rho=rho,
                theil_sen=ts,
                grid_label=f'p={p:.2f}|twirl={depth}',
                size_label='d=2',
            )
            rows.append(row)
            record(model_key, 'channel', X, Y, size_label='d=2', D_value=None)

    alpha_per_size_rows = aggregate_alpha_per_size(size_data)
    alpha_csv_path = output_dir / 'finite_size_alpha_per_size.csv'
    with alpha_csv_path.open('w', newline='') as f_alpha:
        alpha_writer = csv.DictWriter(
            f_alpha,
            fieldnames=ALPHA_PER_SIZE_FIELDS,
        )
        alpha_writer.writeheader()
        for row in alpha_per_size_rows:
            alpha_writer.writerow(row)

    var_csv_path = output_dir / 'phase3_varY_by_D.csv'
    with var_csv_path.open('w', newline='') as f_var:
        var_writer = csv.DictWriter(
            f_var,
            fieldnames=['kind', 'model', 'size_label', 'nS', 'nE', 'D', 'y_value'],
        )
        var_writer.writeheader()
        for row in var_y_rows:
            var_writer.writerow(row)

    per_model_records: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row in alpha_per_size_rows:
        D_value = float(row['D'])
        abs_alpha = float(row['abs_alpha'])
        if np.isfinite(D_value) and np.isfinite(abs_alpha) and D_value > 0 and abs_alpha > 0:
            per_model_records[row['model']].append((D_value, abs_alpha))

    gamma_summary = compute_gamma_statistics(per_model_records, 'gamma_boot')

    pooled_rows: List[Dict[str, object]] = []
    gamma_rows: List[Dict[str, object]] = []

    all_models = list(pooled_data.keys())
    for model in all_models:
        data = pooled_data[model]
        X = np.concatenate(data['X']) if data['X'] else np.array([])
        Y = np.concatenate(data['Y']) if data['Y'] else np.array([])
        a, b, R2, n, lx, ly = fit_slope_loglog(X, Y)
        boot_rng = np.random.default_rng(stable_hash32(model, 'pooled'))
        amean, alo, ahi = bootstrap_slope(lx, ly, B=2000, rng=boot_rng)
        tau = kendall_tau(lx, ly)
        rho = spearman_rho(lx, ly)
        ts = theil_sen_slope(lx, ly)

        info = gamma_summary.get(model, {
            'num_sizes': len({float(rec[0]) for rec in per_model_records.get(model, [])}),
            'gamma': float('nan'),
            'gamma_lo': float('nan'),
            'gamma_hi': float('nan'),
            'log_c': float('nan'),
        })
        gamma_point = info['gamma']
        gamma_lo = info['gamma_lo']
        gamma_hi = info['gamma_hi']
        logc_point = info['log_c']
        num_sizes = info['num_sizes']

        gamma_rows.append({
            'model': model,
            'num_sizes': num_sizes,
            'gamma': gamma_point,
            'gamma_lo': gamma_lo,
            'gamma_hi': gamma_hi,
            'log_c': logc_point,
            'note': 'fit on |alpha|',
        })

        pass_flag = passes_finite_size(info)

        pooled_rows.append({
            'model': model,
            'kind': data['kind'],
            'n': n,
            'alpha': a,
            'intercept': b,
            'alpha_lo': alo,
            'alpha_hi': ahi,
            'R2': R2,
            'kendall_tau': tau,
            'spearman_rho': rho,
            'slope_theil_sen': ts,
            'log_c': logc_point,
            'gamma': gamma_point,
            'gamma_lo': gamma_lo,
            'gamma_hi': gamma_hi,
            'num_sizes': num_sizes,
            'pass_universality': bool(pass_flag),
        })

    fieldnames = [
        'kind', 'model', 'dim', 'param', 'seed_tag', 'n',
        'alpha', 'intercept', 'alpha_lo', 'alpha_hi', 'R2',
        'kendall_tau', 'spearman_rho', 'slope_theil_sen',
        'grid_label', 'size_label'
    ]
    sweep_path = output_dir / 'universality_sweep.csv'
    with sweep_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                'kind': row.kind,
                'model': row.model,
                'dim': row.dim,
                'param': row.param,
                'seed_tag': row.seed_tag,
                'n': row.n,
                'alpha': row.alpha,
                'intercept': row.intercept,
                'alpha_lo': row.alpha_lo,
                'alpha_hi': row.alpha_hi,
                'R2': row.R2,
                'kendall_tau': row.tau,
                'spearman_rho': row.rho,
                'slope_theil_sen': row.theil_sen,
                'grid_label': row.grid_label,
                'size_label': row.size_label,
            })

    pooled_fields = POOLED_FIELDS
    pooled_path = output_dir / 'universality_pooled.csv'
    with pooled_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=pooled_fields)
        writer.writeheader()
        for row in pooled_rows:
            writer.writerow(row)

    gamma_fields = GAMMA_FIELDS
    gamma_path = output_dir / 'finite_size_gamma.csv'
    with gamma_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=gamma_fields)
        writer.writeheader()
        for row in gamma_rows:
            writer.writerow(row)

    plot_cache = build_plot_cache(
        pooled_data,
        size_data,
        rows,
        pooled_rows,
        alpha_per_size=alpha_per_size_rows,
        gamma_rows=gamma_rows,
        var_y_samples=var_y_rows,
    )
    with cache_path.open('wb') as fh:
        pickle.dump(plot_cache, fh)
    saved_figs = plot_phase2_results(plot_cache, output_dir)

    print('Phase-II universality sweep completed.')
    print(f"Grid rows: {len(rows)}, pooled models: {len(pooled_rows)}")
    if saved_figs:
        pngs = [path.name for path in saved_figs if path.suffix == '.png']
        if pngs:
            print("Saved figures:", ", ".join(pngs))
    print(f"CSV outputs: {sweep_path}, {pooled_path}, {alpha_csv_path}, {gamma_path}, {var_csv_path}")
    print(f"Plot cache written to {cache_path.resolve()}")
    var_summary = compute_variance_summary(var_y_rows)
    print("\nVariance scaling summary:")
    for model_name in ('random2body', 'depolarizing'):
        info = var_summary.get(model_name)
        if info:
            slope = info.get('slope', float('nan'))
            slope_lo = info.get('slope_lo', float('nan'))
            slope_hi = info.get('slope_hi', float('nan'))
            if np.isfinite(slope):
                if np.isfinite(slope_lo) and np.isfinite(slope_hi):
                    window = f"{slope:+.3f} [{slope_lo:+.3f},{slope_hi:+.3f}]"
                else:
                    window = f"{slope:+.3f}"
            else:
                window = "nan"
            print(f"  Var[Y] slope ({model_name}): {window} over {info['num_sizes']} dims")
    print("\nFinite-size gamma summary (|alpha| fits):")
    for entry in gamma_rows:
        model = entry['model']
        num_sizes = entry['num_sizes']
        gamma_val = entry['gamma']
        gamma_lo = entry['gamma_lo']
        gamma_hi = entry['gamma_hi']
        status = "PASS" if passes_finite_size(entry) else "FAIL"
        if np.isfinite(gamma_val):
            summary_str = f"{gamma_val:+.3f} [{gamma_lo:+.3f},{gamma_hi:+.3f}]"
        else:
            summary_str = "nan"
        print(f"  {model:14s} D-count={num_sizes:2d}  gamma={summary_str}  -> {status}")

    return {
        "rows": rows,
        "pooled_rows": pooled_rows,
        "gamma_rows": gamma_rows,
        "alpha_per_size_rows": alpha_per_size_rows,
        "pooled_data": pooled_data,
        "size_data": size_data,
        "saved_figs": saved_figs,
        "cache_path": cache_path,
        "output_dir": output_dir,
    }


def run_gamma_only(output_dir: Path | str, cache_path: Path | str | None = None) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha_csv_path = output_dir / 'finite_size_alpha_per_size.csv'
    if not alpha_csv_path.exists():
        raise FileNotFoundError(f"Per-size CSV '{alpha_csv_path}' not found. Run with --sweep first.")

    alpha_rows: List[Dict[str, Any]] = []
    with alpha_csv_path.open('r', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                entry = {
                    'model': row.get('model', ''),
                    'size_label': row.get('size_label', ''),
                    'nS': int(float(row.get('nS', -1))),
                    'nE': int(float(row.get('nE', -1))),
                    'D': float(row.get('D', float('nan'))),
                    'alpha_signed': float(row.get('alpha_signed', float('nan'))),
                    'abs_alpha': float(row.get('abs_alpha', float('nan'))),
                }
            except (TypeError, ValueError):
                continue
            alpha_rows.append(entry)

    var_rows: List[Dict[str, Any]] = []
    var_csv_path = output_dir / 'phase3_varY_by_D.csv'
    if var_csv_path.exists():
        with var_csv_path.open('r', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    entry = {
                        'kind': row.get('kind', ''),
                        'model': row.get('model', ''),
                        'size_label': row.get('size_label', ''),
                        'nS': int(float(row.get('nS', -1))),
                        'nE': int(float(row.get('nE', -1))),
                        'D': float(row.get('D', float('nan'))),
                        'y_value': float(row.get('y_value', float('nan'))),
                    }
                except (TypeError, ValueError):
                    continue
                var_rows.append(entry)

    per_model_records: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for entry in alpha_rows:
        D_value = entry['D']
        abs_alpha = entry['abs_alpha']
        if np.isfinite(D_value) and np.isfinite(abs_alpha) and D_value > 0 and abs_alpha > 0:
            per_model_records[entry['model']].append((D_value, abs_alpha))

    gamma_summary = compute_gamma_statistics(per_model_records, 'gamma_only_boot')
    var_summary = compute_variance_summary(var_rows)

    pooled_path = output_dir / 'universality_pooled.csv'
    pooled_rows_data: List[Dict[str, Any]] = []
    if pooled_path.exists():
        with pooled_path.open('r', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                pooled_rows_data.append(dict(row))

    model_order: List[str] = [row.get('model', '') for row in pooled_rows_data] if pooled_rows_data else []
    for model in gamma_summary.keys():
        if model not in model_order:
            model_order.append(model)

    gamma_rows: List[Dict[str, Any]] = []
    for model in model_order:
        info = gamma_summary.get(
            model,
            {
                'num_sizes': len({float(rec[0]) for rec in per_model_records.get(model, [])}),
                'gamma': float('nan'),
                'gamma_lo': float('nan'),
                'gamma_hi': float('nan'),
                'log_c': float('nan'),
            },
        )
        gamma_rows.append({
            'model': model,
            'num_sizes': info['num_sizes'],
            'gamma': info['gamma'],
            'gamma_lo': info['gamma_lo'],
            'gamma_hi': info['gamma_hi'],
            'log_c': info['log_c'],
            'note': 'fit on |alpha|',
        })

    gamma_path = output_dir / 'finite_size_gamma.csv'
    with gamma_path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=GAMMA_FIELDS)
        writer.writeheader()
        for row in gamma_rows:
            writer.writerow(row)

    updated_pooled: List[Dict[str, Any]] = []
    if pooled_rows_data:
        for row in pooled_rows_data:
            model = row.get('model', '')
            info = gamma_summary.get(
                model,
                {
                    'num_sizes': len({float(rec[0]) for rec in per_model_records.get(model, [])}),
                    'gamma': float('nan'),
                    'gamma_lo': float('nan'),
                    'gamma_hi': float('nan'),
                    'log_c': float('nan'),
                },
            )
            row.update({
                'num_sizes': info['num_sizes'],
                'gamma': info['gamma'],
                'gamma_lo': info['gamma_lo'],
                'gamma_hi': info['gamma_hi'],
                'log_c': info['log_c'],
                'pass_universality': passes_finite_size(info),
            })
            updated_pooled.append(row)
        with pooled_path.open('w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=POOLED_FIELDS)
            writer.writeheader()
            for row in updated_pooled:
                writer.writerow(row)

    gamma_stats_map = {row['model']: row for row in gamma_rows}
    saved_figs: List[Path] = []
    saved_figs.extend(plot_alpha_vs_invD(alpha_rows, output_dir))
    saved_figs.extend(plot_gamma_from_alpha(alpha_rows, gamma_stats_map, output_dir))
    saved_figs.extend(plot_variance_scaling(var_rows, output_dir))

    cache_path = Path(cache_path) if cache_path is not None else output_dir / 'phase2_plot_cache.pkl'
    cache: Dict[str, Any] = {}
    if cache_path.exists():
        try:
            with cache_path.open('rb') as fh:
                cache = pickle.load(fh)
        except Exception:
            cache = {}
    cache['alpha_per_size'] = alpha_rows
    cache['gamma_rows'] = gamma_rows
    if var_rows:
        cache['var_y_by_D'] = var_rows
    if 'pooled_rows' not in cache and updated_pooled:
        cache['pooled_rows'] = updated_pooled
    with cache_path.open('wb') as fh:
        pickle.dump(cache, fh)

    print('Gamma-only refresh complete.')
    print(f"Updated gamma CSV written to {gamma_path}")
    if saved_figs:
        pngs = [path.name for path in saved_figs if path.suffix == '.png']
        if pngs:
            print("Regenerated figures:", ", ".join(pngs))
    if var_summary:
        print("\nVariance scaling summary:")
        for model_name in ('random2body', 'depolarizing'):
            info = var_summary.get(model_name)
            if not info:
                continue
            slope = info.get('slope', float('nan'))
            slope_lo = info.get('slope_lo', float('nan'))
            slope_hi = info.get('slope_hi', float('nan'))
            if np.isfinite(slope):
                if np.isfinite(slope_lo) and np.isfinite(slope_hi):
                    window = f"{slope:+.3f} [{slope_lo:+.3f},{slope_hi:+.3f}]"
                else:
                    window = f"{slope:+.3f}"
            else:
                window = "nan"
            print(f"  Var[Y] slope ({model_name}): {window} over {info['num_sizes']} dims")
    print("\nFinite-size gamma summary (|alpha| fits):")
    for row in gamma_rows:
        status = "PASS" if passes_finite_size(row) else "FAIL"
        gamma_val = row['gamma']
        gamma_lo = row['gamma_lo']
        gamma_hi = row['gamma_hi']
        if np.isfinite(gamma_val):
            summary = f"{gamma_val:+.3f} [{gamma_lo:+.3f},{gamma_hi:+.3f}]"
        else:
            summary = "nan"
        print(f"  {row['model']:14s} D-count={row['num_sizes']:2d}  gamma={summary}  -> {status}")

    return {
        'alpha_rows': alpha_rows,
        'gamma_rows': gamma_rows,
        'saved_figs': saved_figs,
        'cache_path': cache_path,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase-II universality utilities.")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run the full Phase-II sweep (expensive).",
    )
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=6,
        help="Maximum total qubits (nS+nE) to include for unitary models (default: 6).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for CSVs, cache, and figures (default: current directory).",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="Path to plot cache (default: <output-dir>/phase2_plot_cache.pkl).",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Only regenerate figures from an existing cache.",
    )
    parser.add_argument(
        "--gamma-only",
        action="store_true",
        help="Recompute finite-size gamma and plots from existing CSVs (no new simulations).",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    cache_path = Path(args.cache) if args.cache else output_dir / "phase2_plot_cache.pkl"

    actions = sum(int(flag) for flag in (args.sweep, args.plots_only, args.gamma_only))
    if actions > 1:
        parser.error("Specify at most one of --sweep, --plots-only, or --gamma-only.")

    if args.sweep:
        run_phase2(output_dir=output_dir, cache_path=cache_path, max_qubits=args.max_qubits)
        return

    if args.gamma_only:
        run_gamma_only(output_dir=output_dir, cache_path=cache_path)
        return

    if args.plots_only:
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Plot cache '{cache_path}' not found. Run with --sweep first."
            )
        with cache_path.open('rb') as fh:
            plot_cache = pickle.load(fh)
        saved_figs = plot_phase2_results(plot_cache, output_dir)
        if saved_figs:
            pngs = [path.name for path in saved_figs if path.suffix == '.png']
            if pngs:
                print("Saved figures:", ", ".join(pngs))
        return

    parser.print_help()


if __name__ == '__main__':
    main()
