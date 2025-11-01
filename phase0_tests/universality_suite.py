#!/usr/bin/env python3
"""Numerical universality checks for the collapse invariant.

This script samples several dynamical scenarios and records how the quantity

    Y = sqrt(d_eff - 1) * (A^2 / I)

behaves.  We contrast chaotic two-body dynamics, an integrable XXZ chain,
an amplitude-damping (dissipative) channel, and a measurement-induced
dephasing channel.  Results are written to
`analysis/universality_results.json`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import numpy.linalg as LA
from numpy.random import Generator, default_rng
from scipy.linalg import expm

# ---------------------------------------------------------------------------
# Basic linear-algebra helpers
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)


def dagger(mat: np.ndarray) -> np.ndarray:
    return mat.conj().T


def dm(psi: np.ndarray) -> np.ndarray:
    """Projector |psi><psi| for a pure state vector."""
    psi = psi.reshape(-1, 1)
    return psi @ dagger(psi)


def haar_state(n_qubits: int, rng: Generator) -> np.ndarray:
    dim = 2**n_qubits
    vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    vec /= LA.norm(vec)
    return vec


def partial_trace(rho: np.ndarray, dims: Iterable[int], keep: Iterable[int]) -> np.ndarray:
    """Trace out all subsystems except those listed in `keep`."""
    dims = list(dims)
    keep = sorted(keep)
    n = len(dims)
    trace_over = [i for i in range(n) if i not in keep]

    perm = keep + trace_over + [i + n for i in keep] + [i + n for i in trace_over]
    reshaped = rho.reshape(*(dims + dims)).transpose(perm)

    d_keep = int(np.prod([dims[i] for i in keep])) if keep else 1
    d_trace = int(np.prod([dims[i] for i in trace_over])) if trace_over else 1
    reshaped = reshaped.reshape(d_keep, d_trace, d_keep, d_trace)
    return np.einsum("aibi->ab", reshaped)


def von_neumann_entropy_bits(rho: np.ndarray, tol: float = 1e-12) -> float:
    evals = np.real_if_close(LA.eigvalsh((rho + dagger(rho)) / 2))
    evals = np.clip(evals, 0.0, 1.0)
    nz = evals[evals > tol]
    if not len(nz):
        return 0.0
    return float(-(nz * np.log2(nz)).sum())


def fidelity_to_pure(rho: np.ndarray, psi: np.ndarray) -> float:
    """Uhlmann fidelity against a pure state |psi>."""
    return float(np.real(np.vdot(psi, rho @ psi)))


def effective_dimension(rho: np.ndarray) -> float:
    tr2 = float(np.real(np.trace(rho @ rho)))
    tr2 = max(tr2, 1e-15)
    return 1.0 / tr2


def fit_log_slope(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit log-log slope y ~ C * x^alpha."""
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan"), float("nan")
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    A = np.vstack([lx, np.ones_like(lx)]).T
    alpha, logC = LA.lstsq(A, ly, rcond=None)[0]
    pred = alpha * lx + logC
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(alpha), float(r2)


def embed_single(op: np.ndarray, index: int, total: int) -> np.ndarray:
    """Embed a single-qubit operator at position `index` (0-based)."""
    ops = [I2] * total
    ops[index] = op
    out = ops[0]
    for mat in ops[1:]:
        out = np.kron(out, mat)
    return out


def embed_two_site(op1: np.ndarray, idx1: int, op2: np.ndarray, idx2: int, total: int) -> np.ndarray:
    """Embed a two-qubit operator (op1 on idx1, op2 on idx2) into total register."""
    if idx1 == idx2:
        raise ValueError("Indices must differ for two-site embedding.")
    ops = [I2] * total
    ops[idx1] = op1
    ops[idx2] = op2
    out = ops[0]
    for mat in ops[1:]:
        out = np.kron(out, mat)
    return out


# ---------------------------------------------------------------------------
# Scenario implementations
# ---------------------------------------------------------------------------


def random_two_body_unitary(nS: int, nE: int, kappa: float, rng: Generator) -> np.ndarray:
    """Chaotic two-body Hamiltonian exp(-i kappa H) coupling every S qubit to every E qubit."""
    total = nS + nE
    dim = 2 ** total
    H = np.zeros((dim, dim), dtype=complex)
    scale = 1.0 / np.sqrt(max(1, nS * nE))
    for i in range(nS):
        for j in range(nE):
            s_idx = i
            e_idx = nS + j
            for Pa in (sigma_x, sigma_y, sigma_z):
                for Pb in (sigma_x, sigma_y, sigma_z):
                    coeff = scale * rng.normal()
                    op = embed_single(Pa, s_idx, total) @ embed_single(Pb, e_idx, total)
                    H += coeff * op
    return expm(-1j * kappa * H)


def xxz_chain_unitary(nS: int, nE: int, kappa: float, delta: float, rng: Generator) -> np.ndarray:
    """Integrable XXZ chain on the combined S+E register with open boundary conditions."""
    del rng  # deterministic
    total = nS + nE
    dim = 2 ** total
    H = np.zeros((dim, dim), dtype=complex)
    for idx in range(total - 1):
        H += embed_two_site(sigma_x, idx, sigma_x, idx + 1, total)
        H += embed_two_site(sigma_y, idx, sigma_y, idx + 1, total)
        H += delta * embed_two_site(sigma_z, idx, sigma_z, idx + 1, total)
    return expm(-1j * kappa * H)


def amplitude_damping_joint_state(psiS: np.ndarray, gamma: float) -> np.ndarray:
    """Return |Psi_SE> for amplitude damping with probability gamma."""
    a, b = psiS
    amp = np.zeros(4, dtype=complex)
    amp[0] = a  # |0,0>
    amp[1] = b * np.sqrt(gamma)  # |0,1>
    amp[2] = b * np.sqrt(max(0.0, 1.0 - gamma))  # |1,0>
    return amp


def measurement_pointer_state(psiS: np.ndarray, p_meas: float) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Return |Psi_SE> for a measurement-induced dephasing channel with probability p_meas.

    Environment has dimension 3: |0> (no measurement), |1> (outcome 0), |2> (outcome 1).
    """
    a, b = psiS
    dS, dE = 2, 3
    amp = np.zeros(dS * dE, dtype=complex)
    # No measurement branch
    amp[0] = np.sqrt(max(0.0, 1.0 - p_meas)) * a          # |0>_S |0>_E
    amp[dE] = np.sqrt(max(0.0, 1.0 - p_meas)) * b         # |1>_S |0>_E
    # Outcome 0 branch
    amp[1] = np.sqrt(p_meas) * a                          # |0>_S |1>_E
    # Outcome 1 branch
    amp[2 + dE] = np.sqrt(p_meas) * b                     # |1>_S |2>_E
    return amp, (dS, dE)


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    name: str
    n_samples: int
    kappa: float
    extra: Dict[str, float]
    alpha: float
    alpha_r2: float
    ratio_mean: float
    ratio_std: float
    y_mean: float
    y_std: float
    y_cv: float
    i_mean: float
    deff_mean: float

    def to_dict(self) -> Dict[str, float]:
        out = {
            "name": self.name,
            "n_samples": self.n_samples,
            "kappa": self.kappa,
            "alpha": self.alpha,
            "alpha_r2": self.alpha_r2,
            "ratio_mean": self.ratio_mean,
            "ratio_std": self.ratio_std,
            "Y_mean": self.y_mean,
            "Y_std": self.y_std,
            "Y_cv": self.y_cv,
            "I_mean": self.i_mean,
            "d_eff_mean": self.deff_mean,
        }
        out.update(self.extra)
        return out


def sample_unitary_scenario(
    name: str,
    build_U: Callable[[int, int, float, Generator], np.ndarray],
    nS: int,
    nE: int,
    kappa: float,
    *,
    samples: int,
    seed: int,
    extra: Dict[str, float] | None = None,
) -> ScenarioResult:
    rng = default_rng(seed)
    U = build_U(nS, nE, kappa, rng)
    dims = [2] * (nS + nE)
    I_vals, ratio_vals, Y_vals, deff_vals = [], [], [], []
    eps = 1e-12

    for _ in range(samples):
        psiS = haar_state(nS, rng)
        psiE = haar_state(nE, rng)
        psiSE0 = np.kron(psiS, psiE)
        psiSE1 = U @ psiSE0
        rhoSE1 = dm(psiSE1)
        rhoS1 = partial_trace(rhoSE1, dims, keep=list(range(nS)))
        Ibits = 2.0 * von_neumann_entropy_bits(rhoS1)
        F = fidelity_to_pure(rhoS1, psiS)
        A2 = float(np.arccos(np.sqrt(np.clip(F, 0.0, 1.0))) ** 2)
        deff = effective_dimension(rhoS1)
        X = max(deff - 1.0, eps)
        ratio = max(A2 / max(Ibits, eps), eps)
        Y = np.sqrt(X) * ratio
        I_vals.append(Ibits)
        ratio_vals.append(ratio)
        Y_vals.append(Y)
        deff_vals.append(deff)

    X_array = np.clip(np.array(deff_vals) - 1.0, eps, None)
    ratio_array = np.array(ratio_vals)
    Y_array = np.array(Y_vals)
    I_array = np.array(I_vals)

    alpha, r2 = fit_log_slope(X_array, ratio_array)
    return ScenarioResult(
        name=name,
        n_samples=len(I_vals),
        kappa=kappa,
        extra=extra or {},
        alpha=alpha,
        alpha_r2=r2,
        ratio_mean=float(ratio_array.mean()),
        ratio_std=float(ratio_array.std()),
        y_mean=float(Y_array.mean()),
        y_std=float(Y_array.std()),
        y_cv=float(Y_array.std() / max(abs(Y_array.mean()), eps)),
        i_mean=float(I_array.mean()),
        deff_mean=float(np.mean(deff_vals)),
    )


def sample_amplitude_damping(
    gamma: float,
    *,
    samples: int,
    seed: int,
) -> ScenarioResult:
    rng = default_rng(seed)
    I_vals, ratio_vals, Y_vals, deff_vals = [], [], [], []
    eps = 1e-12

    for _ in range(samples):
        psiS = haar_state(1, rng)
        joint = amplitude_damping_joint_state(psiS, gamma)
        rhoSE1 = dm(joint)
        rhoS1 = partial_trace(rhoSE1, [2, 2], keep=[0])
        Ibits = 2.0 * von_neumann_entropy_bits(rhoS1)
        F = fidelity_to_pure(rhoS1, psiS)
        A2 = float(np.arccos(np.sqrt(np.clip(F, 0.0, 1.0))) ** 2)
        deff = effective_dimension(rhoS1)
        X = max(deff - 1.0, eps)
        ratio = max(A2 / max(Ibits, eps), eps)
        Y = np.sqrt(X) * ratio

        I_vals.append(Ibits)
        ratio_vals.append(ratio)
        Y_vals.append(Y)
        deff_vals.append(deff)

    X_array = np.clip(np.array(deff_vals) - 1.0, eps, None)
    ratio_array = np.array(ratio_vals)
    Y_array = np.array(Y_vals)
    I_array = np.array(I_vals)

    alpha, r2 = fit_log_slope(X_array, ratio_array)
    return ScenarioResult(
        name="amplitude-damping",
        n_samples=len(I_vals),
        kappa=gamma,
        extra={"gamma": gamma},
        alpha=alpha,
        alpha_r2=r2,
        ratio_mean=float(ratio_array.mean()),
        ratio_std=float(ratio_array.std()),
        y_mean=float(Y_array.mean()),
        y_std=float(Y_array.std()),
        y_cv=float(Y_array.std() / max(abs(Y_array.mean()), eps)),
        i_mean=float(I_array.mean()),
        deff_mean=float(np.mean(deff_vals)),
    )


def sample_measurement_channel(
    p_meas: float,
    *,
    samples: int,
    seed: int,
) -> ScenarioResult:
    rng = default_rng(seed)
    I_vals, ratio_vals, Y_vals, deff_vals = [], [], [], []
    eps = 1e-12

    for _ in range(samples):
        psiS = haar_state(1, rng)
        amp, dims = measurement_pointer_state(psiS, p_meas)
        rhoSE1 = dm(amp)
        rhoS1 = partial_trace(rhoSE1, dims, keep=[0])
        Ibits = 2.0 * von_neumann_entropy_bits(rhoS1)
        F = fidelity_to_pure(rhoS1, psiS)
        A2 = float(np.arccos(np.sqrt(np.clip(F, 0.0, 1.0))) ** 2)
        deff = effective_dimension(rhoS1)
        X = max(deff - 1.0, eps)
        ratio = max(A2 / max(Ibits, eps), eps)
        Y = np.sqrt(X) * ratio

        I_vals.append(Ibits)
        ratio_vals.append(ratio)
        Y_vals.append(Y)
        deff_vals.append(deff)

    X_array = np.clip(np.array(deff_vals) - 1.0, eps, None)
    ratio_array = np.array(ratio_vals)
    Y_array = np.array(Y_vals)
    I_array = np.array(I_vals)

    alpha, r2 = fit_log_slope(X_array, ratio_array)
    return ScenarioResult(
        name="measurement-dephasing",
        n_samples=len(I_vals),
        kappa=p_meas,
        extra={"p_meas": p_meas},
        alpha=alpha,
        alpha_r2=r2,
        ratio_mean=float(ratio_array.mean()),
        ratio_std=float(ratio_array.std()),
        y_mean=float(Y_array.mean()),
        y_std=float(Y_array.std()),
        y_cv=float(Y_array.std() / max(abs(Y_array.mean()), eps)),
        i_mean=float(I_array.mean()),
        deff_mean=float(np.mean(deff_vals)),
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    out_dir = Path(__file__).resolve().parent.parent / "analysis"
    out_dir.mkdir(exist_ok=True)
    results = []

    # Chaotic random two-body
    results.append(
        sample_unitary_scenario(
            name="chaotic-random-2body",
            build_U=random_two_body_unitary,
            nS=2,
            nE=2,
            kappa=0.60,
            samples=800,
            seed=11,
            extra={"nS": 2, "nE": 2},
        )
    )

    # Integrable XXZ chain
    results.append(
        sample_unitary_scenario(
            name="integrable-xxz",
            build_U=lambda nS, nE, kappa, rng: xxz_chain_unitary(nS, nE, kappa, delta=1.2, rng=rng),
            nS=2,
            nE=2,
            kappa=0.60,
            samples=800,
            seed=23,
            extra={"nS": 2, "nE": 2, "delta": 1.2},
        )
    )

    # Amplitude damping (dissipative)
    results.append(
        sample_amplitude_damping(
            gamma=0.35,
            samples=1000,
            seed=37,
        )
    )

    # Measurement-induced dephasing
    results.append(
        sample_measurement_channel(
            p_meas=0.40,
            samples=1000,
            seed=53,
        )
    )

    data = [res.to_dict() for res in results]
    out_path = out_dir / "universality_results.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    for res in results:
        print(f"[{res.name}] n={res.n_samples} | alpha={res.alpha:+.3f} (R2={res.alpha_r2:.3f}) | "
              f"<Y>={res.y_mean:.3f} +/- {res.y_std:.3f} | CV={res.y_cv:.3f}")

    print(f"\nSaved summary to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
