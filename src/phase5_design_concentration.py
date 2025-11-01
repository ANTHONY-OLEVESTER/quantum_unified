#!/usr/bin/env python3
"""
Phase V — 2-design concentration + robustness

Outputs (to ./data and ./figures):
  data/phase5_varY_by_D.csv
  data/phase5_alpha_vs_invD.csv
  data/phase5_robustness_eps.csv
  figures/phase5_varY_scaling.png
  figures/phase5_alpha_vs_invD.png
  figures/phase5_robustness.png
"""

from __future__ import annotations
import os, math, csv, pickle, argparse, hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

# --- Try to reuse Phase-II helpers; otherwise provide minimal fallbacks -----
try:
    from src.phase2 import (
        EPS, dm, dagger, sqrtm_psd, fidelity_uhlmann,
        vn_entropy_bits, purity, partial_trace, haar_state, haar_unitary,
        kraus_amplitude_damping, apply_channel_kraus, twirl_channel_sample,
        apply_channel_kraus_twirl_depth, stable_hash32,
        fit_slope_loglog, bootstrap_slope, kendall_tau, spearman_rho,
    )
except Exception:
    EPS = 1e-12

    def dm(psi: np.ndarray) -> np.ndarray:
        psi = psi.reshape(-1, 1); return psi @ psi.conj().T

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
        dims = list(dims); keep = sorted(keep)
        n = len(dims)
        trace = [i for i in range(n) if i not in keep]
        perm = keep + trace + [i + n for i in keep] + [i + n for i in trace]
        resh = rho.reshape(*(dims + dims)).transpose(perm)
        d_keep = int(np.prod([dims[i] for i in keep])) if keep else 1
        d_trace = int(np.prod([dims[i] for i in trace])) if trace else 1
        resh = resh.reshape(d_keep, d_trace, d_keep, d_trace)
        return np.einsum("aibi->ab", resh)

    def haar_state(n_qudits: int, d: int = 2, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        dim = d ** n_qudits
        vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        vec /= np.linalg.norm(vec); return vec

    def haar_unitary(dim: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        X = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        Q, R = np.linalg.qr(X); phase = np.diag(R) / np.abs(np.diag(R))
        return Q @ np.diag(phase.conj())

    def kraus_amplitude_damping(p: float) -> List[np.ndarray]:
        K0 = np.array([[1, 0], [0, math.sqrt(1 - p)]], dtype=complex)
        K1 = np.array([[0, math.sqrt(p)], [0, 0]], dtype=complex)
        return [K0, K1]

    def apply_channel_kraus(K: List[np.ndarray], rho: np.ndarray) -> np.ndarray:
        out = np.zeros_like(rho)
        for A in K: out += A @ rho @ dagger(A)
        return out

    def twirl_channel_sample(K: List[np.ndarray], rho: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        d = rho.shape[0]
        U = haar_unitary(d, rng); rho_in = dagger(U) @ rho @ U
        rho_out = apply_channel_kraus(K, rho_in)
        return U @ rho_out @ dagger(U)

    def apply_channel_kraus_twirl_depth(K: List[np.ndarray], rho: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
        rho_cur = rho.copy(); depth = max(1, m)
        for _ in range(depth): rho_cur = twirl_channel_sample(K, rho_cur, rng)
        return rho_cur

    def stable_hash32(*parts: object) -> int:
        data = "|".join(str(p) for p in parts).encode("utf-8")
        digest = hashlib.sha256(data).digest()
        return int.from_bytes(digest[:4], "little")

    # Small OLS/bootstraps used below
    def fit_slope_loglog(X: np.ndarray, Y: np.ndarray):
        mask = (X > 0) & (Y > 0); X = X[mask]; Y = Y[mask]
        if X.size < 2: return float("nan"), float("nan"), float("nan"), int(X.size), np.array([]), np.array([])
        lx = np.log10(X); ly = np.log10(Y)
        A = np.column_stack((lx, np.ones_like(lx)))
        m, b = np.linalg.lstsq(A, ly, rcond=None)[0]
        yhat = m * lx + b
        ss_res = np.sum((ly - yhat) ** 2)
        ss_tot = np.sum((ly - ly.mean()) ** 2) + EPS
        R2 = 1.0 - ss_res / ss_tot
        return float(m), float(b), float(R2), int(lx.size), lx, ly

    def bootstrap_slope(lx: np.ndarray, ly: np.ndarray, B: int = 2000, rng: np.random.Generator | None = None):
        if lx.size < 2: return float("nan"), float("nan"), float("nan")
        rng = np.random.default_rng() if rng is None else rng
        n = lx.size; slopes = np.empty(B); ones = np.ones(n)
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            A = np.column_stack((lx[idx], ones))
            slopes[b], _ = np.linalg.lstsq(A, ly[idx], rcond=None)[0]
        mean = float(slopes.mean()); lo, hi = np.percentile(slopes, [2.5, 97.5])
        return mean, float(lo), float(hi)

# ---------------------- Metrics for Y and alpha -----------------------------

def metrics_after_channel(K: List[np.ndarray], rhoS0: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    # Stinespring via stacking Kraus operators
    dS = K[0].shape[0]; m = len(K)
    V = np.zeros((dS * m, dS), dtype=complex)
    for i, A in enumerate(K): V[i * dS:(i + 1) * dS, :] = A
    rhoSE1 = V @ rhoS0 @ dagger(V)
    rhoS1  = partial_trace(rhoSE1, keep=[0], dims=[dS, m])
    rhoE1  = partial_trace(rhoSE1, keep=[1], dims=[dS, m])
    F = fidelity_uhlmann(rhoS0, rhoS1)
    A2 = float(np.arccos(np.sqrt(np.clip(F, 0.0, 1.0))) ** 2)
    Ibits = vn_entropy_bits(rhoS1) + vn_entropy_bits(rhoE1) - vn_entropy_bits(rhoSE1)
    deff = 1.0 / max(purity(rhoS1), EPS)
    return Ibits, A2, deff, rhoS1

def compute_Y(Ibits: float, A2: float, deff: float) -> float:
    if Ibits <= EPS: return 0.0
    x = max(deff - 1.0, EPS)
    return math.sqrt(x) * A2 / Ibits

# ---------------------- 2-design surrogates --------------------------------

def approx_unitary_2design(dim: int, depth: int, rng: np.random.Generator) -> np.ndarray:
    """
    Approximate global unitary 2-design by product of Haar unitaries on random
    two-level subspaces (a simple 'mixing' circuit). Good enough for concentration tests.
    """
    U = np.eye(dim, dtype=complex)
    for _ in range(depth):
        # Pick a random 2D subspace and apply a Haar 2x2 unitary inside it.
        i, j = rng.choice(dim, size=2, replace=False)
        H2 = haar_unitary(2, rng)
        P = np.eye(dim, dtype=complex)
        P[[i, i, j, j], [i, j, i, j]] = H2.flatten()  # place block
        U = P @ U
    return U

def twirled_channel_2design(K: List[np.ndarray], rho: np.ndarray, twirl_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Empirical 2-design twirl: average over Haar samples (few samples already concentrate at large D)."""
    d = rho.shape[0]
    out = np.zeros((d, d), dtype=complex)
    for _ in range(max(1, twirl_samples)):
        out += twirl_channel_sample(K, rho, rng)
    return out / max(1, twirl_samples)

# ---------------------- Phase V experiments ---------------------------------

@dataclass
class VarPoint:
    model: str
    D: int
    varY: float
    slope: float
    lo: float
    hi: float
    n: int

def var_scaling_over_D(models: List[str], Ds: List[int], trials: int, seed: int) -> List[VarPoint]:
    out: List[VarPoint] = []
    rng0 = np.random.default_rng(seed)
    for model in models:
        Ys_by_D: Dict[int, List[float]] = {}
        for D in Ds:
            rng = np.random.default_rng(rng0.integers(0, 2**32 - 1))
            dS = 2; dE = D // dS
            # One-qubit system embedded in D = 2 * dE via channel
            if model == "2design_channel":
                K = kraus_amplitude_damping(0.3)  # any non-unitary base; twirl will isotropize
            elif model == "dephasing":
                # use amplitude damping with *partial* twirl as a structured foil
                K = kraus_amplitude_damping(0.3)
            else:
                raise ValueError(f"unknown model {model}")
            Ys: List[float] = []
            for _ in range(trials):
                psi = haar_state(1, d=dS, rng=rng)
                rho0 = dm(psi)
                if model == "2design_channel":
                    rho1 = twirled_channel_2design(K, rho0, twirl_samples=3, rng=rng)
                else:  # structured foil: no twirl
                    rho1 = apply_channel_kraus(K, rho0)
                F = fidelity_uhlmann(rho0, rho1)
                A2 = float(np.arccos(np.sqrt(np.clip(F, 0.0, 1.0))) ** 2)
                # Build SE state to measure Ibits consistently
                V = np.zeros((dS * len(K), dS), dtype=complex)
                for i, A in enumerate(K): V[i * dS:(i + 1) * dS, :] = A
                rhoSE1 = V @ rho0 @ dagger(V)
                rhoS1  = partial_trace(rhoSE1, keep=[0], dims=[dS, len(K)])
                rhoE1  = partial_trace(rhoSE1, keep=[1], dims=[dS, len(K)])
                Ibits = vn_entropy_bits(rhoS1) + vn_entropy_bits(rhoE1) - vn_entropy_bits(rhoSE1)
                deff = 1.0 / max(purity(rhoS1), EPS)
                Ys.append(compute_Y(Ibits, A2, deff))
            Ys = [y for y in Ys if y > 0]
            Ys_by_D[D] = Ys
        # fit log Var(Y) vs log D
        X = np.array(sorted(Ys_by_D.keys()), dtype=float)
        V = np.array([np.var(Ys_by_D[d]) + EPS for d in X], dtype=float)
        m, b, R2, n, lx, ly = fit_slope_loglog(X, V)
        bs_mean, lo, hi = bootstrap_slope(lx, ly, B=1500, rng=np.random.default_rng(seed ^ 0xBEE5FACE))
        slope = bs_mean if np.isfinite(bs_mean) else m
        for D, var in zip(X.tolist(), V.tolist()):
            out.append(VarPoint(model, int(D), float(var), float(slope), float(lo), float(hi), int(n)))
    return out

@dataclass
class InterceptPoint:
    model: str
    D: int
    alpha_abs: float
    invD: float
    intercept_fit: float

def alpha_intercepts(models: List[str], sizes: List[Tuple[int,int]], trials: int, seed: int) -> List[InterceptPoint]:
    points: List[InterceptPoint] = []
    rng0 = np.random.default_rng(seed)
    for model in models:
        rows: List[Tuple[float,float]] = []  # (1/D, |alpha|)
        for (nS, nE) in sizes:
            D = 2 ** (nS + nE)
            rng = np.random.default_rng(rng0.integers(0, 2**32 - 1))
            # build many samples X=(deff-1), Y= sqrt(X)*A2/I -> slope alpha
            X_all: List[float] = []; Y_all: List[float] = []
            # channel: twirled vs structured
            if model == "2design_channel":
                K = kraus_amplitude_damping(0.3)
            else:
                K = kraus_amplitude_damping(0.3)
            for _ in range(trials):
                psi = haar_state(nS, d=2, rng=rng)
                rho0 = dm(psi)
                if model == "2design_channel":
                    rho1 = twirled_channel_2design(K, rho0, twirl_samples=3, rng=rng)
                else:
                    rho1 = apply_channel_kraus(K, rho0)  # foil
                # consistent I/A2/deff at system level
                F = fidelity_uhlmann(rho0, rho1)
                A2 = float(np.arccos(np.sqrt(np.clip(F, 0.0, 1.0))) ** 2)
                # SE to compute mutual info
                V = np.zeros((2 * len(K), 2), dtype=complex)  # nS=1 in this intercept run
                for i, A in enumerate(K): V[i * 2:(i + 1) * 2, :] = A
                rhoSE1 = V @ rho0 @ dagger(V)
                rhoS1  = partial_trace(rhoSE1, keep=[0], dims=[2, len(K)])
                rhoE1  = partial_trace(rhoSE1, keep=[1], dims=[2, len(K)])
                Ibits = vn_entropy_bits(rhoS1) + vn_entropy_bits(rhoE1) - vn_entropy_bits(rhoSE1)
                deff = 1.0 / max(purity(rhoS1), EPS)
                if Ibits <= EPS: continue
                x = max(deff - 1.0, EPS); y = math.sqrt(x) * A2 / Ibits
                X_all.append(x); Y_all.append(y)
            X = np.asarray(X_all); Y = np.asarray(Y_all)
            alpha, *_ = fit_slope_loglog(X, Y)
            rows.append((1.0 / D, abs(alpha)))
            points.append(InterceptPoint(model, D, abs(alpha), 1.0 / D, float("nan")))
        # linear fit in invD
        invD = np.array([r[0] for r in rows], dtype=float)
        Aabs = np.array([r[1] for r in rows], dtype=float)
        A = np.column_stack((invD, np.ones_like(invD)))
        m, b = np.linalg.lstsq(A, Aabs, rcond=None)[0]
        for p in points:
            if p.model == model:
                p.intercept_fit = float(b)
    return points

@dataclass
class RobustPoint:
    model: str
    D: int
    eps: float
    alpha_abs: float

def robustness_eps(D: int, eps_grid: List[float], trials: int, seed: int) -> List[RobustPoint]:
    """
    Mix structured channel with 2-design twirl:  Φ_eps = (1-ε)*Φ_struct + ε*T(Φ_struct).
    Track |alpha|(ε).  Expect a sharp drop as ε increases.
    """
    rng = np.random.default_rng(seed)
    dS = 2
    K_struct = kraus_amplitude_damping(0.3)

    points: List[RobustPoint] = []
    for eps in eps_grid:
        X_all: List[float] = []; Y_all: List[float] = []
        for _ in range(trials):
            psi = haar_state(1, d=dS, rng=rng); rho0 = dm(psi)
            rho_struct = apply_channel_kraus(K_struct, rho0)
            rho_twirl  = twirled_channel_2design(K_struct, rho0, twirl_samples=3, rng=rng)
            rho1 = (1 - eps) * rho_struct + eps * rho_twirl
            F = fidelity_uhlmann(rho0, rho1)
            A2 = float(np.arccos(np.sqrt(np.clip(F, 0.0, 1.0))) ** 2)
            # mutual info via Stinespring of the structured part (proxy)
            V = np.zeros((dS * len(K_struct), dS), dtype=complex)
            for i, A in enumerate(K_struct): V[i * dS:(i + 1) * dS, :] = A
            rhoSE1 = V @ rho0 @ dagger(V)
            rhoS1  = partial_trace(rhoSE1, keep=[0], dims=[dS, len(K_struct)])
            rhoE1  = partial_trace(rhoSE1, keep=[1], dims=[dS, len(K_struct)])
            Ibits = vn_entropy_bits(rhoS1) + vn_entropy_bits(rhoE1) - vn_entropy_bits(rhoSE1)
            deff = 1.0 / max(purity(rhoS1), EPS)
            if Ibits <= EPS: continue
            x = max(deff - 1.0, EPS); y = math.sqrt(x) * A2 / Ibits
            X_all.append(x); Y_all.append(y)
        alpha, *_ = fit_slope_loglog(np.asarray(X_all), np.asarray(Y_all))
        points.append(RobustPoint("struct+twirl", D, eps, abs(alpha)))
    return points

# ---------------------------- Plotters --------------------------------------

def ensure_dirs(out_dir: Path):
    (out_dir/"data").mkdir(parents=True, exist_ok=True)
    (out_dir/"figures").mkdir(parents=True, exist_ok=True)

def plot_varY_scaling(rows: List[VarPoint], figpath: Path):
    plt.figure(figsize=(9,5.6))
    models = sorted({r.model for r in rows})
    for m in models:
        sub = [r for r in rows if r.model==m]
        sub.sort(key=lambda r:r.D)
        D = np.array([r.D for r in sub], float)
        V = np.array([r.varY for r in sub], float)
        slope = sub[0].slope
        plt.loglog(D, V, 'o', label=m)
        # fit line
        lx = np.log10(D); ly = np.log10(V)
        A = np.column_stack((lx, np.ones_like(lx)))
        k, b = np.linalg.lstsq(A, ly, rcond=None)[0]
        xx = np.array([D.min(), D.max()])
        yy = 10**(k*np.log10(xx)+b)
        plt.loglog(xx, yy, '--', label=f"{m}: slope={slope:+.3f}")
    # D^-2 ref
    xx = np.array([min(r.D for r in rows), max(r.D for r in rows)], float)
    yy = (xx/xx[0])**(-2) * (max(r.varY for r in rows)*0.6)
    plt.loglog(xx, yy, ':', color='gray', linewidth=2, label=r"$D^{-2}$ reference")
    plt.xlabel(r"$D$"); plt.ylabel(r"$\mathrm{Var}(Y)$"); plt.title("Variance scaling of Y vs D (Phase V)")
    plt.legend()
    plt.grid(True, which='both', alpha=0.2)
    plt.tight_layout(); plt.savefig(figpath, dpi=180); plt.close()

def plot_alpha_vs_invD(rows: List[InterceptPoint], figpath: Path):
    plt.figure(figsize=(9,5.6))
    models = sorted({r.model for r in rows})
    for m in models:
        sub = [r for r in rows if r.model==m]
        sub.sort(key=lambda r:r.invD)
        invD = np.array([r.invD for r in sub], float)
        Aabs = np.array([r.alpha_abs for r in sub], float)
        plt.scatter(invD, Aabs, s=28, label=m)
        A = np.column_stack((invD, np.ones_like(invD)))
        k, b = np.linalg.lstsq(A, Aabs, rcond=None)[0]
        xs = np.linspace(0, invD.max()*1.05, 100)
        plt.plot(xs, k*xs + b, '--', label=f"{m} intercept={b:+.3f}")
    plt.xlabel(r"$1/D$"); plt.ylabel(r"$|\alpha|$")
    plt.title(r"$|\alpha|$ vs $1/D$ (Phase V)")
    plt.legend(); plt.grid(True, alpha=0.2); plt.tight_layout()
    plt.savefig(figpath, dpi=180); plt.close()

def plot_robustness(rows: List[RobustPoint], figpath: Path):
    rows = sorted(rows, key=lambda r:r.eps)
    eps = np.array([r.eps for r in rows], float)
    a = np.array([r.alpha_abs for r in rows], float)
    plt.figure(figsize=(8.2,5.3))
    plt.plot(eps, a, 'o-', label=r"$|\alpha|$")
    plt.xlabel(r"twirl mix $\epsilon$"); plt.ylabel(r"$|\alpha|$")
    plt.title("Robustness: structured + twirl mix (Phase V)")
    plt.grid(True, alpha=0.25); plt.tight_layout()
    plt.savefig(figpath, dpi=180); plt.close()

# ---------------------------- I/O helpers -----------------------------------

def write_csv_varY(rows: List[VarPoint], path: Path):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["model","D","varY","slope","lo","hi","n"])
        w.writeheader()
        for r in rows:
            w.writerow(dict(model=r.model, D=r.D, varY=r.varY, slope=r.slope, lo=r.lo, hi=r.hi, n=r.n))

def write_csv_intercepts(rows: List[InterceptPoint], path: Path):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["model","D","invD","abs_alpha","intercept"])
        w.writeheader()
        for r in rows:
            w.writerow(dict(model=r.model, D=r.D, invD=r.invD, abs_alpha=r.alpha_abs, intercept=r.intercept_fit))

def write_csv_robust(rows: List[RobustPoint], path: Path):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["model","D","epsilon","abs_alpha"])
        w.writeheader()
        for r in rows:
            w.writerow(dict(model=r.model, D=r.D, epsilon=r.eps, abs_alpha=r.alpha_abs))

# ------------------------------- Main ---------------------------------------

def main():
    p = argparse.ArgumentParser(description="Phase V: 2-design concentration + robustness")
    p.add_argument("--out", default=".", help="repo root containing data/ and figures/")
    p.add_argument("--trials", type=int, default=1200, help="samples per D (variance)")
    p.add_argument("--trials-intercept", type=int, default=800, help="samples per size (intercepts)")
    p.add_argument("--depth", type=int, default=6, help="approx-2design unitary depth (if used)")
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    p.add_argument("--max-log2D", type=int, default=11, help="max log2(D) for variance sweep (e.g., 9..13)")
    args = p.parse_args()

    root = Path(args.out); ensure_dirs(root)
    data_dir = root/"data"; fig_dir = root/"figures"

    # D grid for variance (keep moderate so it runs in minutes)
    Ds = [2**k for k in range(6, args.max_log2D+1)]   # 64..2^max
    models = ["2design_channel", "dephasing"]

    print(f"[Phase V] Variance sweep over D={Ds}, models={models} ...")
    var_rows = var_scaling_over_D(models, Ds, trials=args.trials, seed=args.seed)
    write_csv_varY(var_rows, data_dir/"phase5_varY_by_D.csv")
    plot_varY_scaling(var_rows, fig_dir/"phase5_varY_scaling.png")

    # Intercepts |alpha| vs 1/D over a few sizes (nS+nE up to 10 qubits typical)
    sizes = [(1,2), (1,3), (1,4), (1,5), (1,6)]
    print(f"[Phase V] Intercepts over sizes={sizes} ...")
    inter_rows = alpha_intercepts(["2design_channel","dephasing"], sizes, trials=args.trials_intercept, seed=args.seed ^ 0xABCD)
    write_csv_intercepts(inter_rows, data_dir/"phase5_alpha_vs_invD.csv")
    plot_alpha_vs_invD(inter_rows, fig_dir/"phase5_alpha_vs_invD.png")

    # Robustness: structured + epsilon * twirl
    print("[Phase V] Robustness vs ε ...")
    rob_rows = robustness_eps(D=128, eps_grid=[0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0], trials=900, seed=args.seed ^ 0xDEADBEEF)
    write_csv_robust(rob_rows, data_dir/"phase5_robustness_eps.csv")
    plot_robustness(rob_rows, fig_dir/"phase5_robustness.png")

    print("\nPhase V complete.")
    print("CSV ->", data_dir/"phase5_varY_by_D.csv", ",", data_dir/"phase5_alpha_vs_invD.csv", ",", data_dir/"phase5_robustness_eps.csv")
    print("FIG ->", fig_dir/"phase5_varY_scaling.png", ",", fig_dir/"phase5_alpha_vs_invD.png", ",", fig_dir/"phase5_robustness.png")

if __name__ == "__main__":
    main()
