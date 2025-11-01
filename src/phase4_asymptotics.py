#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# Phase IV — Asymptotics & Concentration
#   * Push to larger D = 2^(nS+nE)
#   * Fit |alpha| vs 1/D  (flat intercept -> universality)
#   * Fit Var(Y) vs D     (target slope ≈ -2 under 2-design/isotropy)
#   * Save CSV artifacts + figures
#   * Deterministic seeding via SHA256 of labels
# ============================================================================
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Try to reuse Phase 2 utilities if present; otherwise fall back to locals.
# We only need: collect_unitary, collect_depolarizing (optional),
#               fit_slope_loglog (log-log Y vs (deff-1) to get alpha).
# ---------------------------------------------------------------------------
try:
    from phase2 import (  # type: ignore
        collect_unitary as _p2_collect_unitary,
        collect_depolarizing as _p2_collect_depolarizing,
        fit_slope_loglog as _p2_fit_slope_loglog,
    )
    HAVE_PHASE2 = True
except Exception:
    HAVE_PHASE2 = False

# ==== local fallbacks (mirrors your Phase-2 code paths) =====================
EPS = 1e-12
I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]], complex)
sz = np.array([[1, 0], [0, -1]], complex)
PAULIS = (sx, sy, sz)

def _dag(A: np.ndarray) -> np.ndarray:
    return A.conj().T

def _dm(psi: np.ndarray) -> np.ndarray:
    psi = psi.reshape(-1, 1)
    return psi @ psi.conj().T

def _eigh_psd(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H = (M + _dag(M)) / 2
    w, V = np.linalg.eigh(H)
    return np.clip(np.real(w), 0.0, None), V

def _sqrtm_psd(M: np.ndarray) -> np.ndarray:
    w, V = _eigh_psd(M)
    return V @ np.diag(np.sqrt(w)) @ _dag(V)

def _fidelity_uhlmann(rho: np.ndarray, sigma: np.ndarray) -> float:
    s = _sqrtm_psd(rho) @ sigma @ _sqrtm_psd(rho)
    return float(np.trace(_sqrtm_psd(s)).real ** 2)

def _vn_entropy_bits(rho: np.ndarray, tol: float = 1e-12) -> float:
    w, _ = _eigh_psd(rho)
    w = w[w > tol]
    return float((-w * np.log2(w)).sum()) if w.size else 0.0

def _purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))

def _kron(*ops: np.ndarray) -> np.ndarray:
    out = np.array(1.0 + 0.0j)
    for op in ops:
        out = np.kron(out, op)
    return out

def _embed_pair(op_a: np.ndarray, idx_a: int, op_b: np.ndarray, idx_b: int, total: int) -> np.ndarray:
    ops = [I2] * total
    ops[idx_a] = op_a
    ops[idx_b] = op_b
    return _kron(*ops)

def _partial_trace(rho: np.ndarray, keep: Iterable[int], dims: List[int]) -> np.ndarray:
    dims = list(dims)
    keep = sorted(keep)
    n = len(dims)
    trace_over = [i for i in range(n) if i not in keep]
    resh = rho.reshape(*(dims + dims))
    for t in sorted(trace_over, reverse=True):
        resh = np.trace(resh, axis1=t, axis2=t+n)
        dims.pop(t)
        n -= 1
    d_keep = int(np.prod(dims)) if dims else 1
    return resh.reshape(d_keep, d_keep)

def _haar_state(n_qudits: int, d: int = 2, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    dim = d ** n_qudits
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v /= np.linalg.norm(v)
    return v

def _random_2body_H(nS: int, nE: int, rng: np.random.Generator) -> np.ndarray:
    total = nS + nE
    dim = 2 ** total
    H = np.zeros((dim, dim), complex)
    scale = 1.0 / math.sqrt(max(1, nS * nE))
    for i in range(nS):
        for j in range(nE):
            for a in PAULIS:
                for b in PAULIS:
                    H += scale * rng.normal() * _embed_pair(a, i, b, nS + j, total)
    return H

def _structured_H(model: str, nS: int, nE: int) -> np.ndarray:
    total = nS + nE
    dim = 2 ** total
    H = np.zeros((dim, dim), complex)
    for i in range(min(nS, nE)):
        if model == "dephasing":
            H += _embed_pair(sz, i, sz, nS + i, total)
        elif model == "pswap":
            H += 0.25 * (
                _embed_pair(sx, i, sx, nS + i, total)
                + _embed_pair(sy, i, sy, nS + i, total)
                + _embed_pair(sz, i, sz, nS + i, total)
            )
        else:
            raise ValueError(f"unknown structured model {model}")
    return H

def _unitary_from_H(H: np.ndarray, kappa: float) -> np.ndarray:
    w, V = np.linalg.eigh((H + _dag(H)) / 2)
    phases = np.exp(-1j * kappa * w)
    return V @ np.diag(phases) @ _dag(V)

def _metrics_after_unitary(U: np.ndarray, psiS: np.ndarray, psiE: np.ndarray, nS: int, nE: int) -> Tuple[float, float, float]:
    dims = [2] * (nS + nE)
    psiSE0 = _kron(psiS, psiE).reshape(-1)
    rhoSE0 = _dm(psiSE0)
    rhoSE1 = _dm(U @ psiSE0)
    rhoS0 = _partial_trace(rhoSE0, keep=list(range(nS)), dims=dims)
    rhoS1 = _partial_trace(rhoSE1, keep=list(range(nS)), dims=dims)
    F = _fidelity_uhlmann(rhoS0, rhoS1)
    A2 = float(np.arccos(np.sqrt(np.clip(F, 0.0, 1.0))) ** 2)
    Sbits = _vn_entropy_bits(rhoS1)
    Ibits = 2.0 * Sbits
    deff = 1.0 / max(_purity(rhoS1), EPS)
    return Ibits, A2, deff

def _collect_unitary_local(model: str, nS: int, nE: int, kappa: float, n_trials: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if model == "random2body":
        H = _random_2body_H(nS, nE, rng)
    else:
        H = _structured_H(model, nS, nE)
    U = _unitary_from_H(H, kappa)
    X: List[float] = []
    Y: List[float] = []
    Y_raw: List[float] = []
    for _ in range(n_trials):
        psiS = _haar_state(nS, 2, rng)
        psiE = _haar_state(nE, 2, rng)
        Ibits, A2, deff = _metrics_after_unitary(U, psiS, psiE, nS, nE)
        if Ibits <= EPS:
            continue
        x = max(deff - 1.0, EPS)
        y = math.sqrt(x) * A2 / Ibits
        X.append(x); Y.append(y); Y_raw.append(y)
    return np.asarray(X), np.asarray(Y), np.asarray(Y_raw)

def collect_unitary(model: str, nS: int, nE: int, kappa: float, n_trials: int, seed: int):
    if HAVE_PHASE2:
        X, Y = _p2_collect_unitary(model, nS, nE, kappa, n_trials, seed)  # type: ignore
        return X, Y, Y.copy()
    return _collect_unitary_local(model, nS, nE, kappa, n_trials, seed)

def fit_slope_loglog(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float, float, int, np.ndarray, np.ndarray]:
    if HAVE_PHASE2:
        a, b, R2, n, lx, ly = _p2_fit_slope_loglog(X, Y)  # type: ignore
        return a, b, R2, n, lx, ly
    mask = (X > 0) & (Y > 0)
    X = X[mask]; Y = Y[mask]
    if X.size < 2:
        return float("nan"), float("nan"), float("nan"), int(X.size), np.array([]), np.array([])
    lx = np.log10(X); ly = np.log10(Y)
    A = np.column_stack((lx, np.ones_like(lx)))
    m, b = np.linalg.lstsq(A, ly, rcond=None)[0]
    yhat = m*lx + b
    ss_res = np.sum((ly - yhat) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2) + EPS
    R2 = 1.0 - ss_res / ss_tot
    return float(m), float(b), float(R2), int(lx.size), lx, ly

# ---------------------------------------------------------------------------
def stable_hash32(*parts: object) -> int:
    data = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:4], "little")

def linfit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return slope, intercept for y = a x + b (least squares) with finite mask."""
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 2:
        return float("nan"), float("nan")
    A = np.column_stack((x, np.ones_like(x)))
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)

@dataclass
class AlphaPoint:
    model: str
    nS: int
    nE: int
    D: int
    alpha: float

# ---------------------------------------------------------------------------
def run_phase4(
    output_dir: Path | str = "phase4-out",
    models: Tuple[str, ...] = ("random2body", "pswap", "dephasing"),
    kappa: float = 0.6,
    max_qubits: int = 9,
    trials_per_size: int = 1200,
) -> Dict[str, object]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # sizes with nS+nE up to max_qubits
    sizes = []
    for total in range(4, max_qubits + 1):  # start at 4 to see scaling
        for nS in range(1, total):
            nE = total - nS
            sizes.append((nS, nE))
    sizes = sorted(set(sizes), key=lambda p: (p[0] + p[1], p[0]))

    alpha_rows: List[AlphaPoint] = []
    var_by_D: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

    # --- sweep and collect per-size alpha & variance(Y) ---
    for model in models:
        for (nS, nE) in sizes:
            D = 2 ** (nS + nE)
            seed = stable_hash32("phase4", model, nS, nE, kappa)
            X, Y, Yraw = collect_unitary(model, nS, nE, kappa, trials_per_size, seed)
            a, _, R2, n, _, _ = fit_slope_loglog(X, Y)
            if np.isfinite(a):
                alpha_rows.append(AlphaPoint(model, nS, nE, D, float(a)))
            if Yraw.size:
                var_by_D[model][D].append(float(np.var(Yraw)))

    # ---- CSV: alpha vs 1/D (per-point) ------------------------------------
    csv_alpha = outdir / "phase4_alpha_vs_invD.csv"
    with csv_alpha.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "nS", "nE", "D", "invD", "alpha", "abs_alpha"])
        for r in alpha_rows:
            w.writerow([r.model, r.nS, r.nE, r.D, 1.0 / r.D, r.alpha, abs(r.alpha)])

    # ---- CSV: variance(Y) aggregated per D --------------------------------
    csv_var = outdir / "phase4_varY_by_D.csv"
    with csv_var.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "D", "count", "mean_varY"])
        for model, bucket in var_by_D.items():
            for D, vals in sorted(bucket.items()):
                arr = np.asarray(vals, float)
                w.writerow([model, D, arr.size, float(np.mean(arr))])

    # ---- Figure: |alpha| vs 1/D with linear extrapolation ------------------
    fig1 = outdir / "phase4_alpha_vs_invD.png"
    plt.figure(figsize=(8, 5))
    colors = {"random2body": "#2ca02c", "pswap": "#ff7f0e", "dephasing": "#1f77b4"}
    for model in models:
        pts = [r for r in alpha_rows if r.model == model]
        if not pts:
            continue
        invD = np.array([1.0 / p.D for p in pts], float)
        absA = np.array([abs(p.alpha) for p in pts], float)
        srt = np.argsort(invD)
        invD = invD[srt]; absA = absA[srt]
        a, b = linfit(invD, absA)
        xfit = np.linspace(0.0, max(invD)*1.05, 256)
        yfit = a * xfit + b
        plt.scatter(invD, absA, s=40, alpha=0.8, label=model, c=colors.get(model, None))
        plt.plot(xfit, yfit, "--", c=colors.get(model, None),
                 label=f"{model} intercept={b:+.3f}")
    plt.xlabel(r"$1/D$")
    plt.ylabel(r"$|\alpha|$")
    plt.title(r"$|\alpha|$ versus $1/D$ (Phase IV)")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1, dpi=150)
    plt.close()

    # ---- Figure: Var(Y) vs D with log–log slope ----------------------------
    fig2 = outdir / "phase4_varY_scaling.png"
    plt.figure(figsize=(8.2, 5.2))
    for model in models:
        Dvals = np.array(sorted(var_by_D[model].keys()), float)
        if Dvals.size < 2:
            continue
        means = np.array([np.mean(var_by_D[model][int(D)]) for D in Dvals], float)
        # Fit slope in log10 space: log Var = s * log D + c
        lx = np.log10(Dvals + EPS)
        ly = np.log10(np.maximum(means, EPS))
        A = np.column_stack((lx, np.ones_like(lx)))
        slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
        xx = np.logspace(np.log10(Dvals.min()), np.log10(Dvals.max()), 200)
        yy = 10 ** (slope * np.log10(xx) + intercept)
        plt.scatter(Dvals, means, s=50, alpha=0.85, label=f"{model}")
        plt.plot(xx, yy, "--", alpha=0.9, label=f"{model}: slope={slope:+.3f}")
    # D^-2 reference
    Dref = np.logspace(np.log10(max(4, min((var_by_D[models[0]] or {4:[1.0]}).keys()))), np.log10(2 ** max_qubits), 100)
    ref = (Dref / Dref[0]) ** (-2.0) * 0.1
    plt.plot(Dref, ref, color="gray", ls="--", lw=1.5, alpha=0.7, label=r"$D^{-2}$ reference")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel(r"$D$")
    plt.ylabel(r"$\mathrm{Var}(Y)$")
    plt.title(r"Variance scaling of $Y$ vs $D$ (Phase IV)")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2, dpi=150)
    plt.close()

    # ---- tiny summary printout --------------------------------------------
    print("Phase IV asymptotics complete.")
    print(f"Saved: {csv_alpha}, {csv_var}, {fig1}, {fig2}")
    return {
        "alpha_points": alpha_rows,
        "varY_buckets": var_by_D,
        "alpha_csv": csv_alpha,
        "var_csv": csv_var,
        "alpha_fig": fig1,
        "var_fig": fig2,
    }

# ---------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Phase IV asymptotics & concentration")
    ap.add_argument("--output-dir", default="phase4-out", help="Where to write CSVs & figures")
    ap.add_argument("--max-qubits", type=int, default=9, help="Max total qubits nS+nE (default: 9)")
    ap.add_argument("--trials", type=int, default=1200, help="Trials per size (default: 1200)")
    ap.add_argument("--kappa", type=float, default=0.6, help="Coupling scale for unitary models")
    ap.add_argument("--models", default="random2body,pswap,dephasing",
                    help="Comma list of models (subset of random2body,pswap,dephasing)")
    args = ap.parse_args(argv)

    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    run_phase4(
        output_dir=args.output_dir,
        models=models,
        kappa=args.kappa,
        max_qubits=args.max_qubits,
        trials_per_size=args.trials,
    )

if __name__ == "__main__":
    main()
