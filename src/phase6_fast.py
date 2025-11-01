#!/usr/bin/env python3
# Phase 6 (FAST): approximate isotropic sampling for the curvature–information invariant.
#   * Draws Haar random pure states on C^{2 d_E} (nS=1 fast path) and computes Y without QR/eigh.
#   * Matches expectation scaling; variance scaling can be conservative vs full 4-design Stinespring.
#   * For theorem-grade variance, swap sampler for Haar/Clifford 4-design (see paper appendix).

from __future__ import annotations
import argparse, csv, math, os, pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional GPU
try:
    import cupy as cp  # type: ignore
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------- config -------------------------

EPS = 1e-12
DEFAULT_NS = 1
DEFAULT_NE_LIST = [5, 6, 7, 8, 9, 10, 11]
DEFAULT_TRIALS = 4000
BOOT_B = 1200
RNG_SEED = 0xC0FFEE

# ------------------------- small helpers -------------------------

def binary_entropy_bits(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))

def linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    A = np.column_stack((x, np.ones_like(x)))
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)

def bootstrap_line(x: np.ndarray, y: np.ndarray, B: int, seed: int) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(x)
    a_s = np.empty(B, float)
    b_s = np.empty(B, float)
    ones = np.ones(n)
    X = np.column_stack((x, ones))
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        a_s[b], b_s[b] = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
    (a_lo, a_hi) = np.percentile(a_s, [2.5, 97.5])
    (b_lo, b_hi) = np.percentile(b_s, [2.5, 97.5])
    return (float(a_s.mean()), float(b_s.mean())), (float(a_lo), float(a_hi)), (float(b_lo), float(b_hi))

def bootstrap_slope_loglog(Dvals: np.ndarray, vY: np.ndarray, B: int, seed: int) -> Tuple[float, Tuple[float, float]]:
    lx = np.log10(np.clip(Dvals, EPS, None))
    ly = np.log10(np.clip(vY, EPS, None))
    (a_m, _), (a_lo, a_hi), _ = bootstrap_line(lx, ly, B=B, seed=seed)
    return a_m, (a_lo, a_hi)

def _xp(device: str):
    if device == "cuda":
        if not _HAS_CUPY:
            raise RuntimeError("device=cuda requested but CuPy is not available.")
        return cp
    return np

def _to_numpy(arr):
    if _HAS_CUPY and isinstance(arr, cp.ndarray):  # type: ignore[name-defined]
        return cp.asnumpy(arr)                     # type: ignore[name-defined]
    return np.asarray(arr)

# ------------------------- fast sampler (nS=1) -------------------------

def _sample_Y_batch_nS1(device: str, nE: int, trials: int, batch_size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized sampler for nS=1 (dS=2):
      - Haar state on C^{2 * dE} as (2, dE) array Psi; rho_S = Psi Psi^†.
      - Purity from entries; entropy via eigenvalues (closed-form).
      - A^2 via fidelity with |0⟩.
    Returns:
      X = max(d_eff - 1, EPS),  Y = sqrt(X) * A^2 / Ibits
    """
    xp = _xp(device)
    dS = 2
    dE = 2 ** nE
    remain = trials

    X_chunks: List[np.ndarray] = []
    Y_chunks: List[np.ndarray] = []
    I_chunks: List[np.ndarray] = []

    rng = np.random.default_rng(seed)

    while remain > 0:
        B = min(batch_size, remain)
        remain -= B

        re = rng.normal(size=(B, dS, dE))
        im = rng.normal(size=(B, dS, dE))
        Z = xp.asarray(re + 1j * im)

        # normalize per state
        norms = xp.linalg.norm(Z.reshape(B, -1), axis=1).reshape(B, 1, 1)
        Psi = Z / xp.maximum(norms, xp.asarray(EPS))

        rho = xp.einsum("bie,bje->bij", Psi, Psi.conj())  # (B,2,2)

        a = rho[:, 0, 0].real
        b = rho[:, 0, 1]
        purity = a * a + (1.0 - a) * (1.0 - a) + 2.0 * (b.real * b.real + b.imag * b.imag)
        purity = xp.clip(purity.real, 0.5, 1.0)

        deff = 1.0 / xp.maximum(purity, xp.asarray(EPS))
        X = xp.maximum(deff - 1.0, xp.asarray(EPS))

        s = xp.sqrt(xp.maximum(2.0 * purity - 1.0, 0.0))
        lam1 = (1.0 + s) * 0.5
        Ibits = 2.0 * binary_entropy_bits(_to_numpy(lam1))

        F = np.clip(_to_numpy(a), 0.0, 1.0)
        A2 = np.arccos(np.sqrt(F)) ** 2

        Xh = _to_numpy(X)
        Y = (np.sqrt(Xh) * A2) / np.maximum(Ibits, EPS)

        X_chunks.append(Xh)
        Y_chunks.append(Y)
        I_chunks.append(Ibits)

        if device == "cuda":
            # best-effort free on GPU
            try:
                cp.get_default_memory_pool().free_all_blocks()  # type: ignore[name-defined]
            except Exception:
                pass

    X_all = np.concatenate(X_chunks, axis=0)
    Y_all = np.concatenate(Y_chunks, axis=0)
    I_all = np.concatenate(I_chunks, axis=0)

    mask = (I_all > EPS) & np.isfinite(X_all) & np.isfinite(Y_all) & (Y_all > 0)
    return X_all[mask], Y_all[mask]

def _fit_slope_loglog(X: np.ndarray, Y: np.ndarray):
    mask = (X > 0) & (Y > 0) & np.isfinite(X) & np.isfinite(Y)
    if mask.sum() < 2:
        return float("nan"), float("nan"), float("nan"), 0, np.array([]), np.array([])
    lx = np.log10(X[mask])
    ly = np.log10(Y[mask])
    A = np.column_stack((lx, np.ones_like(lx)))
    m, b = np.linalg.lstsq(A, ly, rcond=None)[0]
    yhat = m * lx + b
    ss_res = float(np.sum((ly - yhat) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2) + EPS)
    R2 = 1.0 - ss_res / ss_tot
    return float(m), float(b), float(R2), int(lx.size), lx, ly

# ------------------------- per-D worker -------------------------

@dataclass
class PerD:
    D: int
    invD: float
    n: int
    alpha: float
    abs_alpha: float
    alpha_R2: float
    varY: float
    meanY: float

def _per_nE_job(nS: int, nE: int, trials: int, base_seed: int, device: str, batch_size: int) -> Tuple[int, Tuple[np.ndarray, np.ndarray], PerD]:
    if nS != 1:
        raise NotImplementedError("Fast path currently supports nS=1 (dS=2).")
    dS = 2 ** nS
    dE = 2 ** nE
    D = dS * dE
    seed = (base_seed ^ (nE * 0x9E3779B1)) & 0xFFFFFFFF
    X, Y = _sample_Y_batch_nS1(device=device, nE=nE, trials=trials, batch_size=batch_size, seed=seed)
    a, b, R2, n, _, _ = _fit_slope_loglog(X, Y)
    row = PerD(
        D=D,
        invD=1.0 / D,
        n=n,
        alpha=a,
        abs_alpha=abs(a),
        alpha_R2=R2,
        varY=float(np.var(Y, ddof=1)),
        meanY=float(np.mean(Y)),
    )
    return D, (X, Y), row

# ------------------------- main sweep -------------------------

def run_sweep_fast(
    nS: int,
    nE_list: List[int],
    trials: int,
    seed: int,
    outdir: Path | str,
    device: str = "auto",
    batch_size: int = 65536,
    workers: int = 0,
) -> Dict[str, object]:

    if device == "auto":
        device = "cuda" if _HAS_CUPY else "cpu"
    if device == "cuda" and not _HAS_CUPY:
        print("[warn] CuPy not found; falling back to CPU.")
        device = "cpu"

    outdir = Path(outdir)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "data").mkdir(parents=True, exist_ok=True)

    print(f"Phase 6 (fast) | nS={nS} | trials={trials} | device={device} | batch={batch_size} | workers={workers or os.cpu_count()}")

    rows: List[PerD] = []
    all_points: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    max_workers = workers if workers > 0 else (os.cpu_count() or 4)
    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for nE in nE_list:
            futures[ex.submit(_per_nE_job, nS, nE, trials, seed, device, batch_size)] = nE
        for fut in as_completed(futures):
            nE = futures[fut]
            try:
                D, (X, Y), row = fut.result()
            except Exception as e:
                raise RuntimeError(f"Worker failed for nE={nE}: {e}") from e
            print(f"  D={D:5d} (nE={nE}), trials={trials} -> n={row.n}, alpha={row.alpha:+.3f}, varY={row.varY:.3e}")
            rows.append(row)
            all_points[D] = (X, Y)

    rows.sort(key=lambda r: r.D)

    # ---- write per-D CSV
    perD_path = outdir / "data" / "phase6_theorem_perD.csv"
    with perD_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames="D,invD,n,alpha,abs_alpha,alpha_R2,varY,meanY".split(","))
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)
    print(f"Wrote {perD_path}")

    # ---- |alpha| vs 1/D (+ CSV)
    invD = np.array([r.invD for r in rows], float)
    absA = np.array([r.abs_alpha for r in rows], float)
    a_line, b_line = linear_fit(invD, absA)
    (a_m, b_m), (a_lo, a_hi), (b_lo, b_hi) = bootstrap_line(invD, absA, B=BOOT_B, seed=(seed ^ 0xA5A5A5A5))

    alpha_csv = outdir / "data" / "phase6_alpha_vs_invD.csv"
    with alpha_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["invD", "abs_alpha"])
        w.writerows(zip(invD.tolist(), absA.tolist()))
    print(f"Wrote {alpha_csv}")

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.scatter(invD, absA, s=40)
    xg = np.linspace(0, invD.max() * 1.05, 200)
    ax.plot(xg, a_line * xg + b_line, "--", label=f"intercept={b_line:+.3f}")
    ax.set_xlabel(r"$1/D$")
    ax.set_ylabel(r"$|\alpha|$")
    ax.set_title(r"$|\alpha|$ vs $1/D$ (Phase VI, fast)")
    ax.legend()
    fig.tight_layout()
    f1 = outdir / "figures" / "phase6_alpha_vs_invD.png"
    fig.savefig(f1, dpi=160)
    plt.close(fig)

    # ---- Var(Y) vs D (+ CSV) on log–log
    Dvals = np.array([r.D for r in rows], float)
    vY = np.array([r.varY for r in rows], float)
    slope_mean, (s_lo, s_hi) = bootstrap_slope_loglog(Dvals, vY, B=BOOT_B, seed=(seed ^ 0x5C5C5C5C))

    var_csv = outdir / "data" / "phase6_varY_by_D.csv"
    with var_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["D", "VarY"])
        w.writerows(zip(Dvals.tolist(), vY.tolist()))
    print(f"Wrote {var_csv}")

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.scatter(Dvals, vY, s=50, label="isotropic (fast)")
    gx = np.logspace(np.log10(Dvals.min() * 0.9), np.log10(Dvals.max() * 1.1), 200)
    a_fit, b_fit = linear_fit(np.log10(Dvals), np.log10(np.clip(vY, EPS, None)))
    gy = 10 ** (a_fit * np.log10(gx) + b_fit)
    ax.plot(gx, gy, "--", label=f"slope={a_fit:+.3f} [{s_lo:+.3f},{s_hi:+.3f}]")
    refy = (vY.max()) * (gx / gx[0]) ** (-2)
    ax.plot(gx, refy, ":", color="gray", alpha=0.7, label=r"$D^{-2}$ reference")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$D$"); ax.set_ylabel(r"$\mathrm{Var}(Y)$")
    ax.set_title(r"Variance scaling of $Y$ vs $D$ (Phase VI, fast)")
    ax.legend()
    fig.tight_layout()
    f2 = outdir / "figures" / "phase6_varY_scaling.png"
    fig.savefig(f2, dpi=160)
    plt.close(fig)

    # ---- histogram at largest D
    Dmax = int(Dvals.max())
    Ymax = all_points[Dmax][1]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.hist(Ymax, bins=80, alpha=0.85, density=True)
    ax.set_xlabel("Y"); ax.set_ylabel("density")
    ax.set_title(f"Distribution of Y at largest D={Dmax}")
    fig.tight_layout()
    f3 = outdir / "figures" / "phase6_Y_hist_Dmax.png"
    fig.savefig(f3, dpi=160)
    plt.close(fig)

    # ---- summary
    summary = {
        "alpha_intercept_estimate": b_line,
        "alpha_intercept_CI": [b_lo, b_hi],
        "alpha_vs_invD_slope": a_line,
        "VarY_loglog_slope": a_fit,
        "VarY_loglog_slope_CI": [s_lo, s_hi],
        "perD_csv": str(perD_path),
        "alpha_vs_invD_csv": str(alpha_csv),
        "varY_by_D_csv": str(var_csv),
        "fig_alpha_vs_invD": str(f1),
        "fig_varY_scaling": str(f2),
        "fig_hist_Dmax": str(f3),
    }
    with (outdir / "phase6_summary.pkl").open("wb") as fh:
        pickle.dump(summary, fh)
    with (outdir / "phase6_summary.txt").open("w") as fh:
        fh.write("Phase VI concentration summary (fast path)\n")
        fh.write(f"alpha intercept @ 1/D->0: {b_line:+.4f}  (bootstrap CI [{b_lo:+.4f},{b_hi:+.4f}])\n")
        fh.write(f"Var(Y) slope (log Var vs log D): {a_fit:+.4f}  (bootstrap CI [{s_lo:+.4f},{s_hi:+.4f}])\n")

    return {
        "rows": rows,
        "alpha_line": (a_line, b_line, (a_lo, a_hi), (b_lo, b_hi)),
        "varY_slope": (slope_mean, (s_lo, s_hi)),
        "paths": {
            "perD": str(perD_path),
            "alpha_csv": str(alpha_csv),
            "var_csv": str(var_csv),
            "alpha_fig": str(f1),
            "var_fig": str(f2),
            "hist_fig": str(f3),
        },
    }

# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 6 — FAST theorem harness & concentration tests")
    ap.add_argument("--nS", type=int, default=DEFAULT_NS, help="system qubits (fast path supports nS=1)")
    ap.add_argument("--nE-min", type=int, default=min(DEFAULT_NE_LIST), help="min environment qubits")
    ap.add_argument("--nE-max", type=int, default=max(DEFAULT_NE_LIST), help="max environment qubits")
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="trials per D")
    ap.add_argument("--seed", type=int, default=RNG_SEED, help="rng seed")
    ap.add_argument("--outdir", default="phase6-out", help="output directory")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="use GPU if available (CuPy)")
    ap.add_argument("--batch-size", type=int, default=65536, help="trials batch for vectorization")
    ap.add_argument("--workers", type=int, default=0, help="processes across nE (0=cpu_count)")
    args = ap.parse_args()

    nE_list = list(range(args.nE_min, args.nE_max + 1))
    run_sweep_fast(
        nS=args.nS,
        nE_list=nE_list,
        trials=args.trials,
        seed=args.seed,
        outdir=args.outdir,
        device=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
    )

if __name__ == "__main__":
    main()
