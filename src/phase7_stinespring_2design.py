#!/usr/bin/env python3
"""
Phase VII: Stinespring-2-design confirmation (Haar isometry)

What this does
--------------
* Draws true Haar-random isometries V : C^{d_S} -> C^{d_S d_E}
  by QR on Ginibre matrices (columns are orthonormal).
* Applies one-step Stinespring dynamics |psi_S><psi_S| -> rho'_S = Tr_E[V |psi><psi| V^\dagger].
  We use |psi_S> = |0...0> (pure); under a Haar isometry the output is isotropic.
* Computes the curvature–information invariant:
      Y = sqrt(d_eff - 1) * A^2 / I,
  with A^2 the Bures/Uhlmann angle between rho_S and rho'_S, and I = 2 S(rho'_S)
  (since the joint state V|psi> is pure).
* For each D = d_S d_E, aggregates:
  - alpha (slope of log Y vs log(deff - 1))
  - Var(Y), mean(Y)
* Fits:
  - |alpha| vs 1/D  (OLS + bootstrap CI for intercept)
  - Var(Y) vs D     (log-log slope + bootstrap CI)

Outputs
-------
  data/phase7_perD.csv               : one row per D with alpha, Var(Y), ...
  data/phase7_alpha_vs_invD.csv      : points used in |alpha| vs 1/D
  data/phase7_varY_by_D.csv          : points used in Var(Y) vs D
  figures/phase7_alpha_vs_invD.png
  figures/phase7_varY_scaling.png
  figures/phase7_Y_hist_Dmax.png
  phase7_summary.txt / .pkl          : key estimates and CIs

Notes
-----
* This is a *true Haar isometry* model of the Stinespring channel (stronger than a 2-design).
* If you want an exact unitary 2-design instead of Haar, replace `haar_isometry` with a Clifford
  isometry sampler (requires an external Clifford library).
* GPU execution is available with `--device cuda`; the QR and partial-trace steps are promoted to CuPy while
  scalar diagnostics remain on the host. GPU mode forces `workers=1` to avoid oversubscribing the device.
"""

from __future__ import annotations
import argparse, csv, math, os, pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

try:  # optional GPU backend
    import cupy as cp  # type: ignore
    _HAS_CUPY = True
except Exception:  # pragma: no cover - CuPy not available
    cp = None
    _HAS_CUPY = False

# ---------------------------- numerics ----------------------------

EPS = 1e-12
RNG_SEED = 0xA57A2D17

DEFAULT_NS = 1
DEFAULT_NE_MIN = 6
DEFAULT_NE_MAX = 14
DEFAULT_TRIALS = 1200
BOOT_B = 1500

# --------------------------- LA helpers ---------------------------

def dagger(A):
    return A.conj().T

def _asnumpy(arr):
    if _HAS_CUPY and cp is not None and isinstance(arr, cp.ndarray):  # type: ignore
        return cp.asnumpy(arr)  # type: ignore
    return np.asarray(arr)

def partial_trace(rho, keep: Iterable[int], dims: List[int], xp=np):
    """
    Partial trace over all subsystems not in `keep`.
    dims: list of local dimensions of each subsystem.
    rho : density matrix on ⊗ dims.
    """
    keep = sorted(keep)
    n = len(dims)
    trace = [i for i in range(n) if i not in keep]
    perm = keep + trace + [i + n for i in keep] + [i + n for i in trace]
    resh = rho.reshape(*(dims + dims)).transpose(perm)
    d_keep = int(np.prod([dims[i] for i in keep])) if keep else 1
    d_trace = int(np.prod([dims[i] for i in trace])) if trace else 1
    resh = resh.reshape(d_keep, d_trace, d_keep, d_trace)
    return xp.einsum("aibi->ab", resh)

def vn_entropy_bits(rho: np.ndarray, tol: float = 1e-12) -> float:
    w, _ = np.linalg.eigh((rho + dagger(rho)) / 2)
    w = np.clip(np.real(w), 0.0, None)
    w = w[w > tol]
    return float((-w * np.log2(w)).sum()) if w.size else 0.0

def purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))

# ---------------------- Haar isometry sampler ---------------------

def haar_isometry(d_out: int, d_in: int, xp, rng_cpu: np.random.Generator, rng_gpu=None):
    """
    Haar-random isometry V (d_out x d_in) with V^\dagger V = I_{d_in}.
    Construct by QR on a Ginibre matrix.
    """
    if xp is np:
        Z = rng_cpu.normal(size=(d_out, d_in)) + 1j * rng_cpu.normal(size=(d_out, d_in))
    else:
        rng = rng_gpu if rng_gpu is not None else xp.random.default_rng()
        Z = rng.standard_normal(size=(d_out, d_in)) + 1j * rng.standard_normal(size=(d_out, d_in))
    Q, R = xp.linalg.qr(Z)  # Q is d_out x d_in with orthonormal columns
    phases = xp.diag(R)
    phases = phases / xp.maximum(xp.abs(phases), xp.asarray(1e-12, dtype=phases.dtype))
    V = Q @ xp.diag(xp.conj(phases))
    return V

# -------------------- Curvature–information Y ---------------------

def metrics_from_isometry(V, nS: int, nE: int, xp) -> Tuple[float, float]:
    """
    One Stinespring step with input rho_S = |0...0><0...0|.
    Returns (X, Y) with:
       X = max(d_eff - 1, EPS), d_eff = 1 / Tr(rho_S'^2)
       Y = sqrt(X) * A^2 / I,     A^2 = arccos^2( sqrt(F) ),  F = <0|rho_S'|0>
       I = 2 S(rho_S')  (since the joint state is pure)
    """
    dS = 2 ** nS
    dE = 2 ** nE


    # |0_S><0_S|
    e0 = xp.zeros((dS, 1), dtype=xp.complex128)
    e0[0, 0] = 1.0 + 0.0j
    rhoS0 = e0 @ e0.conj().T

    # |Psi_SE> = V |0_S>  -> rho_SE = |Psi><Psi|
    psiSE = V @ e0  # (dS*dE x 1)
    psi_mat = psiSE.reshape(dS, dE)
    rhoS1 = psi_mat @ dagger(psi_mat)
    rhoS1_np = _asnumpy(rhoS1)

    # d_eff
    deff = 1.0 / max(purity(rhoS1_np), EPS)
    X = max(deff - 1.0, EPS)

    # A^2 via fidelity to |0> (since rho_S0 is pure |0><0|)
    F = float(np.real(rhoS1_np[0, 0]))
    F = float(np.clip(F, 0.0, 1.0))
    A2 = float(np.arccos(np.sqrt(F)) ** 2)

    # I = 2 S(rho_S')
    Ibits = 2.0 * vn_entropy_bits(rhoS1_np)

    Y = (math.sqrt(X) * A2) / max(Ibits, EPS)
    return X, Y

# ------------------------ stats & fits ----------------------------

def fit_slope_loglog(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float, float, int, np.ndarray, np.ndarray]:
    mask = (X > 0) & (Y > 0) & np.isfinite(X) & np.isfinite(Y)
    if mask.sum() < 2:
        return float("nan"), float("nan"), float("nan"), int(mask.sum()), np.array([]), np.array([])
    lx = np.log10(X[mask])
    ly = np.log10(Y[mask])
    A = np.column_stack((lx, np.ones_like(lx)))
    m, b = np.linalg.lstsq(A, ly, rcond=None)[0]
    yhat = m * lx + b
    ss_res = float(np.sum((ly - yhat) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2)) + EPS
    R2 = 1.0 - ss_res / ss_tot
    return float(m), float(b), float(R2), int(lx.size), lx, ly

def linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    A = np.column_stack((x, np.ones_like(x)))
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)

def bootstrap_line(x: np.ndarray, y: np.ndarray, B: int, rng: np.random.Generator) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    n = len(x)
    a_s = np.empty(B)
    b_s = np.empty(B)
    ones = np.ones(n)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        A = np.column_stack((x[idx], ones))
        a_s[b], b_s[b] = np.linalg.lstsq(A, y[idx], rcond=None)[0]
    (a_lo, a_hi) = np.percentile(a_s, [2.5, 97.5])
    (b_lo, b_hi) = np.percentile(b_s, [2.5, 97.5])
    return (float(a_s.mean()), float(b_s.mean())), (float(a_lo), float(a_hi)), (float(b_lo), float(b_hi))

def bootstrap_slope_loglog(Dvals: np.ndarray, vY: np.ndarray, B: int, rng: np.random.Generator) -> Tuple[float, Tuple[float, float]]:
    lx = np.log10(Dvals)
    ly = np.log10(np.maximum(vY, EPS))
    (a_m, _), (a_lo, a_hi), _ = bootstrap_line(lx, ly, B=B, rng=rng)
    return float(a_m), (float(a_lo), float(a_hi))

# ------------------------ per-D worker ---------------------------

@dataclass
class RowPerD:
    D: int
    invD: float
    n: int
    alpha: float
    abs_alpha: float
    alpha_R2: float
    varY: float
    meanY: float

def _perD_job(nS: int, nE: int, trials: int, seed: int, device: str) -> Tuple[int, np.ndarray, np.ndarray, RowPerD]:
    dS = 2 ** nS
    dE = 2 ** nE
    D = dS * dE
    rng_cpu = np.random.default_rng(seed ^ (nE * 0x9E3779B1))
    if device == "cuda":
        if not _HAS_CUPY:
            raise RuntimeError("device=cuda requested but CuPy not available")
        xp = cp  # type: ignore
        rng_gpu = cp.random.default_rng(seed ^ (nE * 0x6C8E9CF5))  # type: ignore
    else:
        xp = np
        rng_gpu = None

    Xs: List[float] = []
    Ys: List[float] = []

    for _ in range(trials):
        V = haar_isometry(dS * dE, dS, xp, rng_cpu, rng_gpu)
        X, Y = metrics_from_isometry(V, nS, nE, xp)
        if np.isfinite(X) and np.isfinite(Y) and X > 0 and Y > 0:
            Xs.append(X)
            Ys.append(Y)

    X = np.asarray(Xs, float)
    Y = np.asarray(Ys, float)
    a, b, R2, n, _, _ = fit_slope_loglog(X, Y)

    row = RowPerD(
        D=D,
        invD=1.0 / D,
        n=n,
        alpha=float(a),
        abs_alpha=float(abs(a)),
        alpha_R2=float(R2),
        varY=float(np.var(Y)),
        meanY=float(np.mean(Y)),
    )
    return D, X, Y, row

# ---------------------------- driver ------------------------------

def run_phase7(
    nS: int,
    nE_min: int,
    nE_max: int,
    trials: int,
    workers: int,
    outdir: Path | str,
    seed: int = RNG_SEED,
    device: str = "cpu",
) -> Dict[str, object]:
    outdir = Path(outdir)
    (outdir / "data").mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    if device == "cuda" and not _HAS_CUPY:
        raise RuntimeError("CuPy is required for device='cuda'. Install cupy-cuda11x (or matching wheel).")
    if device == "cuda" and workers != 1:
        print("[Phase VII] device=cuda forces workers=1 (serial GPU execution).")
        workers = 1

    nE_list = list(range(nE_min, nE_max + 1))
    print(f"[Phase VII] nS={nS} | nE={nE_list} | trials={trials} | workers={workers or os.cpu_count()} | device={device}")

    rows: List[RowPerD] = []
    points: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    if device == "cuda":
        for nE in nE_list:
            D, X, Y, row = _perD_job(nS, nE, trials, seed, device)
            print(f"  D={D:6d} (nE={nE}) -> n={row.n:5d}, alpha={row.alpha:+.4f}, Var(Y)={row.varY:.3e}")
            rows.append(row)
            points[D] = (X, Y)
    else:
        max_workers = workers if workers > 0 else (os.cpu_count() or 4)
        futures = {}
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for nE in nE_list:
                futures[ex.submit(_perD_job, nS, nE, trials, seed, device)] = nE
            for fut in as_completed(futures):
                nE = futures[fut]
                try:
                    D, X, Y, row = fut.result()
                except Exception as e:
                    raise RuntimeError(f"Worker failed for nE={nE}: {e}") from e
                print(f"  D={D:6d} (nE={nE}) -> n={row.n:5d}, alpha={row.alpha:+.4f}, Var(Y)={row.varY:.3e}")
                rows.append(row)
                points[D] = (X, Y)

    rows.sort(key=lambda r: r.D)
    Dvals = np.array([r.D for r in rows], float)
    invD = np.array([r.invD for r in rows], float)
    absA = np.array([r.abs_alpha for r in rows], float)
    vY = np.array([r.varY for r in rows], float)

    # --------------------- CSV dumps ---------------------
    perD_csv = outdir / "data" / "phase7_perD.csv"
    with perD_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames="D,invD,n,alpha,abs_alpha,alpha_R2,varY,meanY".split(","))
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)

    alpha_csv = outdir / "data" / "phase7_alpha_vs_invD.csv"
    with alpha_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["invD", "abs_alpha"])
        for r in rows:
            w.writerow([r.invD, r.abs_alpha])

    vary_csv = outdir / "data" / "phase7_varY_by_D.csv"
    with vary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["D", "VarY"])
        for r in rows:
            w.writerow([r.D, r.varY])

    # --------------------- |alpha| vs 1/D ----------------
    a_line, b_line = linear_fit(invD, absA)
    (a_m, b_m), (a_lo, a_hi), (b_lo, b_hi) = bootstrap_line(
        invD, absA, B=BOOT_B, rng=np.random.default_rng(seed ^ 0xABCDEF01)
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(invD, absA, s=48)
    xg = np.linspace(0, invD.max() * 1.05, 256)
    ax.plot(xg, a_line * xg + b_line, "--", label=f"intercept={b_line:+.3f}")
    ax.set_xlabel(r"$1/D$")
    ax.set_ylabel(r"$|\alpha|$")
    ax.set_title(r"$|\alpha|$ vs $1/D$ (Phase VII, Haar isometry)")
    ax.legend()
    fig.tight_layout()
    alpha_fig = outdir / "figures" / "phase7_alpha_vs_invD.png"
    fig.savefig(alpha_fig, dpi=160)
    plt.close(fig)

    # --------------------- Var(Y) vs D -------------------
    slope_mle, (s_lo, s_hi) = bootstrap_slope_loglog(
        Dvals, vY, B=BOOT_B, rng=np.random.default_rng(seed ^ 0x55AA55AA)
    )
    # Plot line using OLS on (log10 D, log10 VarY) for a clean caption
    ols_a, ols_b = linear_fit(np.log10(Dvals), np.log10(np.maximum(vY, EPS)))
    gx = np.logspace(np.log10(Dvals.min() * 0.9), np.log10(Dvals.max() * 1.1), 256)
    gy = 10 ** (ols_a * np.log10(gx) + ols_b)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.scatter(Dvals, vY, s=60, label="Haar isometry")
    ax.plot(gx, gy, "--", label=f"slope={ols_a:+.3f} [{s_lo:+.3f},{s_hi:+.3f}]")
    refy = (vY.max()) * (gx / gx[0]) ** (-2)
    ax.plot(gx, refy, ":", color="gray", alpha=0.7, label=r"$D^{-2}$ reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$D$")
    ax.set_ylabel(r"$\mathrm{Var}(Y)$")
    ax.set_title(r"Variance scaling of $Y$ vs $D$ (Phase VII, Haar isometry)")
    ax.legend()
    fig.tight_layout()
    vary_fig = outdir / "figures" / "phase7_varY_scaling.png"
    fig.savefig(vary_fig, dpi=160)
    plt.close(fig)

    # ----------------- histogram at largest D -------------
    Dmax = int(Dvals.max())
    Ymax = points[Dmax][1]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(Ymax, bins=80, density=True, alpha=0.85)
    ax.set_xlabel("Y")
    ax.set_ylabel("density")
    ax.set_title(f"Distribution of Y at largest D={Dmax}")
    fig.tight_layout()
    hist_fig = outdir / "figures" / "phase7_Y_hist_Dmax.png"
    fig.savefig(hist_fig, dpi=160)
    plt.close(fig)

    # --------------------- summary -----------------------
    summary = {
        "alpha_line_OLS": {"slope": a_line, "intercept": b_line},
        "alpha_line_boot": {
            "slope_mean": a_m,
            "intercept_mean": b_m,
            "slope_CI": [a_lo, a_hi],
            "intercept_CI": [b_lo, b_hi],
        },
        "VarY_slope_loglog": {"slope_boot_mean": slope_mle, "slope_OLS": ols_a, "CI": [s_lo, s_hi]},
        "perD_csv": str(perD_csv),
        "alpha_csv": str(alpha_csv),
        "varY_csv": str(vary_csv),
        "alpha_fig": str(alpha_fig),
        "varY_fig": str(vary_fig),
        "hist_fig": str(hist_fig),
    }
    with (outdir / "phase7_summary.pkl").open("wb") as fh:
        pickle.dump(summary, fh)
    with (outdir / "phase7_summary.txt").open("w") as fh:
        fh.write("Phase VII (Haar isometry) summary\n")
        fh.write(f"|alpha| vs 1/D intercept (OLS)   : {b_line:+.4f}\n")
        fh.write(f"|alpha| vs 1/D intercept (boot)  : {b_m:+.4f}  CI [{summary['alpha_line_boot']['intercept_CI'][0]:+.4f},{summary['alpha_line_boot']['intercept_CI'][1]:+.4f}]\n")
        fh.write(f"Var(Y) vs D slope (OLS)          : {ols_a:+.4f}\n")
        fh.write(f"Var(Y) vs D slope (boot mean)    : {slope_mle:+.4f}  CI [{s_lo:+.4f},{s_hi:+.4f}]\n")

    print(f"[Phase VII] CSVs: {perD_csv}, {alpha_csv}, {vary_csv}")
    print(f"[Phase VII] FIGs: {alpha_fig.name}, {vary_fig.name}, {hist_fig.name}")
    return summary

# ----------------------------- CLI --------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase VII — Haar Stinespring 2-design confirmation")
    ap.add_argument("--nS", type=int, default=DEFAULT_NS, help="system qubits (typically small, e.g. 1)")
    ap.add_argument("--nE-min", type=int, default=DEFAULT_NE_MIN, help="min environment qubits")
    ap.add_argument("--nE-max", type=int, default=DEFAULT_NE_MAX, help="max environment qubits")
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="trials (isometries) per D")
    ap.add_argument("--workers", type=int, default=0, help="processes across nE (0 => cpu_count)")
    ap.add_argument("--outdir", default="phase7-out", help="output directory")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="backend device (default: cpu)")
    ap.add_argument("--seed", type=int, default=RNG_SEED, help="base RNG seed")
    args = ap.parse_args()

    run_phase7(
        nS=args.nS,
        nE_min=args.nE_min,
        nE_max=args.nE_max,
        trials=args.trials,
        workers=(args.workers if args.workers > 0 else (os.cpu_count() or 4)),
        outdir=args.outdir,
        seed=args.seed,
        device=args.device,
    )

if __name__ == "__main__":
    main()
