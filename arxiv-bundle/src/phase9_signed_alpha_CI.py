#!/usr/bin/env python3
# Phase 9 — Purist Haar isometry, accelerated (Haar-correct, not “fast-mode”)
from __future__ import annotations
import argparse, csv, math, os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

EPS = 1e-12

# ---------------- util ----------------
def _rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)

def binary_entropy_bits(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return -(p*np.log2(p) + (1-p)*np.log2(1-p))

def ols_slope(x, y):
    A = np.column_stack((x, np.ones_like(x)))
    a,b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)

def bootstrap_alpha_loglog(X, Y, B, seed):
    X = np.asarray(X); Y = np.asarray(Y)
    m = (X>0) & (Y>0) & np.isfinite(X) & np.isfinite(Y)
    lx = np.log10(X[m]); ly = np.log10(Y[m])
    n = lx.size
    if n < 8:
        return float("nan"), float("nan"), float("nan")
    g = _rng(seed)
    slopes = np.empty(B, float)
    for b in range(B):
        idx = g.integers(0, n, size=n)
        a,_ = ols_slope(lx[idx], ly[idx])
        slopes[b] = a
    mean = float(np.mean(slopes))
    lo,hi = np.percentile(slopes, [2.5, 97.5])
    return mean, float(lo), float(hi)

def bootstrap_line(x, y, B, seed):
    x = np.asarray(x); y = np.asarray(y)
    n = x.size
    if n < 3:
        nan = (float("nan"), float("nan"))
        return nan, nan, nan
    g = _rng(seed)
    A = np.empty(B, float)
    Bv = np.empty(B, float)
    for b in range(B):
        idx = g.integers(0, n, size=n)
        a_hat, b_hat = ols_slope(x[idx], y[idx])
        A[b] = a_hat; Bv[b] = b_hat
    a_mu, b_mu = float(np.mean(A)), float(np.mean(Bv))
    a_lo, a_hi = np.percentile(A, [2.5, 97.5])
    b_lo, b_hi = np.percentile(Bv, [2.5, 97.5])
    return (a_mu, b_mu), (float(a_lo), float(a_hi)), (float(b_lo), float(b_hi))

# ---------------- Haar 2-frame (CPU/GPU) ----------------
def _haar_two_frame(D: int, xp, rg: Optional[object]=None):
    """
    Returns (v0, v1) \in C^D, orthonormal, Haar distributed on the Stiefel V_2(C^D).
    xp = np (CPU) or cupy (GPU). rg: xp.random Generator.
    """
    if rg is None:
        rg = xp.random
    z0 = rg.normal(size=D) + 1j*rg.normal(size=D)
    z1 = rg.normal(size=D) + 1j*rg.normal(size=D)
    v0 = z0 / xp.linalg.norm(z0)
    # Gram–Schmidt
    z1 = z1 - (xp.vdot(v0, z1) * v0)
    n1 = xp.linalg.norm(z1)
    # Extremely rare: numerical underflow; resample z1
    if float(n1.real) < 1e-20:
        z1 = rg.normal(size=D) + 1j*rg.normal(size=D)
        z1 = z1 - (xp.vdot(v0, z1) * v0)
        n1 = xp.linalg.norm(z1)
    v1 = z1 / xp.maximum(n1, 1e-32)
    return v0, v1

def _metrics_from_v0(v0, dS: int, dE: int, xp):
    """
    v0 is the first column of the isometry (shape D=dS*dE).
    Reshape v0 -> (dS, dE) and compute rho_S = M M^† where M[:,e] are columns.
    This is O(dS^2 dE). For dS=2 that’s cheap even at large dE.
    Returns (X, Y) where:
        X = d_eff - 1
        Y = (sqrt(X) * A^2) / Ibits
    """
    M = v0.reshape(dS, dE)
    rhoS = M @ xp.conjugate(M).T                       # (2,2)
    # Purity and fidelity
    purity = xp.real(xp.trace(rhoS @ rhoS))
    purity = float(purity)
    deff = 1.0 / max(purity, EPS)
    X = max(deff - 1.0, EPS)

    a = float(xp.real(rhoS[0,0]))                      # ⟨0|ρ|0⟩
    s = math.sqrt(max(2.0*purity - 1.0, 0.0))
    lam1 = 0.5*(1.0 + s)
    Ibits = 2.0 * float(binary_entropy_bits(np.array([lam1]))[0])
    F = min(max(a, 0.0), 1.0)
    A2 = math.acos(math.sqrt(F))**2
    Y = (math.sqrt(X) * A2) / max(Ibits, EPS)
    return X, Y

def sample_pairs_haar_isometry_purist(nE: int, trials: int, seed: int, device: str="cpu"):
    """
    Purist Haar isometry sampler (exact). Uses Haar two-frame (D×2) via Gram–Schmidt.
    device: 'cpu' (NumPy) or 'gpu' (CuPy). Falls back to CPU if CuPy unavailable.
    """
    dS = 2; dE = 2**nE; D = dS*dE
    use_gpu = (device == "gpu")
    xp = np
    rg = None
    if use_gpu:
        try:
            import cupy as cp
            xp = cp
            rg = cp.random
            # create a per-call seed for CuPy
            cp.random.seed(seed & 0x7fffffff)
        except Exception:
            xp = np
            use_gpu = False

    # CPU RNG
    if not use_gpu:
        g = _rng(seed)

    Xs = []
    Ys = []
    for _ in range(trials):
        if use_gpu:
            v0, v1 = _haar_two_frame(D, xp, rg)   # on GPU
            X, Y = _metrics_from_v0(v0, dS, dE, xp)
        else:
            # CPU path
            z0 = g.normal(size=D) + 1j*g.normal(size=D)
            z1 = g.normal(size=D) + 1j*g.normal(size=D)
            v0 = z0 / np.linalg.norm(z0)
            z1 = z1 - (np.vdot(v0, z1) * v0)
            n1 = np.linalg.norm(z1)
            if n1 < 1e-20:
                z1 = g.normal(size=D) + 1j*g.normal(size=D)
                z1 = z1 - (np.vdot(v0, z1) * v0)
                n1 = np.linalg.norm(z1)
            v1 = z1 / max(n1, 1e-32)
            X, Y = _metrics_from_v0(v0, dS, dE, np)
        if math.isfinite(X) and math.isfinite(Y) and Y >= 0:
            Xs.append(X); Ys.append(Y)

    return np.asarray(Xs, float), np.asarray(Ys, float)

# --------------- Phase 9 outer loop (mostly same as yours) ---------------
@dataclass
class AlphaRow:
    D: int
    invD: float
    trials: int
    alpha: float
    alpha_lo: float
    alpha_hi: float

def _per_seed_job(nE: int, trials: int, seed: int, device: str):
    X, Y = sample_pairs_haar_isometry_purist(nE, trials, seed, device=device)
    mu, lo, hi = bootstrap_alpha_loglog(X, Y, B=1200, seed=seed ^ 0xA5A5A5A5)
    return (nE, seed, mu, lo, hi, X, Y)

def run_phase9_purist(
    nE_list: List[int],
    trials: int,
    seeds_per_D: int,
    boot_B_point: int,
    boot_B_intercept: int,
    outdir: str,
    workers: int,
    device: str,
) -> Dict[str, object]:

    out = Path(outdir); (out/"data").mkdir(parents=True, exist_ok=True); (out/"figures").mkdir(parents=True, exist_ok=True)
    perD: List[AlphaRow] = []
    perD_cloud: Dict[int, Dict[str, np.ndarray]] = {}

    jobs = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for nE in nE_list:
            for k in range(seeds_per_D):
                D = 2 * (2**nE)
                seed = (0xC0FFEE ^ (D*0x9E3779B1) ^ k) & 0xFFFFFFFF
                jobs.append(ex.submit(_per_seed_job, nE, trials, seed, device))
        # collect results grouped by D
        buckets: Dict[int, List[tuple]] = {}
        for fut in as_completed(jobs):
            nE, seed, mu, lo, hi, X, Y = fut.result()
            D = 2 * (2**nE)
            buckets.setdefault(D, []).append((mu, lo, hi, X, Y))

    for D in sorted(buckets.keys()):
        invD = 1.0 / D
        items = buckets[D]
        alphas = [mu for (mu,_,_,_,_) in items if np.isfinite(mu)]
        if not alphas:
            continue
        a_mu = float(np.mean(alphas))
        a_lo, a_hi = np.percentile(alphas, [2.5, 97.5])
        Xcat = np.concatenate([X for (_,_,_,X,_) in items], axis=0)
        Ycat = np.concatenate([Y for (_,_,_,_,Y) in items], axis=0)
        perD.append(AlphaRow(D=D, invD=invD, trials=trials*len(items), alpha=a_mu, alpha_lo=float(a_lo), alpha_hi=float(a_hi)))
        perD_cloud[D] = {"X": Xcat, "Y": Ycat}

    perD.sort(key=lambda r: r.D)

    # write per-D CSV
    perD_csv = out/"data"/"phase9_alpha_perD.csv"
    with perD_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames="D,invD,trials,alpha,alpha_lo,alpha_hi".split(","))
        w.writeheader()
        for r in perD:
            w.writerow(r.__dict__)

    # α CI at largest D
    Dmax = perD[-1].D
    Xmax, Ymax = perD_cloud[Dmax]["X"], perD_cloud[Dmax]["Y"]
    a_mu, a_lo, a_hi = bootstrap_alpha_loglog(Xmax, Ymax, B=boot_B_point, seed=0xABCDEF01)

    # plot α bootstrap at Dmax
    import numpy as _np
    g = _rng(0xABCDEF01)
    m = (Xmax>0) & (Ymax>0)
    lx, ly = _np.log10(Xmax[m]), _np.log10(Ymax[m])
    n = lx.size
    slopes = []
    for _ in range(boot_B_point):
        idx = g.integers(0, n, size=n)
        a,_ = ols_slope(lx[idx], ly[idx])
        slopes.append(a)
    plt.figure(figsize=(6.4,4.0))
    plt.hist(slopes, bins=60, density=True, alpha=0.85)
    plt.axvline(0.0, color="k", ls="--", lw=1)
    plt.title(f"Bootstrap of signed α at largest D={Dmax}\nmean={a_mu:+.3f}, 95% CI [{a_lo:+.3f},{a_hi:+.3f}]")
    plt.xlabel("α"); plt.ylabel("density")
    fig1 = out/"figures"/"phase9_alpha_hist_Dmax.png"
    plt.tight_layout(); plt.savefig(fig1, dpi=160); plt.close()
    pass_alpha = (a_lo <= 0.0 <= a_hi)

    # intercept CI
    invD = np.array([r.invD for r in perD], float)
    alpha_hat = np.array([r.alpha for r in perD], float)
    (slope_mu, intercept_mu), (slope_CI_lo, slope_CI_hi), (b_lo, b_hi) = bootstrap_line(invD, alpha_hat, B=boot_B_intercept, seed=0x13572468)

    xx = np.linspace(0.0, invD.max()*1.05, 200)
    plt.figure(figsize=(7.2,4.6))
    plt.scatter(invD, np.abs(alpha_hat), s=35, label="|α| per D")
    plt.plot(xx, slope_mu*xx + intercept_mu, "--", label=f"intercept={intercept_mu:+.3f} [{b_lo:+.3f},{b_hi:+.3f}]")
    plt.xlabel(r"$1/D$"); plt.ylabel(r"$|\alpha|$")
    plt.title(r"$|\alpha|$ vs $1/D$ — intercept CI")
    plt.legend()
    fig2 = out/"figures"/"phase9_alpha_vs_invD_CI.png"
    plt.tight_layout(); plt.savefig(fig2, dpi=160); plt.close()
    pass_intercept = (b_lo <= 0.0 <= b_hi)

    # summary CSV
    intercept_csv = out/"data"/"phase9_intercept_summary.csv"
    with intercept_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "mode","nS","Dmin","Dmax","points","alpha_Dmax_mean","alpha_Dmax_lo","alpha_Dmax_hi",
            "intercept_mu","intercept_lo","intercept_hi","slope_mu","slope_lo","slope_hi",
            "pass_alpha_CI_zero","pass_intercept_CI_zero"
        ])
        w.writeheader()
        w.writerow({
            "mode": "haar-accelerated", "nS": 1, "Dmin": perD[0].D, "Dmax": Dmax, "points": len(perD),
            "alpha_Dmax_mean": a_mu, "alpha_Dmax_lo": a_lo, "alpha_Dmax_hi": a_hi,
            "intercept_mu": intercept_mu, "intercept_lo": b_lo, "intercept_hi": b_hi,
            "slope_mu": slope_mu, "slope_lo": slope_CI_lo, "slope_hi": slope_CI_hi,
            "pass_alpha_CI_zero": bool(pass_alpha), "pass_intercept_CI_zero": bool(pass_intercept),
        })

    print("\n=== Phase 9 (haar-accelerated) ===")
    print(f"largest D={Dmax}:  α mean = {a_mu:+.4f}, 95% CI [{a_lo:+.4f},{a_hi:+.4f}]  -> {'PASS' if pass_alpha else 'FAIL'}")
    print(f"α vs 1/D intercept: b = {intercept_mu:+.4f}, 95% CI [{b_lo:+.4f},{b_hi:+.4f}]  -> {'PASS' if pass_intercept else 'FAIL'}")
    print(f"Wrote: {perD_csv}")
    print(f"Wrote: {intercept_csv}")
    print(f"Saved: {fig1}")
    print(f"Saved: {fig2}")

    return {
        "perD_csv": str(perD_csv),
        "intercept_csv": str(intercept_csv),
        "figures": {"alpha_hist_Dmax": str(fig1), "alpha_vs_invD": str(fig2)},
    }

# ---------------- CLI ----------------
def parse_ne_list(s: str) -> List[int]:
    if "," in s:
        parts = []
        for tok in s.split(","):
            tok = tok.strip()
            if "-" in tok:
                a,b = tok.split("-")
                parts.extend(list(range(int(a), int(b)+1)))
            else:
                parts.append(int(tok))
        return sorted(set(parts))
    if "-" in s:
        a,b = s.split("-")
        return list(range(int(a), int(b)+1))
    return [int(s)]

def main():
    ap = argparse.ArgumentParser(description="Phase 9 — Haar isometry (purist) accelerated with GPU/CPU workers")
    ap.add_argument("--nE", type=str, default="7-10", help="environment qubits (e.g. 7-10)")
    ap.add_argument("--trials", type=int, default=1500, help="trials per D per seed")
    ap.add_argument("--seeds-per-D", type=int, default=3, help="independent repeats")
    ap.add_argument("--boot-B-point", type=int, default=2000, help="bootstrap samples at fixed D")
    ap.add_argument("--boot-B-intercept", type=int, default=5000, help="bootstrap samples for intercept")
    ap.add_argument("--outdir", default="phase9-out", help="output dir")
    ap.add_argument("--workers", type=int, default=8, help="CPU processes")
    ap.add_argument("--device", choices=["cpu","gpu"], default="cpu", help="NumPy (cpu) or CuPy (gpu)")
    args = ap.parse_args()

    nE_list = parse_ne_list(args.nE)
    run_phase9_purist(
        nE_list=nE_list,
        trials=args.trials,
        seeds_per_D=args.seeds_per_D,
        boot_B_point=args.boot_B_point,
        boot_B_intercept=args.boot_B_intercept,
        outdir=args.outdir,
        workers=args.workers,
        device=args.device,
    )

if __name__ == "__main__":
    main()
