#!/usr/bin/env python3
# Phase 6: Theorem harness — concentration under 2-design/isotropic dynamics
# Produces CSVs + figures for: Var(Y) ~ D^beta, |alpha| vs 1/D, histograms.

from __future__ import annotations
import argparse, csv, math, pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

# ------------------------- numerics config -------------------------

EPS = 1e-12

# Reasonable defaults (tune with CLI):
DEFAULT_NS = 1                 # system qubits
DEFAULT_NE_LIST = [5,6,7,8,9,10,11]  # environment qubits -> D = 2^(nS+nE)
DEFAULT_TRIALS = 4000
BOOT_B = 3000
RNG = np.random.default_rng(0xC0FFEE)

# ------------------------- linear algebra -------------------------

def dagger(A: np.ndarray) -> np.ndarray:
    return A.conj().T

def dm(psi: np.ndarray) -> np.ndarray:
    psi = psi.reshape(-1,1)
    return psi @ psi.conj().T

def eigh_psd(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H = (M + dagger(M)) / 2
    w,V = np.linalg.eigh(H)
    w = np.clip(w.real, 0.0, None)
    return w, V

def sqrtm_psd(M: np.ndarray) -> np.ndarray:
    w, V = eigh_psd(M)
    return V @ np.diag(np.sqrt(w + 0.0)) @ dagger(V)

def vn_entropy_bits(rho: np.ndarray, tol: float=1e-12) -> float:
    w,_ = eigh_psd(rho)
    w = w[w>tol]
    return float((-w*np.log2(w)).sum()) if w.size else 0.0

def purity(rho: np.ndarray) -> float:
    return float(np.trace(rho @ rho).real)

def haar_state(dim: int, rng=RNG) -> np.ndarray:
    z = rng.normal(size=dim) + 1j*rng.normal(size=dim)
    z /= np.linalg.norm(z)
    return z

def haar_unitary(dim: int, rng=RNG) -> np.ndarray:
    X = rng.normal(size=(dim,dim)) + 1j*rng.normal(size=(dim,dim))
    Q,R = np.linalg.qr(X)
    phase = np.diag(R)/np.abs(np.diag(R))
    return Q @ np.diag(phase.conj())

def fidelity_uhlmann(rho: np.ndarray, sigma: np.ndarray) -> float:
    s = sqrtm_psd(rho) @ sigma @ sqrtm_psd(rho)
    return float(np.trace(sqrtm_psd(s)).real**2)

def partial_trace(rho: np.ndarray, keep: Iterable[int], dims: List[int]) -> np.ndarray:
    dims = list(dims); n = len(dims)
    keep = sorted(keep); trace = [i for i in range(n) if i not in keep]
    perm = keep + trace + [i+n for i in keep] + [i+n for i in trace]
    resh = rho.reshape(*(dims+dims)).transpose(perm)
    d_keep = int(np.prod([dims[i] for i in keep])) if keep else 1
    d_trace= int(np.prod([dims[i] for i in trace])) if trace else 1
    resh = resh.reshape(d_keep, d_trace, d_keep, d_trace)
    return np.einsum("aibi->ab", resh)

# ------------------------- invariant pieces -------------------------

def metrics_isotropic_step(nS:int, nE:int, rng=RNG) -> Tuple[float,float,float]:
    """
    One isotropic step via Haar on S⊗E.
    Start in |psi_S>⊗|0_E>; apply Haar U on S⊗E; trace E.
    Returns Ibits, A^2, d_eff for the reduced state of S.
    """
    dS = 2**nS; dE = 2**nE; D = dS*dE
    # initial
    psiS0 = haar_state(dS, rng)
    psiE0 = np.zeros(dE, complex); psiE0[0] = 1.0
    psi0  = np.kron(psiS0, psiE0)  # |psiS>⊗|0>
    U = haar_unitary(D, rng)
    psi1 = U @ psi0
    rhoSE1 = dm(psi1)
    rhoS0 = dm(psiS0)
    rhoS1 = partial_trace(rhoSE1, keep=[0], dims=[dS,dE])
    # Because global is pure: I = 2 S(rhoS1)
    Sbits = vn_entropy_bits(rhoS1)
    Ibits = 2.0 * Sbits
    F = fidelity_uhlmann(rhoS0, rhoS1)
    A2 = float(np.arccos(np.sqrt(np.clip(F,0.0,1.0)))**2)
    deff = 1.0/max(purity(rhoS1), EPS)
    return Ibits, A2, deff

def sample_Y_batch(nS:int, nE:int, trials:int, rng=RNG) -> Tuple[np.ndarray, np.ndarray]:
    Xs, Ys = [], []
    for _ in range(trials):
        Ibits, A2, deff = metrics_isotropic_step(nS, nE, rng)
        if Ibits <= EPS:  # skip degenerate
            continue
        x = max(deff - 1.0, EPS)
        y = math.sqrt(x) * A2 / Ibits
        Xs.append(x); Ys.append(y)
    return np.asarray(Xs), np.asarray(Ys)

# ------------------------- stats helpers -------------------------

def fit_slope_loglog(X: np.ndarray, Y: np.ndarray):
    mask = (X>0) & (Y>0)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan, 0, np.array([]), np.array([])
    lx = np.log10(X[mask]); ly = np.log10(Y[mask])
    A = np.column_stack((lx, np.ones_like(lx)))
    m,b = np.linalg.lstsq(A, ly, rcond=None)[0]
    # R^2 (on log space)
    yhat = m*lx + b
    ss_res = np.sum((ly - yhat)**2)
    ss_tot = np.sum((ly - ly.mean())**2) + EPS
    R2 = 1.0 - ss_res/ss_tot
    return float(m), float(b), float(R2), int(lx.size), lx, ly

def linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float,float]:
    A = np.column_stack((x, np.ones_like(x)))
    a,b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)

def bootstrap_line(x: np.ndarray, y: np.ndarray, B:int=1000, rng=RNG) -> Tuple[Tuple[float,float], Tuple[float,float], Tuple[float,float]]:
    n=len(x); a_s=[]; b_s=[]
    for _ in range(B):
        idx = rng.integers(0,n,size=n)
        a,b = linear_fit(x[idx], y[idx])
        a_s.append(a); b_s.append(b)
    a_s = np.array(a_s); b_s=np.array(b_s)
    ci = (np.percentile(a_s,[2.5,97.5]), np.percentile(b_s,[2.5,97.5]))
    return (a_s.mean(), b_s.mean()), (ci[0][0], ci[0][1]), (ci[1][0], ci[1][1])

def bootstrap_slope_loglog(xD: np.ndarray, vY: np.ndarray, B:int=1000, rng=RNG):
    # fit log Var vs log D
    lx = np.log10(xD); ly = np.log10(np.maximum(vY, EPS))
    (a,b), (a_lo,a_hi), _ = bootstrap_line(lx, ly, B=B, rng=rng)
    return a, (a_lo, a_hi)

# ------------------------- main sweep -------------------------

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

def run_sweep(
    nS:int=DEFAULT_NS,
    nE_list:List[int]=DEFAULT_NE_LIST,
    trials:int=DEFAULT_TRIALS,
    seed:int=0xC0FFEE,
    outdir:Path|str="phase6-out"
) -> Dict[str,object]:
    rng = np.random.default_rng(seed)
    outdir = Path(outdir); (outdir/"figures").mkdir(parents=True, exist_ok=True); (outdir/"data").mkdir(parents=True, exist_ok=True)

    rows: List[PerD] = []
    all_points: Dict[int, Tuple[np.ndarray,np.ndarray]] = {}

    print("Phase 6 sweep (isotropic Haar U on S⊗E):")
    for nE in nE_list:
        dS = 2**nS; dE = 2**nE; D = dS*dE
        print(f"  D={D:5d} (nS={nS}, nE={nE}), trials={trials} ...")
        X, Y = sample_Y_batch(nS, nE, trials, rng)
        a,b,R2,n,lx,ly = fit_slope_loglog(X,Y)
        rows.append(PerD(D=D, invD=1.0/D, n=n, alpha=a, abs_alpha=abs(a), alpha_R2=R2, varY=float(np.var(Y)), meanY=float(np.mean(Y))))
        all_points[D] = (X,Y)

    # ---- write CSV
    perD_path = outdir/"data"/"phase6_theorem_perD.csv"
    with perD_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames="D,invD,n,alpha,abs_alpha,alpha_R2,varY,meanY".split(","))
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)
    print(f"Wrote {perD_path}")

    # ---- |alpha| vs 1/D
    invD = np.array([r.invD for r in rows], float)
    absA = np.array([r.abs_alpha for r in rows], float)
    a_line, b_line = linear_fit(invD, absA)
    (a_m,b_m), (a_lo,a_hi), (b_lo,b_hi) = bootstrap_line(invD, absA, B=BOOT_B, rng=rng)

    fig,ax = plt.subplots(figsize=(7.0,4.6))
    ax.scatter(invD, absA, s=40)
    xg = np.linspace(0, invD.max()*1.05, 200)
    ax.plot(xg, a_line*xg + b_line, '--', label=f"intercept={b_line:+.3f}")
    ax.set_xlabel(r"$1/D$")
    ax.set_ylabel(r"$|\alpha|$")
    ax.set_title(r"$|\alpha|$ vs $1/D$ (Phase VI)")
    ax.legend()
    fig.tight_layout()
    f1 = outdir/"figures"/"phase6_alpha_vs_invD.png"
    fig.savefig(f1, dpi=160); plt.close(fig)

    # ---- Var(Y) vs D (log-log)
    Dvals = np.array([r.D for r in rows], float)
    vY    = np.array([r.varY for r in rows], float)
    slope, (s_lo, s_hi) = bootstrap_slope_loglog(Dvals, vY, B=BOOT_B, rng=rng)

    fig,ax = plt.subplots(figsize=(7.6,4.8))
    ax.scatter(Dvals, vY, s=50, label="isotropic")
    gx = np.logspace(np.log10(Dvals.min()*0.9), np.log10(Dvals.max()*1.1), 200)
    # fit line in log space:
    a,b = linear_fit(np.log10(Dvals), np.log10(vY+EPS))
    gy = 10**(a*np.log10(gx) + b)
    ax.plot(gx, gy, '--', label=f"slope={a:+.3f} [{s_lo:+.3f},{s_hi:+.3f}]")
    # D^-2 reference
    ref = (Dvals.min(), vY.max())
    refx = gx
    refy = (ref[1]) * (refx/refx[0])**(-2)
    ax.plot(refx, refy, ':', color='gray', alpha=0.7, label=r"$D^{-2}$ reference")
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r"$D$"); ax.set_ylabel(r"$\mathrm{Var}(Y)$")
    ax.set_title(r"Variance scaling of $Y$ vs $D$ (Phase VI)")
    ax.legend()
    fig.tight_layout()
    f2 = outdir/"figures"/"phase6_varY_scaling.png"
    fig.savefig(f2, dpi=160); plt.close(fig)

    # ---- optional histogram at largest D
    Dmax = int(Dvals.max()); Ymax = all_points[Dmax][1]
    fig,ax = plt.subplots(figsize=(6.4,4.0))
    ax.hist(Ymax, bins=80, alpha=0.85, density=True)
    ax.set_xlabel("Y"); ax.set_ylabel("density")
    ax.set_title(f"Distribution of Y at largest D={Dmax}")
    fig.tight_layout()
    f3 = outdir/"figures"/"phase6_Y_hist_Dmax.png"
    fig.savefig(f3, dpi=160); plt.close(fig)

    # ---- save quick summary
    summary = {
        "alpha_intercept_estimate": b_line,
        "alpha_intercept_CI": [b_lo, b_hi],
        "alpha_vs_invD_slope": a_line,
        "VarY_loglog_slope": a,
        "VarY_loglog_slope_CI": [s_lo, s_hi],
        "perD_csv": str(perD_path),
        "fig_alpha_vs_invD": str(f1),
        "fig_varY_scaling": str(f2),
        "fig_hist_Dmax": str(f3),
    }
    with (outdir/"phase6_summary.pkl").open("wb") as fh:
        pickle.dump(summary, fh)
    print("Summary:", summary)

    # human-readable txt
    with (outdir/"phase6_summary.txt").open("w") as fh:
        fh.write("Phase VI concentration summary\n")
        fh.write(f"alpha intercept @ 1/D->0: {b_line:+.4f}  (bootstrap CI [{b_lo:+.4f},{b_hi:+.4f}])\n")
        fh.write(f"Var(Y) slope (log Var vs log D): {a:+.4f}  (bootstrap CI [{s_lo:+.4f},{s_hi:+.4f}])\n")

    return {
        "rows": rows,
        "alpha_line": (a_line, b_line, (a_lo,a_hi), (b_lo,b_hi)),
        "varY_slope": (a, (s_lo,s_hi)),
        "paths": {"perD": perD_path, "alpha_fig": f1, "vary_fig": f2, "hist_fig": f3},
    }

# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 6 — theorem harness & concentration tests")
    ap.add_argument("--nS", type=int, default=DEFAULT_NS, help="system qubits")
    ap.add_argument("--nE-min", type=int, default=min(DEFAULT_NE_LIST), help="min environment qubits")
    ap.add_argument("--nE-max", type=int, default=max(DEFAULT_NE_LIST), help="max environment qubits")
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="trials per D")
    ap.add_argument("--seed", type=int, default=0xC0FFEE, help="rng seed")
    ap.add_argument("--outdir", default="phase6-out", help="output directory")
    args = ap.parse_args()

    nE_list = list(range(args.nE_min, args.nE_max+1))
    run_sweep(nS=args.nS, nE_list=nE_list, trials=args.trials, seed=args.seed, outdir=args.outdir)

if __name__ == "__main__":
    main()
