#!/usr/bin/env python3
# Phase 9+ — Purist Haar (accelerated) + Fast sampler, signed/|α|/α², WLS intercept,
# Var(Y) ~ D^β slope, and finite-D debias (global or leave-one-D-out).
from __future__ import annotations
import argparse, csv, math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

EPS = 1e-12

# ---------------- utils ----------------
def _rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)

def binary_entropy_bits(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return -(p*np.log2(p) + (1-p)*np.log2(1-p))

def ols_line(x, y):
    x = np.asarray(x); y = np.asarray(y)
    A = np.column_stack((x, np.ones_like(x)))
    a,b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)

def wls_line(x, y, w, ridge: float = 1e-10):
    """WLS y = a x + b with weights w (≈ 1/Var). Ridge-stabilized."""
    x = np.asarray(x, float); y = np.asarray(y, float); w = np.asarray(w, float)
    n = x.size
    X = np.column_stack((x, np.ones_like(x)))
    sqrtw = np.sqrt(np.maximum(w, EPS))
    Xw = X * sqrtw[:, None]; yw = y * sqrtw
    XtX = Xw.T @ Xw
    try:
        beta = np.linalg.solve(XtX + ridge * np.eye(2), Xw.T @ yw)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtX) @ (Xw.T @ yw)
    a, b = float(beta[0]), float(beta[1])
    yhat = X @ beta
    resid = y - yhat
    s2 = float((w * resid**2).sum() / max(n - 2, 1))
    try:
        XtX_inv = np.linalg.inv(XtX + ridge * np.eye(2))
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    cov = s2 * XtX_inv
    se_a = float(np.sqrt(max(cov[0,0], 0.0)))
    se_b = float(np.sqrt(max(cov[1,1], 0.0)))
    return a, b, se_a, se_b

def bootstrap_alpha_loglog(X, Y, B, seed):
    X = np.asarray(X); Y = np.asarray(Y)
    m = (X>0) & (Y>0) & np.isfinite(X) & np.isfinite(Y)
    lx = np.log10(X[m]); ly = np.log10(Y[m])
    n = lx.size
    if n < 8: return float("nan"), float("nan"), float("nan")
    g = _rng(seed)
    slopes = np.empty(B, float)
    for b in range(B):
        idx = g.integers(0, n, size=n)
        a,_ = ols_line(lx[idx], ly[idx]); slopes[b] = a
    mean = float(np.mean(slopes))
    lo,hi = np.percentile(slopes, [2.5, 97.5])
    return mean, float(lo), float(hi)

def bootstrap_var_slope(D_list, var_list, B, seed):
    D = np.asarray(D_list, float); V = np.asarray(var_list, float)
    m = (D>0) & (V>0)
    lx = np.log10(D[m]); ly = np.log10(V[m])
    n = lx.size
    if n < 3: return float("nan"), float("nan"), float("nan")
    g = _rng(seed)
    slopes = np.empty(B, float)
    for b in range(B):
        idx = g.integers(0, n, size=n)
        a,_ = ols_line(lx[idx], ly[idx]); slopes[b] = a
    mean = float(np.mean(slopes))
    lo,hi = np.percentile(slopes, [2.5, 97.5])
    return mean, float(lo), float(hi)

# --------------- samplers ---------------
def sample_pairs_fast(nE: int, trials: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    dS = 2; dE = 2**nE
    g = _rng(seed)
    re = g.normal(size=(trials, dS, dE)); im = g.normal(size=(trials, dS, dE))
    Z = re + 1j*im
    norms = np.linalg.norm(Z.reshape(trials, -1), axis=1).reshape(trials,1,1) + 0.0
    Psi = Z / np.maximum(norms, EPS)
    rho = np.einsum("bie,bje->bij", Psi, Psi.conj())  # (B,2,2)
    a = rho[:,0,0].real; b = rho[:,0,1]
    purity = a*a + (1-a)*(1-a) + 2*(b.real*b.real + b.imag*b.imag)
    purity = np.clip(purity, 0.5, 1.0)
    deff = 1.0 / np.maximum(purity, EPS)
    X = np.maximum(deff - 1.0, EPS)
    s = np.sqrt(np.maximum(2.0*purity - 1.0, 0.0))
    lam1 = 0.5*(1.0 + s)
    Ibits = 2.0 * binary_entropy_bits(lam1)
    F = np.clip(a, 0.0, 1.0); A2 = np.arccos(np.sqrt(F))**2
    Y = (np.sqrt(X) * A2) / np.maximum(Ibits, EPS)
    m = (Ibits > EPS) & np.isfinite(X) & np.isfinite(Y)
    return X[m], Y[m]

def _haar_two_frame(D: int, xp, rg=None):
    if rg is None: rg = xp.random
    z0 = rg.normal(size=D) + 1j*rg.normal(size=D)
    z1 = rg.normal(size=D) + 1j*rg.normal(size=D)
    v0 = z0 / xp.linalg.norm(z0)
    z1 = z1 - (xp.vdot(v0, z1) * v0)
    n1 = xp.linalg.norm(z1)
    if float(n1.real) < 1e-20:
        z1 = rg.normal(size=D) + 1j*rg.normal(size=D)
        z1 = z1 - (xp.vdot(v0, z1) * v0)
        n1 = xp.linalg.norm(z1)
    v1 = z1 / xp.maximum(n1, 1e-32)
    return v0, v1

def _metrics_from_v0(v0, dS, dE, xp):
    M = v0.reshape(dS, dE)
    rhoS = M @ xp.conjugate(M).T
    purity = float(xp.real(xp.trace(rhoS @ rhoS)))
    deff = 1.0 / max(purity, EPS)
    X = max(deff - 1.0, EPS)
    a = float(xp.real(rhoS[0,0]))
    s = math.sqrt(max(2.0*purity - 1.0, 0.0))
    lam1 = 0.5*(1.0 + s)
    Ibits = 2.0 * float(binary_entropy_bits(np.array([lam1]))[0])
    F = min(max(a, 0.0), 1.0); A2 = math.acos(math.sqrt(F))**2
    Y = (math.sqrt(X) * A2) / max(Ibits, EPS)
    return X, Y

def sample_pairs_haar(nE: int, trials: int, seed: int, device: str="cpu"):
    dS=2; dE=2**nE; D=dS*dE
    if device=="gpu":
        try:
            import cupy as cp
            xp=cp; cp.random.seed(seed & 0x7fffffff)
            Xs=[]; Ys=[]
            for _ in range(trials):
                v0,_ = _haar_two_frame(D, xp, cp.random)
                X,Y = _metrics_from_v0(v0, dS, dE, xp)
                Xs.append(X); Ys.append(Y)
            return np.asarray(Xs,float), np.asarray(Ys,float)
        except Exception:
            pass
    g=_rng(seed); Xs=[]; Ys=[]
    for _ in range(trials):
        z0 = g.normal(size=D)+1j*g.normal(size=D)
        z1 = g.normal(size=D)+1j*g.normal(size=D)
        v0 = z0/np.linalg.norm(z0)
        z1 = z1 - (np.vdot(v0,z1)*v0)
        n1 = np.linalg.norm(z1)
        if n1 < 1e-20:
            z1 = g.normal(size=D)+1j*g.normal(size=D)
            z1 = z1 - (np.vdot(v0,z1)*v0)
            n1 = np.linalg.norm(z1)
        v1 = z1/max(n1,1e-32)
        X,Y = _metrics_from_v0(v0, dS, dE, np)
        Xs.append(X); Ys.append(Y)
    return np.asarray(Xs,float), np.asarray(Ys,float)

# --------------- experiment ---------------
@dataclass
class PerDRow:
    D:int; invD:float; trials:int
    alpha:float; alpha_lo:float; alpha_hi:float; alpha_se:float
    varY:float

def _per_seed_job(sampler: str, device: str, nE: int, trials: int, seed: int, boot_B_point: int):
    if sampler=="fast":
        X,Y = sample_pairs_fast(nE, trials, seed)
    else:
        X,Y = sample_pairs_haar(nE, trials, seed, device=device)
    mu, lo, hi = bootstrap_alpha_loglog(X, Y, B=boot_B_point, seed=seed ^ 0xA5A5A5A5)
    se = (hi - lo) / (2*1.96) if (np.isfinite(lo) and np.isfinite(hi)) else float("nan")
    return (nE, seed, mu, lo, hi, se, X, Y)

def parse_ne_list(s: str) -> List[int]:
    if "," in s:
        parts=[]
        for tok in s.split(","):
            tok=tok.strip()
            if "-" in tok:
                a,b = tok.split("-"); parts.extend(range(int(a), int(b)+1))
            else:
                parts.append(int(tok))
        return sorted(set(parts))
    if "-" in s:
        a,b=s.split("-"); return list(range(int(a), int(b)+1))
    return [int(s)]

def transform_alpha(alpha_vec: np.ndarray, mode: str) -> np.ndarray:
    if mode=="abs":   return np.abs(alpha_vec)
    if mode=="sq":    return alpha_vec**2
    return alpha_vec

def run_phase9_plus(
    sampler: str,
    device: str,
    nE_list: List[int],
    trials: int,
    seeds_per_D: int,
    boot_B_point: int,
    boot_B_intercept: int,
    outdir: str,
    workers: int,
    alpha_reg: str,
    use_wls: bool,
    debias: str,   # 'none' | 'global' | 'lodo'
):
    out = Path(outdir); (out/"data").mkdir(parents=True, exist_ok=True); (out/"figures").mkdir(parents=True, exist_ok=True)

    # ---- parallel seeds
    buckets: Dict[int, List[tuple]] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs=[]
        for nE in nE_list:
            D = 2*(2**nE)
            for k in range(seeds_per_D):
                seed = (0xC0FFEE ^ (D*0x9E3779B1) ^ k) & 0xFFFFFFFF
                futs.append(ex.submit(_per_seed_job, sampler, device, nE, trials, seed, boot_B_point))
        for f in as_completed(futs):
            nE, seed, mu, lo, hi, se, X, Y = f.result()
            D = 2*(2**nE)
            buckets.setdefault(D, []).append((mu, lo, hi, se, X, Y))

    # ---- aggregate per D
    perD: List[PerDRow] = []; clouds: Dict[int, Dict[str,np.ndarray]] = {}
    for D in sorted(buckets.keys()):
        invD = 1.0 / D
        items = buckets[D]
        alphas = np.array([t[0] for t in items if np.isfinite(t[0])], float)
        ses    = np.array([max(t[3],EPS) for t in items if np.isfinite(t[0])], float)
        if alphas.size==0: continue
        a_mu = float(np.mean(alphas)); a_lo, a_hi = np.percentile(alphas, [2.5,97.5])
        a_se = float(np.sqrt(np.mean(ses**2)))
        Xcat = np.concatenate([t[4] for t in items], axis=0)
        Ycat = np.concatenate([t[5] for t in items], axis=0)
        varY = float(np.var(Ycat, ddof=1))
        perD.append(PerDRow(D=D, invD=invD, trials=trials*len(items),
                            alpha=a_mu, alpha_lo=float(a_lo), alpha_hi=float(a_hi),
                            alpha_se=a_se, varY=varY))
        clouds[D]={"X":Xcat,"Y":Ycat}
    perD.sort(key=lambda r: r.D)

    # ---- write per-D CSV
    perD_csv = out/"data"/f"phase9_perD_{sampler}.csv"
    with perD_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames="D,invD,trials,alpha,alpha_lo,alpha_hi,alpha_se,varY".split(","))
        w.writeheader()
        for r in perD: w.writerow(r.__dict__)

    # ---- α at largest D
    Dmax = perD[-1].D; Xmax, Ymax = clouds[Dmax]["X"], clouds[Dmax]["Y"]
    a_mu, a_lo, a_hi = bootstrap_alpha_loglog(Xmax, Ymax, B=boot_B_point, seed=0xABCDEF01)
    pass_alpha = (a_lo <= 0.0 <= a_hi)

    # ---- intercept fit (with optional debias)
    x = np.array([r.invD for r in perD], float)
    a_raw = np.array([r.alpha for r in perD], float)
    se = np.array([r.alpha_se for r in perD], float)
    y0 = transform_alpha(a_raw, alpha_reg)

    def fit_line(xv, yv, wv):
        if use_wls:
            s, b, *_ = wls_line(xv, yv, wv); return s, b
        else:
            return ols_line(xv, yv)

    # weights for WLS
    if alpha_reg=="abs":
        se_eff = se
    elif alpha_reg=="sq":
        se_eff = 2*np.abs(a_raw)*se
        se_eff[se_eff<EPS] = np.min(se_eff[se_eff>0]) if np.any(se_eff>0) else EPS
    else:
        se_eff = se
    w = 1.0 / np.maximum(se_eff**2, EPS)

    # debias
    if debias == "global":
        s_hat, b_hat = fit_line(x, y0, w)
        y = y0 - (s_hat*x + b_hat)  # residuals
    elif debias == "lodo":
        y = np.empty_like(y0)
        n = len(x)
        for i in range(n):
            mask = np.ones(n, dtype=bool); mask[i] = False
            s_i, b_i = fit_line(x[mask], y0[mask], w[mask])
            y[i] = y0[i] - (s_i * x[i] + b_i)
    else:
        y = y0

    # final intercept CI by bootstrap (case resampling)
    g=_rng(0x13572468); B=boot_B_intercept; ints=np.empty(B,float); slopes=np.empty(B,float)
    for b in range(B):
        idx = g.integers(0,len(x),size=len(x))
        s_hat, b_hat = fit_line(x[idx], y[idx], w[idx])
        slopes[b]=s_hat; ints[b]=b_hat
    intercept_mu = float(np.mean(ints)); b_lo,b_hi = np.percentile(ints,[2.5,97.5])
    slope_mu = float(np.mean(slopes));   s_lo,s_hi = np.percentile(slopes,[2.5,97.5])
    pass_intercept = (b_lo <= 0.0 <= b_hi)

    # ---- Var(Y) ~ D^β
    D_list = [r.D for r in perD]; var_list = [r.varY for r in perD]
    beta_mu, beta_lo, beta_hi = bootstrap_var_slope(D_list, var_list, B=boot_B_intercept, seed=0x424242)

    # ---- plots
    # α bootstrap at Dmax
    g=_rng(0xABCDEF01)
    m=(Xmax>0)&(Ymax>0); lx,ly=np.log10(Xmax[m]),np.log10(Ymax[m])
    Bplot=min(boot_B_point, 6000); boots=[]
    for _ in range(Bplot):
        idx=g.integers(0,len(lx),size=len(lx)); a,_=ols_line(lx[idx],ly[idx]); boots.append(a)
    plt.figure(figsize=(6.4,4.0))
    plt.hist(boots,bins=60,density=True,alpha=0.85); plt.axvline(0.0,color="k",ls="--",lw=1)
    plt.title(f"Bootstrap of signed α at largest D={Dmax}\nmean={a_mu:+.3f}, 95% CI [{a_lo:+.3f},{a_hi:+.3f}]")
    plt.xlabel("α"); plt.ylabel("density")
    fig1 = out/"figures"/f"phase9_alpha_hist_Dmax_{sampler}.png"
    plt.tight_layout(); plt.savefig(fig1,dpi=160); plt.close()

    # intercept plot (post-debias target)
    xx = np.linspace(0.0, x.max()*1.05, 200)
    s_fit, b_fit = fit_line(x, y, w)
    label_y = {"signed": r"$\alpha$", "abs": r"$|\alpha|$", "sq": r"$\alpha^2$"}[alpha_reg]
    plt.figure(figsize=(7.2,4.6))
    plt.scatter(x, y, s=35, label=f"{label_y} (post-{debias})")
    plt.plot(xx, s_fit*xx + b_fit, "--", label=f"intercept={b_fit:+.3f} [{b_lo:+.3f},{b_hi:+.3f}]")
    plt.xlabel(r"$1/D$"); plt.ylabel(label_y)
    plt.title(f"{label_y} vs 1/D — {'WLS' if use_wls else 'OLS'} ({debias})")
    plt.legend()
    fig2 = out/"figures"/f"phase9_alpha_vs_invD_{sampler}_{alpha_reg}_{'wls' if use_wls else 'ols'}_{debias}.png"
    plt.tight_layout(); plt.savefig(fig2,dpi=160); plt.close()

    # var slope plot
    plt.figure(figsize=(6.8,4.4))
    plt.scatter(np.log10(D_list), np.log10(var_list), s=35)
    a_tmp,b_tmp = ols_line(np.log10(D_list), np.log10(var_list))
    xlin = np.linspace(min(np.log10(D_list))*0.98, max(np.log10(D_list))*1.02, 200)
    plt.plot(xlin, a_tmp*xlin + b_tmp, "--", label=f"slope β={beta_mu:+.3f} [{beta_lo:+.3f},{beta_hi:+.3f}]")
    plt.xlabel(r"$\log_{10} D$"); plt.ylabel(r"$\log_{10} \mathrm{Var}(Y)$")
    plt.title("Var(Y) vs D (log–log)")
    plt.legend()
    fig3 = out/"figures"/f"phase9_varY_vs_D_{sampler}.png"
    plt.tight_layout(); plt.savefig(fig3,dpi=160); plt.close()

    # ---- summary
    summ_csv = out/"data"/f"phase9_summary_{sampler}_{alpha_reg}_{'wls' if use_wls else 'ols'}_{debias}.csv"
    with summ_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sampler","device","alpha_reg","fit","debias","Dmin","Dmax","points",
            "alpha_Dmax_mean","alpha_Dmax_lo","alpha_Dmax_hi","pass_alpha_CI_zero",
            "intercept_mu","intercept_lo","intercept_hi","pass_intercept_CI_zero",
            "slope_mu","slope_lo","slope_hi",
            "var_slope_mu","var_slope_lo","var_slope_hi"
        ])
        w.writeheader()
        w.writerow({
            "sampler": sampler, "device": device, "alpha_reg": alpha_reg,
            "fit": "wls" if use_wls else "ols", "debias": debias,
            "Dmin": perD[0].D, "Dmax": Dmax, "points": len(perD),
            "alpha_Dmax_mean": a_mu, "alpha_Dmax_lo": a_lo, "alpha_Dmax_hi": a_hi,
            "pass_alpha_CI_zero": bool(pass_alpha),
            "intercept_mu": float(intercept_mu), "intercept_lo": float(b_lo), "intercept_hi": float(b_hi),
            "pass_intercept_CI_zero": bool(pass_intercept),
            "slope_mu": float(slope_mu), "slope_lo": float(s_lo), "slope_hi": float(s_hi),
            "var_slope_mu": float(beta_mu), "var_slope_lo": float(beta_lo), "var_slope_hi": float(beta_hi),
        })

    print("\n=== Phase 9+ summary ===")
    print(f"Sampler={sampler} device={device} points={len(perD)} D∈[{perD[0].D},{Dmax}]")
    print(f"  α@Dmax: mean={a_mu:+.4f}, 95% CI [{a_lo:+.4f},{a_hi:+.4f}]  -> {'PASS' if pass_alpha else 'FAIL'} (0∈CI)")
    print(f"  Intercept ({'WLS' if use_wls else 'OLS'}) for {label_y} vs 1/D ({debias}): "
          f"b={intercept_mu:+.4f} [{b_lo:+.4f},{b_hi:+.4f}]  -> {'PASS' if pass_intercept else 'FAIL'}")
    print(f"  Var(Y) slope β: {beta_mu:+.3f} [{beta_lo:+.3f},{beta_hi:+.3f}] (target ≈ -1)")
    print(f"Wrote per-D: {perD_csv}")
    print(f"Wrote summary: {summ_csv}")
    print(f"Saved: {fig1}\n       {fig2}\n       {fig3}")

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Phase 9+ all-in-one (with debias)")
    ap.add_argument("--sampler", choices=["haar","fast"], default="haar")
    ap.add_argument("--device", choices=["cpu","gpu"], default="cpu")
    ap.add_argument("--nE", type=str, default="7-14", help="env qubits (e.g., 7-14 or 7,8,10)")
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--seeds-per-D", type=int, default=10)
    ap.add_argument("--boot-B-point", type=int, default=12000)
    ap.add_argument("--boot-B-intercept", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--alpha-reg", choices=["signed","abs","sq"], default="signed")
    ap.add_argument("--wls", action="store_true", help="use WLS (weights = 1/SE^2)")
    ap.add_argument("--debias", choices=["none","global","lodo"], default="lodo",
                    help="remove finite-D drift via residualization")
    ap.add_argument("--outdir", default="phase9-plus")
    args = ap.parse_args()

    nE_list = parse_ne_list(args.nE)
    run_phase9_plus(
        sampler=args.sampler,
        device=args.device,
        nE_list=nE_list,
        trials=args.trials,
        seeds_per_D=args.seeds_per_D,
        boot_B_point=args.boot_B_point,
        boot_B_intercept=args.boot_B_intercept,
        outdir=args.outdir,
        workers=args.workers,
        alpha_reg=args.alpha_reg,
        use_wls=bool(args.wls),
        debias=args.debias,
    )

if __name__ == "__main__":
    main()
