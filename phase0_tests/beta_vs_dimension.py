# ============================================
# β vs Dimension (Scaling Study) — FULL SCRIPT
# Curvature–Information–Dimension law:
#   A^2 / I  ~  K * ( (d_eff - 1) / (dS - 1) )^{-β}
# Sweeps nS=1..3, nE=1..3 for dephasing / pswap / random-2body
# ============================================

import numpy as np
from numpy.random import default_rng
from scipy.linalg import expm, qr
import matplotlib.pyplot as plt

# ------------------ Config ------------------
GLOBAL_SEED = 4242
DEFAULT_KAPPA = 0.60          # moderate: mixing window
TRIALS_PER_POINT = 1500
USE_BINNING = True
BINS = 24
MIN_PER_BIN = 20
MIN_I = 1e-3                  # filter tiny I (stabilizes A^2/I)
SCRAMBLE_U = True             # random local basis for U (keeps spectrum)
TWIRL_E = False               # per-shot env twirl (set True if you want extra typicality)

# sweep grids
NS_LIST = [1,2,3]
NE_LIST = [1,2,3]
MODELS  = ("dephasing","pswap","random2body")  # structured vs mixing

rng = default_rng(GLOBAL_SEED)

# ------------------ LA & helpers ------------------
I2 = np.eye(2, dtype=complex)
sx = np.array([[0,1],[1,0]], complex)
sy = np.array([[0,-1j],[1j,0]], complex)
sz = np.array([[1,0],[0,-1]], complex)

def dm(psi):
    return np.outer(psi, psi.conj())

def kron_all(ops):
    out = np.array(ops[0], copy=False)
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def partial_trace(rho: np.ndarray, keep, dims):
    """
    Stable partial trace: permute to [keep|trace|keep'|trace'] then einsum over trace.
    dims: list of local dimensions
    """
    n = len(dims)
    d_tot = int(np.prod(dims))
    assert rho.shape == (d_tot, d_tot)
    keep = list(keep)
    trace_over = [i for i in range(n) if i not in keep]

    perm = keep + trace_over + [i+n for i in keep] + [i+n for i in trace_over]
    tens = rho.reshape(*(dims + dims)).transpose(perm)

    dK = int(np.prod([dims[i] for i in keep])) if keep else 1
    dT = int(np.prod([dims[i] for i in trace_over])) if trace_over else 1
    tens = tens.reshape(dK, dT, dK, dT)

    rho_keep = np.einsum('abcb->ac', tens)  # trace over second/fourth axes
    return rho_keep

def haar_state(nqubits, prng):
    d = 2**nqubits
    x = (prng.normal(size=d) + 1j*prng.normal(size=d))/np.sqrt(2)
    x /= np.linalg.norm(x)
    return x

def haar_unitary(dim, prng):
    X = (prng.normal(size=(dim,dim)) + 1j*prng.normal(size=(dim,dim)))/np.sqrt(2)
    Q, R = qr(X)
    ph = np.diag(R)/np.abs(np.diag(R))
    return Q @ np.diag(ph.conj())

def von_neumann_entropy_bits(rho, eps=1e-12):
    evals = np.clip(np.linalg.eigvalsh((rho+rho.conj().T)/2), 0.0, 1.0)
    evals = evals[evals > eps]
    return float(-(evals*np.log2(evals)).sum())

def fidelity_pure_mixed(psi, rho):
    return float(np.real(np.vdot(psi, rho @ psi)))

# ------------------ Embedding helpers ------------------
def embed_1q_op(op, idx, n):
    ops = [I2]*n
    ops[idx] = op
    return kron_all(ops)

def pauli_set():
    return (sx, sy, sz)

# ------------------ Hamiltonians ------------------
def H_dephasing(nS, nE):
    HS = sum(embed_1q_op(sz, i, nS) for i in range(nS))
    HE = sum(embed_1q_op(sz, j, nE) for j in range(nE))
    return np.kron(HS, HE)

def H_pswap(nS, nE):
    Sx = embed_1q_op(sx, 0, nS)
    Sy = embed_1q_op(sy, 0, nS)
    Sz = embed_1q_op(sz, 0, nS)
    Ex = embed_1q_op(sx, 0, nE)
    Ey = embed_1q_op(sy, 0, nE)
    Ez = embed_1q_op(sz, 0, nE)
    return np.kron(Sx,Ex) + np.kron(Sy,Ey) + np.kron(Sz,Ez)

def H_random_2body_SE(nS, nE, prng):
    """
    Mixing model: sum_{i in S, j in E} sum_{a in {x,y,z}} c_{a,ij} σ_a^S(i) ⊗ σ_a^E(j),
    with Gaussian coefficients c_{a,ij}. Produces k-local random SE couplings.
    """
    H = np.zeros((2**(nS+nE), 2**(nS+nE)), complex)
    S_paulis = [ [embed_1q_op(p, i, nS) for i in range(nS)] for p in pauli_set() ]  # 3 x nS
    E_paulis = [ [embed_1q_op(p, j, nE) for j in range(nE)] for p in pauli_set() ]  # 3 x nE
    for a in range(3):
        for i in range(nS):
            for j in range(nE):
                c = prng.normal() / np.sqrt(nS*nE)  # scale to keep norm reasonable
                H += c * np.kron(S_paulis[a][i], E_paulis[a][j])
    return H

def unitary_from_H(H, kappa):
    return expm(-1j * kappa * H)

def scramble_basis(U, nS, nE, prng):
    US = haar_unitary(2**nS, prng)
    UE = haar_unitary(2**nE, prng)
    W  = np.kron(US, UE)
    return W.conj().T @ U @ W

# ------------------ One-collision metrics ------------------
def one_shot_metrics(U, psiS, psiE, nS, nE, prng=None, twirlE=False):
    if prng is None:
        prng = rng
    if twirlE:
        UE = haar_unitary(2**nE, prng)
        psiE = UE @ psiE

    psiSE0 = np.kron(psiS, psiE)
    psiSE1 = U @ psiSE0
    rhoSE1 = dm(psiSE1)

    dims = [2]*(nS+nE)
    rhoS1 = partial_trace(rhoSE1, keep=list(range(nS)), dims=dims)

    Sbits = von_neumann_entropy_bits(rhoS1)
    Ibits = 2.0*Sbits

    Fproj = fidelity_pure_mixed(psiS, rhoS1)
    Fproj = float(np.clip(Fproj, 0.0, 1.0))
    A2    = (np.arccos(np.sqrt(Fproj)))**2

    purity = float(np.real(np.trace(rhoS1 @ rhoS1)))
    purity = max(purity, 1e-12)
    d_eff  = 1.0/purity

    return Ibits, A2, d_eff

# ------------------ Robust fitting (log–log) ------------------
def _ols_loglog(x, y):
    eps = 1e-12
    lx = np.log(np.clip(x, eps, None))
    ly = np.log(np.clip(y, eps, None))
    A  = np.vstack([np.ones_like(lx), -lx]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    c, beta = coef[0], coef[1]
    yhat = A @ coef
    ss_res = float(np.sum((ly - yhat)**2))
    ss_tot = float(np.sum((ly - np.mean(ly))**2))
    R2 = 1 - ss_res/ss_tot if ss_tot>0 else 0.0
    n, p = len(lx), 2
    sigma2 = ss_res / max(n-p, 1)
    XtX_inv = np.linalg.inv(A.T @ A)
    se_beta = np.sqrt(sigma2 * XtX_inv[1,1])
    ci95 = 1.96 * se_beta
    return float(beta), float(R2), float(ci95), int(n)

def _binned_median(x, y, bins=20, min_per_bin=20):
    m = (x>0) & (y>0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < bins:
        return x, y
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), bins+1)
    xb, yb = [], []
    for i in range(bins):
        sel = (x >= edges[i]) & (x < edges[i+1])
        if sel.sum() >= min_per_bin:
            xb.append(np.median(x[sel]))
            yb.append(np.median(y[sel]))
    return np.array(xb), np.array(yb)

def fit_beta_loglog(X, Y, use_binning=True, bins=20, min_per_bin=20):
    if use_binning:
        Xb, Yb = _binned_median(X, Y, bins=bins, min_per_bin=min_per_bin)
        if len(Xb) >= 5:
            return _ols_loglog(Xb, Yb)
    return _ols_loglog(X, Y)

# ------------------ Batch runner ------------------
def collect_and_fit(model, nS, nE, kappa=DEFAULT_KAPPA, shots=TRIALS_PER_POINT,
                    scrambleU=SCRAMBLE_U, twirlE=TWIRL_E, min_I=MIN_I,
                    use_binning=USE_BINNING, bins=BINS, min_per_bin=MIN_PER_BIN,
                    seed=0):
    prng = default_rng(seed)

    # Build H
    if model == "dephasing":
        H = H_dephasing(nS, nE)
    elif model == "pswap":
        H = H_pswap(nS, nE)
    elif model == "random2body":
        H = H_random_2body_SE(nS, nE, prng)
    else:
        raise ValueError("unknown model")

    U = unitary_from_H(H, kappa)
    if scrambleU:
        U = scramble_basis(U, nS, nE, prng)

    # Monte Carlo
    I_list, A2_list, de_list = [], [], []
    for _ in range(shots):
        psiS = haar_state(nS, prng)
        psiE = haar_state(nE, prng)
        Ibits, A2, d_eff = one_shot_metrics(U, psiS, psiE, nS, nE, prng=prng, twirlE=twirlE)
        I_list.append(Ibits); A2_list.append(A2); de_list.append(d_eff)

    I   = np.array(I_list); A2 = np.array(A2_list); deff = np.array(de_list)
    dS  = 2**nS
    X   = (deff - 1.0) / (dS - 1.0)       # normalized dimension axis in (0,1]
    eps = 1e-12

    # Filter small I (stabilizes A^2/I)
    mI = (I >= max(min_I, eps))
    X = X[mI]; I = I[mI]; A2 = A2[mI]

    Y = np.clip(A2 / np.maximum(I, eps), 1e-300, None)
    m = (X>0) & (Y>0) & np.isfinite(X) & np.isfinite(Y)
    X, Y = X[m], Y[m]

    beta, R2, ci95, n_used = fit_beta_loglog(X, Y, use_binning=use_binning, bins=bins, min_per_bin=min_per_bin)

    out = {
        "beta": beta,
        "R2":   R2,
        "ci":   ci95,
        "n_used": n_used,
        "I_mean": float(I.mean()) if len(I) else float('nan'),
        "deff_mean": float(deff[mI].mean()) if len(deff[mI]) else float('nan'),
        "points": len(I),
    }
    return out

# ------------------ Sweep & plotting ------------------
def run_scaling_sweep(kappa=DEFAULT_KAPPA, trials=TRIALS_PER_POINT, seed=GLOBAL_SEED):
    results = {}  # (model, nS, nE) -> dict
    print(f"=== β vs dimension scaling @ κ={kappa} ===")
    for model in MODELS:
        print(f"\n[{model}]")
        for nS in NS_LIST:
            for nE in NE_LIST:
                out = collect_and_fit(model, nS, nE, kappa=kappa, shots=trials,
                                      scrambleU=SCRAMBLE_U, twirlE=TWIRL_E,
                                      min_I=MIN_I, use_binning=USE_BINNING,
                                      bins=BINS, min_per_bin=MIN_PER_BIN,
                                      seed=seed + 1000*(nS*10+nE))
                results[(model,nS,nE)] = out
                dS = 2**nS; dE = 2**nE
                print(f"nS={nS}, nE={nE} | dS={dS:2d}, dE={dE:2d} | "
                      f"β={out['beta']:6.3f} ±{out['ci']:5.3f}  R²={out['R2']:5.3f}  "
                      f"n={out['n_used']:4d} | <I>={out['I_mean']:5.3f}  <d_eff>={out['deff_mean']:5.3f}")
    return results

def plot_beta_vs_dim(results, model, x_axis="dE", kappa_label=None):
    """
    x_axis: "dE" (environment dimension), "dS" (system dimension), or "dTot" (total)
    """
    xs, ys, yerr = [], [], []
    for nS in NS_LIST:
        for nE in NE_LIST:
            key = (model, nS, nE)
            if key not in results: continue
            dS = 2**nS; dE = 2**nE; dTot = dS*dE
            if x_axis=="dE":   x = dE
            elif x_axis=="dS": x = dS
            else:              x = dTot
            xs.append(x); ys.append(results[key]["beta"]); yerr.append(results[key]["ci"])
    xs = np.array(xs); ys = np.array(ys); yerr = np.array(yerr)

    # Sort for an ordered curve
    idx = np.argsort(xs)
    xs, ys, yerr = xs[idx], ys[idx], yerr[idx]

    plt.figure(figsize=(6,4))
    plt.errorbar(xs, ys, yerr=yerr, fmt='o-', capsize=3)
    plt.axhline(0.5, linestyle='--')
    plt.xlabel(x_axis)
    plt.ylabel("β (slope target 0.5)")
    ttl = f"β vs {x_axis} for {model}"
    if kappa_label: ttl += f"  (κ={kappa_label})"
    plt.title(ttl)
    plt.grid(True, ls='--', alpha=0.3)
    plt.show()

def plot_heatmap_beta(results, model):
    # 2D grid of β over nS x nE
    Z = np.zeros((len(NS_LIST), len(NE_LIST)))
    for i,nS in enumerate(NS_LIST):
        for j,nE in enumerate(NE_LIST):
            Z[i,j] = results[(model,nS,nE)]["beta"]
    plt.figure(figsize=(5,4))
    plt.imshow(Z, origin='lower', aspect='auto')
    plt.colorbar(label="β")
    plt.xticks(range(len(NE_LIST)), [2**n for n in NE_LIST])
    plt.yticks(range(len(NS_LIST)), [2**n for n in NS_LIST])
    plt.xlabel("dE"); plt.ylabel("dS")
    plt.title(f"β heatmap — {model}")
    plt.show()

# ------------------ Run all ------------------
results = run_scaling_sweep(kappa=DEFAULT_KAPPA, trials=TRIALS_PER_POINT, seed=GLOBAL_SEED)

# Plots: β vs dE for each model
for model in MODELS:
    plot_beta_vs_dim(results, model, x_axis="dE", kappa_label=DEFAULT_KAPPA)

# Optional: heatmaps (β across dS x dE) — helpful to spot regimes
for model in MODELS:
    plot_heatmap_beta(results, model)
