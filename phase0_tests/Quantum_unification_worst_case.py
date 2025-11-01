# ============================
# WORST TEST v2 — FULL SCRIPT
# Curvature–Information–Dimension law stress test with robustness fixes
# ============================

import numpy as np
from numpy.random import default_rng
from scipy.linalg import expm, qr

# ------------------ Global RNG ------------------
GLOBAL_SEED = 12345
rng = default_rng(GLOBAL_SEED)

# ------------------ Linear algebra helpers ------------------
I2 = np.eye(2, dtype=complex)
sx = np.array([[0,1],[1,0]], complex)
sy = np.array([[0,-1j],[1j,0]], complex)
sz = np.array([[1,0],[0,-1]], complex)

def dm(psi: np.ndarray) -> np.ndarray:
    """Projector |psi><psi|."""
    return np.outer(psi, psi.conj())

def kron_all(ops):
    """Kronecker product of a list of arrays, preserving 1D/2D shapes."""
    out = np.array(ops[0], copy=False)
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def partial_trace(rho: np.ndarray, keep, dims):
    """
    Trace out all subsystems except those in `keep`.
    dims: list of local dimensions (len = number of subsystems).
    Index-stable via permutation to [keep|trace|keep'|trace'] then einsum.
    """
    n = len(dims)
    d_tot = int(np.prod(dims))
    assert rho.shape == (d_tot, d_tot)
    keep = list(keep)
    trace_over = [i for i in range(n) if i not in keep]

    # Permute to [keep, trace, keep', trace']
    perm = keep + trace_over + [i + n for i in keep] + [i + n for i in trace_over]
    tens = rho.reshape(*(dims + dims)).transpose(perm)

    dK = int(np.prod([dims[i] for i in keep])) if keep else 1
    dT = int(np.prod([dims[i] for i in trace_over])) if trace_over else 1
    tens = tens.reshape(dK, dT, dK, dT)

    # Trace over the traced space
    rho_keep = np.einsum('abcb->ac', tens)  # sum over b
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

# ------------------ Model Hamiltonians ------------------
def embed_1q_op_in_register(op, idx, n):
    """Place single-qubit `op` at position idx in an n-qubit register."""
    ops = [I2]*n
    ops = ops.copy()
    ops[idx] = op
    return kron_all(ops)

def H_dephasing(nS, nE):
    """H = (sum_i σz^S_i) ⊗ (sum_j σz^E_j)"""
    HS = sum(embed_1q_op_in_register(sz, i, nS) for i in range(nS))
    HE = sum(embed_1q_op_in_register(sz, j, nE) for j in range(nE))
    return np.kron(HS, HE)

def H_pswap(nS, nE):
    """Heisenberg exchange on the first S–E pair."""
    Sx = embed_1q_op_in_register(sx, 0, nS)
    Sy = embed_1q_op_in_register(sy, 0, nS)
    Sz = embed_1q_op_in_register(sz, 0, nS)
    Ex = embed_1q_op_in_register(sx, 0, nE)
    Ey = embed_1q_op_in_register(sy, 0, nE)
    Ez = embed_1q_op_in_register(sz, 0, nE)
    return np.kron(Sx,Ex)+np.kron(Sy,Ey)+np.kron(Sz,Ez)

def unitary_from_H(H, kappa):
    return expm(-1j*kappa*H)

def scramble_basis(U, nS, nE, prng):
    """Local random basis change: (U_S ⊗ U_E)† U (U_S ⊗ U_E) — preserves spectrum, enforces typical orientation."""
    US = haar_unitary(2**nS, prng)
    UE = haar_unitary(2**nE, prng)
    W  = np.kron(US, UE)
    return W.conj().T @ U @ W

# ------------------ One-collision metrics ------------------
def one_collision_metrics(U, psiS, psiE, nS, nE, twirlE=False, prng=None):
    """
    Return Ibits, A^2, d_eff after one S–E collision under unitary U.
    Optionally twirl environment input each shot.
    """
    if prng is None:
        prng = rng

    dE = psiE.size
    if twirlE:
        UE = haar_unitary(dE, prng)
        psiE = UE @ psiE

    psiSE0 = np.kron(psiS, psiE)           # (dS*dE,)
    psiSE1 = U @ psiSE0
    rhoSE1 = dm(psiSE1)

    # reduce to S (qubits 0..nS-1)
    dims = [2]*(nS + nE)
    rhoS1 = partial_trace(rhoSE1, keep=list(range(nS)), dims=dims)

    # info (global pure ⇒ I=2S(ρS))
    Sbits = von_neumann_entropy_bits(rhoS1)
    Ibits = 2.0*Sbits

    # curvature proxy: A = arccos sqrt(F(|ψ_S⟩, ρS1))
    Fproj = fidelity_pure_mixed(psiS, rhoS1)
    Fproj = float(np.clip(Fproj, 0.0, 1.0))
    A2 = (np.arccos(np.sqrt(Fproj)))**2

    # effective dimension from purity
    purity = float(np.real(np.trace(rhoS1 @ rhoS1)))
    purity = max(purity, 1e-12)
    d_eff = 1.0/purity

    return Ibits, A2, d_eff

# ------------------ Robust fit on log–log ------------------
def _ols_loglog(x, y):
    """Ordinary least squares on logs: log y = c - beta log x."""
    eps = 1e-12
    lx = np.log(np.clip(x, eps, None))
    ly = np.log(np.clip(y, eps, None))
    A = np.vstack([np.ones_like(lx), -lx]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    c, beta = coef[0], coef[1]
    yhat = A @ coef
    ss_res = float(np.sum((ly - yhat)**2))
    ss_tot = float(np.sum((ly - np.mean(ly))**2))
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
    n, p = len(lx), 2
    sigma2 = ss_res / max(n-p, 1)
    XtX_inv = np.linalg.inv(A.T @ A)
    se_beta = np.sqrt(sigma2 * XtX_inv[1,1])
    ci95 = 1.96 * se_beta
    return float(beta), float(R2), float(ci95), int(n)

def _binned_median(x, y, bins=20, min_per_bin=20):
    """Bin by x in log-space, take median y per bin; return compact (xb, yb)."""
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < bins:
        return x, y  # fall back
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), bins+1)
    xb, yb = [], []
    for i in range(bins):
        mask = (x >= edges[i]) & (x < edges[i+1])
        if mask.sum() >= min_per_bin:
            xb.append(np.median(x[mask]))
            yb.append(np.median(y[mask]))
    return np.array(xb), np.array(yb)

def fit_beta_loglog(X, Y, use_binning=True, bins=20, min_per_bin=20):
    """Wrapper: optionally bin (median) then OLS on logs; return beta, R2, ci, n_used."""
    if use_binning:
        Xc, Yc = _binned_median(X, Y, bins=bins, min_per_bin=min_per_bin)
        if len(Xc) >= 5:
            return _ols_loglog(Xc, Yc)
    # fallback: raw
    return _ols_loglog(X, Y)

# ------------------ Batch runner ------------------
def run_batch(model, nS, nE, kappa, n=2000, *,
              twirlE=False, scrambleU=True,
              normalizeX=True, min_I=1e-3,
              use_binning=True, bins=20, min_per_bin=20,
              seed=7):
    """
    Run many random shots and fit the law: A^2/I ~ K * X^{-beta}, with X = normalized d_eff - 1 if normalizeX.
    Returns: beta, R2, <I>, <d_eff>, ci95, n_used
    """
    prng = default_rng(seed)

    # Build model Hamiltonian
    if model == 'dephasing':
        H = H_dephasing(nS, nE)
    elif model == 'pswap':
        H = H_pswap(nS, nE)
    else:
        raise ValueError("model must be 'dephasing' or 'pswap'")
    U = unitary_from_H(H, kappa)
    if scrambleU:
        U = scramble_basis(U, nS, nE, prng)

    # Collect data
    I_list = []; A2_list = []; deff_list = []
    for _ in range(n):
        psiS = haar_state(nS, prng)
        psiE = haar_state(nE, prng)
        Ibits, A2, d_eff = one_collision_metrics(U, psiS, psiE, nS, nE, twirlE=twirlE, prng=prng)
        I_list.append(Ibits); A2_list.append(A2); deff_list.append(d_eff)

    I = np.array(I_list); A2 = np.array(A2_list); deff = np.array(deff_list)

    # Construct regression variables
    if normalizeX:
        dS = 2**nS
        X = (deff - 1.0) / (dS - 1.0)  # in (0, 1]
    else:
        X = deff - 1.0

    eps = 1e-12
    # Filter low-information points (stable Y)
    mI = (I >= max(min_I, eps))
    X = X[mI]; I = I[mI]; A2 = A2[mI]; deff = deff[mI]
    # Response
    Y = np.clip(A2 / np.maximum(I, eps), 1e-300, None)
    # Valid mask
    m = (X > 0) & (Y > 0) & np.isfinite(X) & np.isfinite(Y)
    X, Y = X[m], Y[m]

    beta, R2, ci95, n_used = fit_beta_loglog(X, Y, use_binning=use_binning, bins=bins, min_per_bin=min_per_bin)
    return beta, R2, float(np.mean(I)), float(np.mean(deff)), ci95, n_used

def quick_line(model, nS, nE, kappa, *,
               twirlE, scrambleU=True, normalizeX=True,
               min_I=1e-3, use_binning=True, bins=20, min_per_bin=20, shots=1500, seed=11):
    beta, R2, Ibar, deff_bar, ci, n_used = run_batch(
        model, nS, nE, kappa, n=shots,
        twirlE=twirlE, scrambleU=scrambleU, normalizeX=normalizeX,
        min_I=min_I, use_binning=use_binning, bins=bins, min_per_bin=min_per_bin, seed=seed
    )
    tag = f"{model}{'+twirl' if twirlE else ''}{'+scr' if scrambleU else ''}{'+norm' if normalizeX else ''}{'+bin' if use_binning else ''}"
    print(f"{tag:18s} | S={nS},E={nE} | κ={kappa:>4.2f} | β={beta:>6.3f} ±{ci:>5.3f} (target 0.5) | "
          f"R²={R2:>5.3f} | n={n_used:4d} | <I>={Ibar:>5.3f} | <d_eff>={deff_bar:>5.3f}")

# ------------------ Run matrix ------------------
print("=== WORST TEST v2 (structured vs +twirl; normalized X; basis-scrambled U; binned fit) ===")
CONFIG = dict(scrambleU=True, normalizeX=True, min_I=1e-3, use_binning=True, bins=24, min_per_bin=20, shots=2000)

for (nS, nE) in [(1,1), (2,2)]:
    print(f"\n--- System: S={nS}, E={nE} ---")
    for kappa in (0.20, 0.60, 1.00):
        # Structured, no twirl
        for model in ('dephasing','pswap'):
            quick_line(model, nS, nE, kappa, twirlE=False, **CONFIG)
        # Add environment twirl
        for model in ('dephasing','pswap'):
            quick_line(model, nS, nE, kappa, twirlE=True, **CONFIG)

print("\nInterpretation:")
print("- With normalized X and basis scrambling, β should move upward and stabilize.")
print("- +twirl should further tighten β near the universal 0.5 in mixing regimes.")
print("- If β stays far below 0.5 for a model, that model is strongly structured/non-mixing at that κ.")
