# ====================== Collapse test: Y = (d_eff-1)^{1/2} * A^2 / I ======================
# Goal: If the curvature-information-dimension law is universal, log Y vs log(d_eff-1)
# should be flat (slope ~ 0) across sizes and models in sufficiently mixing regimes.

import numpy as np
import numpy.linalg as LA
from scipy.linalg import expm
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 120})

# ---------- basic linear-algebra helpers ----------
def dm(psi):  # |psi><psi|
    return np.outer(psi, psi.conj())

def kron_op(*ops):
    """Kronecker for operators (2-D)."""
    out = np.array([[1.0+0.0j]])
    for A in ops:
        out = np.kron(out, A)
    return out

def haar_state(n_qubits, rng):
    d = 2**n_qubits
    z = (rng.normal(size=d) + 1j*rng.normal(size=d)) / np.sqrt(2.0)
    return z / LA.norm(z)

def random_unitary(d, rng):
    X = (rng.normal(size=(d,d)) + 1j*rng.normal(size=(d,d))) / np.sqrt(2.0)
    Q, R = LA.qr(X)
    ph = np.diag(R) / np.abs(np.diag(R))
    return Q @ np.diag(ph.conj())

def psd_sqrt(A, tol=1e-14):
    """Hermitian PSD square root via eigendecomposition (stable for rank-deficient)."""
    H = (A + A.conj().T) / 2
    w, V = LA.eigh(H)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.conj().T

def von_neumann_entropy_bits(rho, tol=1e-14):
    w, _ = LA.eigh((rho + rho.conj().T)/2)
    w = np.clip(w, 0.0, 1.0)
    w = w[w > tol]
    if w.size == 0:
        return 0.0
    return float(-np.sum(w * np.log2(w)))

def fidelity_uhlmann(rho, sigma):
    sR = psd_sqrt(rho)
    X  = sR @ sigma @ sR
    sX = psd_sqrt(X)
    val = np.real(np.trace(sX))**2
    return float(np.clip(val, 0.0, 1.0))

def partial_trace(rho, keep, dims):
    """
    rho on ⊗_k C^{d_k}, dims=[d0,...,d_{n-1}], keep is sorted list of subsystems to keep.
    Implementation uses a single permutation-reshape-trace (no axis drift).
    """
    dims = list(dims)
    n = len(dims)
    keep = sorted(keep)
    trace = [i for i in range(n) if i not in keep]

    # Permute indices to (keep, trace | keep', trace')
    perm = keep + trace + [i+n for i in keep] + [i+n for i in trace]
    T = rho.reshape(dims + dims).transpose(perm)

    d_keep  = int(np.prod([dims[i] for i in keep])) if keep else 1
    d_trace = int(np.prod([dims[i] for i in trace])) if trace else 1

    T = T.reshape(d_keep, d_trace, d_keep, d_trace)
    # Trace over the traced subsystem block
    rho_keep = np.einsum('aibi->ab', T)  # trace over the second/last (i)
    return rho_keep.reshape(d_keep, d_keep)

# ---------- Pauli ----------
I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]], complex)
sz = np.array([[1, 0], [0, -1]], complex)
paulis = [sx, sy, sz]

def embed_qubit_op(n, i, P):
    """I⊗...⊗P(i)⊗...⊗I for n qubits."""
    return kron_op(*([I2]*i + [P] + [I2]*(n-1-i)))

# ---------- model builders ----------
def build_U_dephasing(nS, nE, kappa):
    m = min(nS, nE)
    D = 2**(nS+nE)
    H = np.zeros((D, D), complex)
    for i in range(m):
        ZS = embed_qubit_op(nS, i, sz)
        ZE = embed_qubit_op(nE, i, sz)
        H += kron_op(ZS, ZE)
    return expm(-1j * kappa * H)

def build_U_pswap(nS, nE, kappa):
    m = min(nS, nE)
    D = 2**(nS+nE)
    H = np.zeros((D, D), complex)
    for i in range(m):
        for P in (sx, sy, sz):
            H += 0.5 * kron_op(embed_qubit_op(nS, i, P),
                               embed_qubit_op(nE, i, P))
    return expm(-1j * kappa * H)

def build_U_random2body(nS, nE, kappa, rng):
    D = 2**(nS+nE)
    H = np.zeros((D, D), complex)
    scale = 1.0 / np.sqrt(max(1, nS*nE))
    for i in range(nS):
        for j in range(nE):
            for Pa in paulis:
                for Pb in paulis:
                    c = scale * rng.normal()
                    H += c * kron_op(embed_qubit_op(nS, i, Pa),
                                     embed_qubit_op(nE, j, Pb))
    return expm(-1j * kappa * H)

# ---------- one-collision metrics ----------
def one_collision_metrics(U, psiS, psiE):
    nS = int(np.log2(len(psiS)))
    nE = int(np.log2(len(psiE)))
    dims = [2]*(nS+nE)
    S_idx = list(range(nS))

    psiSE0 = np.kron(psiS, psiE)     # shape (D,)
    rhoSE0 = dm(psiSE0)
    rhoSE1 = dm(U @ psiSE0)

    rhoS0 = partial_trace(rhoSE0, keep=S_idx, dims=dims)
    rhoS1 = partial_trace(rhoSE1, keep=S_idx, dims=dims)

    S_bits = von_neumann_entropy_bits(rhoS1)
    I_bits = 2.0 * S_bits

    F = fidelity_uhlmann(rhoS0, rhoS1)
    A = np.arccos(np.sqrt(np.clip(F, 0.0, 1.0)))
    A2 = float(A*A)

    deff = 1.0 / float(np.real(np.trace(rhoS1 @ rhoS1)))
    return I_bits, A2, deff

# ---------- sampling & fit ----------
def sample_collapse_data(model, kappa, sizes, n_trials=300, seed=0):
    rng = np.random.default_rng(seed)
    Ys, Xs = [], []   # Y = (d_eff-1)^{1/2} * A^2 / I ; X = d_eff - 1
    for (nS, nE) in sizes:
        if model == 'dephasing':
            U = build_U_dephasing(nS, nE, kappa)
        elif model == 'pswap':
            U = build_U_pswap(nS, nE, kappa)
        elif model == 'random2body':
            U = build_U_random2body(nS, nE, kappa, rng)
        else:
            raise ValueError("unknown model")

        for _ in range(n_trials):
            psiS = haar_state(nS, rng)
            psiE = haar_state(nE, rng)
            Ibits, A2, deff = one_collision_metrics(U, psiS, psiE)
            if Ibits <= 1e-12:  # avoid division blow-ups
                continue
            X = max(deff - 1.0, 1e-12)
            Y = (X**0.5) * (A2 / Ibits)
            Xs.append(X); Ys.append(Y)
    return np.array(Xs), np.array(Ys)

def fit_slope_loglog(X, Y):
    x = np.log(X)
    y = np.log(np.clip(Y, 1e-300, None))
    A = np.vstack([x, np.ones_like(x)]).T
    alpha, logC = LA.lstsq(A, y, rcond=None)[0]
    yhat = alpha*x + logC
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    R2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
    return float(alpha), float(R2)

# ---------- run & plot ----------
def collapse_test(kappa=0.60, n_trials=300, seed=0):
    sizes = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]  # extend as desired

    fig, axes = plt.subplots(1, 3, figsize=(15,4.2), sharey=True)
    models = ['dephasing', 'pswap', 'random2body']

    for ax, model in zip(axes, models):
        X, Y = sample_collapse_data(model, kappa, sizes, n_trials=n_trials, seed=seed)
        alpha, R2 = fit_slope_loglog(X, Y)

        ax.scatter(X, Y, s=10, alpha=0.35)
        xs = np.logspace(np.log10(np.min(X)), np.log10(np.max(X)), 200)
        C = np.exp(np.mean(np.log(Y)) - alpha*np.mean(np.log(X)))  # geom-mean anchor
        ax.plot(xs, C*(xs**alpha), lw=2, color='tab:red',
                label=f"fit α≈{alpha:+.3f}, R²={R2:.3f}")

        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r"$d_{\rm eff}-1$")
        ax.set_title(f"{model}, κ={kappa}")
        ax.grid(True, which='both', ls=':', alpha=0.5)
        ax.legend()

        print(f"{model:12s} | κ={kappa:.2f} | slope α≈{alpha:+.3f} (want 0) | R²={R2:.3f} | n={len(X)}")

    axes[0].set_ylabel(r"$Y=(d_{\rm eff}-1)^{1/2}\,A^2/I$")
    fig.suptitle("Collapse test: Y vs (d_eff-1) pooled across sizes (flat ⇒ universal)")
    plt.tight_layout()
    plt.show()

# ====================== go ======================
collapse_test(kappa=0.60, n_trials=300, seed=1234)
