# ===================== 2D-INTERPRETATION: Info–Geometry–Dimension =====================
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

# --- use SciPy expm if available; else stable eigendecomp fallback ---
try:
    from scipy.linalg import expm as _expm
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def exp_unitary_from_H(H, theta):
    """U = exp(-i * theta * H) for Hermitian H."""
    if _HAS_SCIPY:
        return _expm(-1j * theta * H)
    # fallback: eigh
    w, V = LA.eigh((H + H.conj().T)/2)
    phases = np.exp(-1j * theta * np.clip(w, None, None))
    return (V * phases) @ V.conj().T

# ---------- Pauli & utilities ----------
I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1],
               [1, 0]], dtype=complex)
sy = np.array([[0, -1j],
               [1j, 0]], dtype=complex)
sz = np.array([[1, 0],
               [0,-1]], dtype=complex)

def kron(*ops):
    out = np.array([[1+0j]], dtype=complex)
    for op in ops:
        out = np.kron(out, op.astype(complex))
    return out

def dag(X): return X.conj().T

def dm(psi):
    psi = psi.reshape(-1,1).astype(complex)
    return psi @ psi.conj().T

def random_ket(n=2, rng=None):
    rng = rng or np.random.default_rng()
    v = rng.normal(size=n) + 1j*rng.normal(size=n)
    v /= LA.norm(v)
    return v.reshape(n,1).astype(complex)

# ---------- Correct partial trace over environment (trace E out of S⊗E) ----------
def ptrace_E_of_SE(rho_SE):
    """
    rho_SE: 4x4 density matrix (S⊗E), order (S,E).
    Returns 2x2 rho_S = Tr_E[rho_SE].
    """
    rho_SE = rho_SE.reshape(2,2,2,2)         # (s1, e1, s2, e2)
    # Sum over e1 and e2 to keep (s1, s2):
    rho_S = np.einsum('ijkl->ik', rho_SE)    # trace env indices j,l
    return rho_S

# ---------- Entropy, purity, fidelity, metric ----------
def von_neumann_entropy_bits(rho, eps=1e-12):
    vals = np.real_if_close(LA.eigvalsh((rho + dag(rho))/2))
    vals = np.clip(vals, 0.0, 1.0)
    nz = vals[vals > eps]
    return float(-np.sum(nz * np.log2(nz))) if nz.size else 0.0

def purity(rho):
    return float(np.real(np.trace(rho @ rho)))

def fidelity_proj_to_initial(rho_S_final, psi_init):
    # F = <psi| rho_S |psi> for pure initial |psi>
    psi = psi_init.reshape(-1,1).astype(complex)
    return float(np.real((psi.conj().T @ rho_S_final @ psi).item()))

def uhlmann_angle(F):
    F = float(np.clip(F, 0.0, 1.0))
    return float(np.arccos(np.sqrt(F)))

# ---------- Couplings ----------
def U_dephasing(kappa):
    # U = exp(-i κ sz ⊗ sz)
    H = kron(sz, sz)
    return exp_unitary_from_H(H, kappa)

def U_partial_swap(kappa):
    # U = exp(-i (κ/2) * (sx⊗sx + sy⊗sy + sz⊗sz))
    H = kron(sx,sx) + kron(sy,sy) + kron(sz,sz)
    return exp_unitary_from_H(H, 0.5*kappa)

# ---------- Core experiment: one collision samples ----------
def one_collision_samples(U_builder, kappa, n_trials=800, seed=0):
    rng = np.random.default_rng(seed)
    U = U_builder(kappa)
    I_list, F_list, A_list, d_list = [], [], [], []
    for _ in range(n_trials):
        psiS = random_ket(2, rng)
        psiE = random_ket(2, rng)
        rho0 = kron(dm(psiS), dm(psiE))
        rhoSE = U @ rho0 @ dag(U)
        rhoS  = ptrace_E_of_SE(rhoSE)

        S_bits = von_neumann_entropy_bits(rhoS)  # entanglement entropy (global pure)
        I_bits = 2.0 * S_bits                     # mutual information
        F      = fidelity_proj_to_initial(rhoS, psiS)
        A      = uhlmann_angle(F)
        d_eff  = 1.0 / max(purity(rhoS), 1e-16)

        I_list.append(I_bits); F_list.append(F); A_list.append(A); d_list.append(d_eff)
    return np.array(I_list), np.array(F_list), np.array(A_list), np.array(d_list)

# ---------- Fits ----------
def fit_geometric_law(I, A):
    # A^2 ≈ γ I (through origin)
    x = I
    y = A**2
    denom = float(np.dot(x, x)) + 1e-16
    gamma = float(np.dot(x, y) / denom)
    yhat = gamma * x
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2)) + 1e-16
    R2 = 1.0 - ss_res/ss_tot
    return gamma, R2

def fit_dimensional_law(d_eff, F):
    # (1-F) ≈ C * (d_eff - 1)^α  (log–log regression)
    x = np.maximum(d_eff - 1.0, 1e-12)
    y = np.maximum(1.0 - F, 1e-12)
    X = np.vstack([np.log(x), np.ones_like(x)]).T
    alpha, lnC = LA.lstsq(X, np.log(y), rcond=None)[0]
    C = float(np.exp(lnC))
    y_pred = C * x**alpha
    ss_res = float(np.sum((np.log(y) - np.log(y_pred))**2))
    ss_tot = float(np.sum((np.log(y) - np.mean(np.log(y)))**2)) + 1e-16
    R2 = 1.0 - ss_res/ss_tot
    return C, float(alpha), R2

# ---------- Plot ----------
def scatter_with_fit(x, y, xlabel, ylabel, title, k, loglog=False):
    plt.figure(figsize=(5.2, 3.6), dpi=130)
    plt.scatter(x, y, s=10, alpha=0.3)
    if loglog:
        plt.xscale('log'); plt.yscale('log')
    coef = float(np.dot(x, y) / (np.dot(x, x) + 1e-16))
    xs = np.linspace(max(1e-6, x.min()), x.max(), 200)
    plt.plot(xs, coef * xs, 'r', lw=2, label=f"fit slope ≈ {coef:.3g}")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(f"{title}\nκ={k:.2f}")
    plt.legend(frameon=False); plt.tight_layout()

# ---------- Master run ----------
def run_2D_interpretation():
    kappas_to_print = [0.1, 0.3, 0.6, 1.0]
    N = 800

    for name, U in [("Dephasing", U_dephasing), ("Partial-SWAP", U_partial_swap)]:
        print(f"\n=== {name} coupling ===")
        k = 0.6  # representative mid-range kappa for dense scatter
        I, F, A, d = one_collision_samples(U, k, N, seed=42)

        γ, R2g = fit_geometric_law(I, A)
        print(f"[Geo]   A² ≈ {γ:.4f} · I     R² = {R2g:.3f}")
        scatter_with_fit(I, A**2, "I(S:E) [bits]", "A² (Uhlmann)", f"{name}: Geometric law", k, loglog=False)

        C, α, R2d = fit_dimensional_law(d, F)
        print(f"[Dim]   (1 - F) ≈ {C:.3g} · (d_eff - 1)^{α:.3f}     R² = {R2d:.3f}")
        scatter_with_fit(d - 1.0, 1.0 - F, "d_eff - 1", "1 - F", f"{name}: Dimensional law", k, loglog=True)

        print("κ  <I>±σI       <F>±σF       <d_eff>")
        for k2 in kappas_to_print:
            Ii, Fi, Ai, di = one_collision_samples(U, k2, 400, seed=1)
            print(f"{k2:.2f}  {Ii.mean():.3f}±{Ii.std():.3f}   {Fi.mean():.3f}±{Fi.std():.3f}   {di.mean():.3f}")

    plt.show()

# ---------------- run ----------------
run_2D_interpretation()
