# === Information–Fidelity Law: full fixed code (self-contained) ===
# Computes (I(S:E), F_proj) after ONE collision for κ-sweep, fits:
#   (1 - F) ≈ c I^α  and  -ln F ≈ a I + b
# Works with or without SciPy (uses eigendecomp fallback if SciPy missing).

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

# ---------- try SciPy expm; else fallback ----------
_HAS_SCIPY = False
try:
    from scipy.linalg import expm as _scipy_expm
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def exp_unitary_from_H(H, theta):
    """
    Return U = exp(-i * theta * H).
    If SciPy is present, use scipy.linalg.expm; otherwise use eigendecomposition.
    Assumes H is Hermitian (true for our couplings), so eigh is stable.
    """
    if _HAS_SCIPY:
        return _scipy_expm(-1j * theta * H)
    # fallback: eigendecomposition
    w, V = LA.eigh((H + H.conj().T) / 2)        # ensure Hermitian
    ph = np.exp(-1j * theta * np.clip(w, None, None))
    return (V * ph) @ V.conj().T

# ------------------ Linear algebra helpers ------------------
I2 = np.eye(2, dtype=complex)
sx = np.array([[0,1],[1,0]], dtype=complex)
sy = np.array([[0,-1j],[1j,0]], dtype=complex)
sz = np.array([[1,0],[0,-1]], dtype=complex)

def kron(*ops):
    out = np.array([[1+0j]])
    for A in ops:
        out = np.kron(out, A)
    return out

def dag(A): return A.conj().T
def dm(psi):  # |psi><psi|
    return np.outer(psi, psi.conj())

def rand_pure_qubit(rng):
    # Haar random on Bloch sphere
    u = rng.random()
    v = rng.random()
    theta = 2*np.arccos(np.sqrt(1-u))
    phi = 2*np.pi*v
    return np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)], dtype=complex)

def partial_trace(rho, keep, dims):
    # dims: list of subsystem dims; keep: indices to keep
    # returns density matrix on subsystems "keep" (in original order)
    dims = list(dims)
    keep = list(keep)
    drop = [i for i in range(len(dims)) if i not in keep]
    R = rho.reshape(*(dims + dims))
    for ax in sorted(drop, reverse=True):
        R = np.trace(R, axis1=ax, axis2=ax+len(dims))
    d_keep = int(np.prod([dims[i] for i in keep])) if keep else 1
    return R.reshape(d_keep, d_keep)

def von_neumann_entropy(rho, base=2):
    evals = np.real_if_close(LA.eigvalsh((rho + dag(rho))/2))
    evals = np.clip(evals, 0, 1)
    nz = evals[evals > 1e-15]
    if base == 2:
        return float(-(nz*np.log2(nz)).sum())
    return float(-(nz*np.log(nz)).sum())

def fidelity(rho, sigma):
    # Uhlmann fidelity F(rho, sigma) in [0,1]
    w, V = LA.eigh((rho + dag(rho))/2)          # ensure Hermitian
    w = np.clip(w, 0, None)
    sqrt_rho = (V * np.sqrt(w)) @ V.conj().T
    M = sqrt_rho @ sigma @ sqrt_rho
    ew, EV = LA.eigh((M + dag(M))/2)
    ew = np.clip(ew, 0, None)
    return float((np.sum(np.sqrt(ew)))**2)

# ------------------ Couplings (2-qubit) ------------------
def U_dephasing(kappa):
    # U = exp(-i kappa sz ⊗ sz)
    H = kron(sz, sz)
    return exp_unitary_from_H(H, kappa)

def U_partial_swap(kappa):
    # U = exp(-i (kappa/2) * (sx⊗sx + sy⊗sy + sz⊗sz))
    H = kron(sx,sx) + kron(sy,sy) + kron(sz,sz)
    return exp_unitary_from_H(H, 0.5*kappa)

# ------------------ One-collision experiment ------------------
def one_collision_IF_points(U_builder, kappas, n_trials=24, seed=7):
    """
    Generates (I(S:E), F_proj) pairs after a SINGLE S–E collision, sweeping κ,
    averaging across random initial pure states of S and E.
    F_proj := Uhlmann fidelity between reduced rho_S (after collision) and initial rho_S0.
    I(S:E) := mutual information S(rho_S) + S(rho_E) - S(rho_SE) in bits.
    """
    rng = np.random.default_rng(seed)
    out = []
    for kappa in kappas:
        U = U_builder(kappa)
        Is, Fs = [], []
        for _ in range(n_trials):
            psiS = rand_pure_qubit(rng)
            psiE = rand_pure_qubit(rng)
            rho0 = dm(kron(psiS, psiE))
            # forward
            rhoSE = U @ rho0 @ dag(U)
            rhoS  = partial_trace(rhoSE, keep=[0], dims=[2,2])
            rhoE  = partial_trace(rhoSE, keep=[1], dims=[2,2])
            # info + projected fidelity
            Ibits = von_neumann_entropy(rhoS,2) + von_neumann_entropy(rhoE,2) - von_neumann_entropy(rhoSE,2)
            Fp    = fidelity(rhoS, dm(psiS))   # vs initial pure |psiS>
            Is.append(Ibits); Fs.append(Fp)
        out.append({
            "kappa": kappa,
            "I_mean": np.mean(Is), "I_std": np.std(Is),
            "F_mean": np.mean(Fs), "F_std": np.std(Fs),
            "I_all": np.array(Is), "F_all": np.array(Fs),
        })
    return out

# ------------------ Fitting + diagnostics ------------------
def fit_power_law(I, F):
    # Fit (1-F) = c * I^alpha  on points with I>0 and F<1
    mask = (I > 1e-12) & (F < 1-1e-12)
    x = np.log(I[mask])
    y = np.log(1 - F[mask])
    if len(x) < 2:
        return None
    A = np.vstack([x, np.ones_like(x)]).T
    alpha, ln_c = LA.lstsq(A, y, rcond=None)[0]
    c = np.exp(ln_c)
    yhat = alpha*x + ln_c
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    R2 = 1 - ss_res/ss_tot if ss_tot>0 else 1.0
    return {"c": c, "alpha": alpha, "R2": R2}

def fit_exp(I, F):
    # Fit -ln F = a I + b  using points with F in (0,1]
    mask = (F > 1e-12) & (F <= 1.0)
    x = I[mask]
    y = -np.log(F[mask])
    if len(x) < 2:
        return None
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = LA.lstsq(A, y, rcond=None)[0]
    yhat = a*x + b
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    R2 = 1 - ss_res/ss_tot if ss_tot>0 else 1.0
    return {"a": a, "b": b, "R2": R2}

def plot_IF_and_fits(points, title):
    # concatenate all trials across κ
    I_all = np.concatenate([p["I_all"] for p in points])
    F_all = np.concatenate([p["F_all"] for p in points])

    # --- Power law: (1-F) ~ c I^alpha (log-log)
    fit_pl = fit_power_law(I_all, F_all)
    plt.figure()
    plt.title(f"{title}\nPower law: (1-F) vs I (log-log)")
    plt.xlabel("I(S:E)  [bits]")
    plt.ylabel("1 - F_proj")
    plt.xscale("log"); plt.yscale("log")
    plt.scatter(I_all, 1-F_all, s=18, alpha=0.6)
    if fit_pl is not None:
        Igrid = np.linspace(max(1e-4, I_all.min() if I_all.min()>0 else 1e-4), I_all.max(), 200)
        yfit = fit_pl["c"] * (Igrid ** fit_pl["alpha"])
        plt.plot(Igrid, yfit, linewidth=2)
        print(f"[Power]  c={fit_pl['c']:.4g},  alpha={fit_pl['alpha']:.4g},  R^2={fit_pl['R2']:.4f}")
    else:
        print("[Power] not enough dynamic range to fit.")

    # --- Exponential: -ln F ~ a I + b (linear)
    fit_ex = fit_exp(I_all, F_all)
    plt.figure()
    plt.title(f"{title}\nExponential: -ln F vs I")
    plt.xlabel("I(S:E)  [bits]")
    plt.ylabel("- ln F_proj")
    plt.scatter(I_all, -np.log(np.clip(F_all,1e-12,1)), s=18, alpha=0.6)
    if fit_ex is not None:
        Igrid = np.linspace(0, I_all.max(), 200)
        yfit = fit_ex["a"]*Igrid + fit_ex["b"]
        plt.plot(Igrid, yfit, linewidth=2)
        print(f"[Exp]    a={fit_ex['a']:.4g},  b={fit_ex['b']:.4g},  R^2={fit_ex['R2']:.4f}")
    else:
        print("[Exp] not enough points to fit.")

    # --- κ summary
    print("\nκ-sweep summary (mean±std over random initial states):")
    print("kappa\t<I>\t\t<F>")
    for p in points:
        print(f"{p['kappa']:.2f}\t{p['I_mean']:.4f}±{p['I_std']:.4f}\t{p['F_mean']:.4f}±{p['F_std']:.4f}")

# ------------------ Run experiments ------------------
def run_information_fidelity_experiment():
    kappas = np.linspace(0.05, 1.0, 12)  # avoid 0 to keep log-log well-conditioned
    Ntrials = 48                         # increase for tighter confidence

    print("=== Dephasing coupling ===")
    pts_dep = one_collision_IF_points(U_dephasing, kappas, n_trials=Ntrials, seed=1)
    plot_IF_and_fits(pts_dep, "Dephasing U = exp(-i κ σz⊗σz)")

    print("\n=== Partial-SWAP coupling ===")
    pts_ps = one_collision_IF_points(U_partial_swap, kappas, n_trials=Ntrials, seed=2)
    plot_IF_and_fits(pts_ps, "Partial-SWAP U = exp(-i κ/2 Σ σi⊗σi)")

    plt.show()

run_information_fidelity_experiment()
