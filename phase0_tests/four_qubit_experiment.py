# 4-QUBIT INFORMATION–GEOMETRY–DIMENSION EXPERIMENT (fixed partial trace)
# S = 2 qubits, E = 2 qubits. Dephasing & Partial-SWAP couplings.

import numpy as np
from numpy.random import default_rng
from scipy.linalg import expm
import matplotlib.pyplot as plt

# ---------- Pauli & utilities ----------
sx = np.array([[0,1],[1,0]], dtype=complex)
sy = np.array([[0,-1j],[1j,0]], dtype=complex)
sz = np.array([[1,0],[0,-1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

def dag(X): return X.conj().T
def kron(*ops):
    out = np.array([[1]], dtype=complex)
    for op in ops:
        out = np.kron(out, op)
    return out

def haar_state(n_qubits, rng):
    d = 2**n_qubits
    v = (rng.normal(size=d) + 1j*rng.normal(size=d))
    v /= np.linalg.norm(v)
    return v

def dm(psi):  # density matrix
    return np.outer(psi, psi.conj())

def von_neumann_entropy_bits(rho, eps=1e-12):
    w = np.linalg.eigvalsh((rho + dag(rho))/2)
    w = np.clip(np.real(w), 0, 1)
    w = w[w > eps]
    return float(-np.sum(w*np.log2(w)))

def purity(rho):
    return float(np.real(np.trace(rho @ rho)))

# ---------- Robust partial trace (fixed) ----------
def partial_trace(rho, keep, dims):
    """
    Trace out all subsystems not in `keep`.
    rho: (prod(dims), prod(dims)) density
    keep: list of subsystem indices to keep
    dims: list of local dimensions (e.g., [2,2,2,2])
    """
    dims = list(map(int, dims))
    n = len(dims)
    D = int(np.prod(dims))
    assert rho.shape == (D, D)

    keep = sorted(keep)
    trace = [i for i in range(n) if i not in keep]

    # reshape to 2n indices: (i0,...,i_{n-1}, j0,...,j_{n-1})
    T = rho.reshape(*(dims + dims))

    # permute to [keep, trace, keep', trace']
    perm = keep + trace + [i + n for i in keep] + [i + n for i in trace]
    T = np.transpose(T, axes=perm)

    d_keep  = int(np.prod([dims[i] for i in keep])) if keep else 1
    d_trace = int(np.prod([dims[i] for i in trace])) if trace else 1

    # reshape to (d_keep, d_trace, d_keep, d_trace)
    T = T.reshape(d_keep, d_trace, d_keep, d_trace)

    # trace over the 'trace' space
    rho_keep = np.einsum('a b c b -> a c', T, optimize=True)
    return rho_keep

# ---------- Embed 2-qubit operators ----------
def embed_two_qubit_op(op_2q, qA, qB, n_qubits):
    assert op_2q.shape == (4,4)
    full = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    P0 = np.array([[1,0],[0,0]], dtype=complex)
    P1 = np.array([[0,0],[0,1]], dtype=complex)
    for a in range(2):
        for b in range(2):
            for a2 in range(2):
                for b2 in range(2):
                    elem = op_2q[2*a+b, 2*a2+b2]
                    site_ops_ket = []
                    site_ops_bra = []
                    for q in range(n_qubits):
                        if q == qA:
                            site_ops_ket.append(P0 if a==0 else P1)
                            site_ops_bra.append(P0 if a2==0 else P1)
                        elif q == qB:
                            site_ops_ket.append(P0 if b==0 else P1)
                            site_ops_bra.append(P0 if b2==0 else P1)
                        else:
                            site_ops_ket.append(I2)
                            site_ops_bra.append(I2)
                    K = kron(*site_ops_ket)
                    B = kron(*site_ops_bra)
                    full += elem * (K @ B.T.conj())
    return full

def embed_pair_pauli(pa, pb, qS, qE, n_qubits):
    return embed_two_qubit_op(np.kron(pa, pb), qS, qE, n_qubits)

# ---------- 4-qubit Hamiltonians: indices 0=S0,1=S1,2=E0,3=E1 ----------
def H_dephasing_4q():
    n = 4
    return (embed_pair_pauli(sz, sz, 0, 2, n) +
            embed_pair_pauli(sz, sz, 1, 3, n))

def H_partial_swap_4q():
    n = 4
    H02 = 0.5*(embed_pair_pauli(sx,sx,0,2,n) +
               embed_pair_pauli(sy,sy,0,2,n) +
               embed_pair_pauli(sz,sz,0,2,n))
    H13 = 0.5*(embed_pair_pauli(sx,sx,1,3,n) +
               embed_pair_pauli(sy,sy,1,3,n) +
               embed_pair_pauli(sz,sz,1,3,n))
    return H02 + H13

# ---------- Metrics ----------
def one_shot_metrics(U, psiS, psiE):
    psiSE0 = np.kron(psiS, psiE)
    rhoSE  = dm(U @ psiSE0)
    rhoS   = partial_trace(rhoSE, keep=[0,1], dims=[2,2,2,2])
    S_bits = von_neumann_entropy_bits(rhoS)
    I_bits = 2.0 * S_bits
    F = float(np.real(psiS.conj() @ (rhoS @ psiS)))
    F = float(np.clip(F, 0.0, 1.0))
    A = float(np.arccos(np.sqrt(F)))
    d_eff = 1.0 / max(purity(rhoS), 1e-12)
    return I_bits, F, A, d_eff

def collect_samples(H, kappa, n_trials=800, seed=0):
    rng = default_rng(seed)
    U = expm(-1j * kappa * H)
    I_list, F_list, A_list, d_list = [], [], [], []
    for _ in range(n_trials):
        psiS = haar_state(2, rng)
        psiE = haar_state(2, rng)
        I,F,A,d = one_shot_metrics(U, psiS, psiE)
        I_list.append(I); F_list.append(F); A_list.append(A); d_list.append(d)
    return np.array(I_list), np.array(F_list), np.array(A_list), np.array(d_list)

# ---------- Fits ----------
def fit_through_origin(x, y):
    x = np.asarray(x); y = np.asarray(y)
    sxx = float(np.dot(x, x))
    gamma = float(np.dot(x, y)/sxx) if sxx>0 else 0.0
    yhat = gamma*x
    ss_res = float(np.dot(y - yhat, y - yhat))
    ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
    R2 = 1.0 - ss_res/ss_tot if ss_tot>0 else 1.0
    return gamma, R2

def fit_power_law(x, y, xmin=1e-12, ymin=1e-12):
    x = np.clip(np.asarray(x), xmin, None)
    y = np.clip(np.asarray(y), ymin, None)
    X = np.log(x); Y = np.log(y)
    A = np.vstack([np.ones_like(X), X]).T
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    lnC, alpha = coef
    C = float(np.exp(lnC)); alpha = float(alpha)
    Yhat = A @ coef
    ss_res = float(np.dot(Y - Yhat, Y - Yhat))
    ss_tot = float(np.dot(Y - Y.mean(), Y - Y.mean()))
    R2 = 1.0 - ss_res/ss_tot if ss_tot>0 else 1.0
    return C, alpha, R2

# ---------- Plots ----------
def plot_geo(I, A, gamma, title):
    plt.figure(figsize=(5.2,3.6))
    plt.scatter(I, A**2, s=10, alpha=0.35)
    xs = np.linspace(0, max(1e-6, I.max()), 200)
    plt.plot(xs, gamma*xs, 'r-', lw=2, label=f'fit slope ≈ {gamma:.2f}')
    plt.xlabel('I(S:E) [bits]'); plt.ylabel(r'$A^2$ (Uhlmann)')
    plt.title(title); plt.legend(); plt.tight_layout()

def plot_dim(deff, F, C, alpha, title):
    x = np.clip(deff - 1.0, 1e-9, None)
    y = 1.0 - np.clip(F, 0.0, 1.0)
    plt.figure(figsize=(5.2,3.6))
    plt.loglog(x, y, '.', alpha=0.35)
    xs = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
    plt.loglog(xs, C*(xs**alpha), 'r-', lw=2, label=f'fit slope ≈ {alpha:.4f}')
    plt.xlabel(r'$d_{\rm eff}-1$'); plt.ylabel(r'$1-F$')
    plt.title(title); plt.legend(); plt.tight_layout()

# ---------- Main ----------
def run_four_qubit_experiment():
    H_dep = H_dephasing_4q()
    H_ps  = H_partial_swap_4q()

    kappas = [0.10, 0.30, 0.60, 1.00]
    Ntrials = 800
    k_rep = 0.60

    for label, H in [("Dephasing", H_dep), ("Partial-SWAP", H_ps)]:
        print(f"\n=== {label} coupling (4 qubits: S(2) + E(2)) ===")
        I, F, A, deff = collect_samples(H, k_rep, n_trials=Ntrials, seed=123)
        gamma, R2_geo = fit_through_origin(I, A**2)
        C, alpha, R2_dim = fit_power_law(np.clip(deff-1.0, 1e-9, None),
                                         np.clip(1.0-F, 1e-12, None))
        print(f"[Geo]   A^2 ≈ {gamma:.4f} · I     R² = {R2_geo:.3f}")
        print(f"[Dim]   (1 - F) ≈ {C:.3f} · (d_eff - 1)^{alpha:.3f}     R² = {R2_dim:.3f}")

        plot_geo(I, A, gamma, f'{label}: Geometric law\nκ={k_rep:.2f}')
        plot_dim(deff, F, C, alpha, f'{label}: Dimensional law\nκ={k_rep:.2f}')

        rows = []
        for k in kappas:
            I, F, A, deff = collect_samples(H, k, n_trials=Ntrials//2, seed=100+int(100*k))
            rows.append((k, I.mean(), I.std(), F.mean(), F.std(), deff.mean()))
        print("κ   <I>±σI       <F>±σF       <d_eff>")
        for (k, mI, sI, mF, sF, mD) in rows:
            print(f"{k:0.2f}  {mI:0.3f}±{sI:0.3f}   {mF:0.3f}±{sF:0.3f}   {mD:0.3f}")

    plt.show()

# ----------------- run -----------------
run_four_qubit_experiment()
