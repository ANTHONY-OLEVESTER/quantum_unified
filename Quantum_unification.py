# ============================================================
# Unification test: curvature / information ~ 1/sqrt(d_eff-1)
# ============================================================
import numpy as np
import numpy.linalg as LA
from scipy.linalg import expm, sqrtm
import matplotlib.pyplot as plt

# ---------- base utils ----------
pi = np.pi
log2 = lambda x: np.log(x)/np.log(2)

def dm(psi):                 # |psi><psi|
    psi = psi.reshape(-1,1)
    return psi @ psi.conj().T

def dag(X): return X.conj().T

def haar_state(nqubits, rng):
    dim = 2**nqubits
    v = rng.normal(size=(dim,)) + 1j*rng.normal(size=(dim,))
    v /= LA.norm(v)
    return v

def von_neumann_entropy_bits(rho, tol=1e-12):
    vals = np.real_if_close(LA.eigvalsh(rho))
    vals = np.clip(vals, 0, 1)
    nz   = vals[vals>tol]
    return float(-np.sum(nz*log2(nz)))

def partial_trace(rho, keep, dims):
    """
    Generic partial trace. rho is (D x D) with D=np.prod(dims).
    keep: list of subsystem indices to keep (0..len(dims)-1).
    dims: dimensions of each subsystem, e.g. [2,2,2,2].
    """
    dims = list(dims)
    N = len(dims)
    keep = sorted(list(keep))
    trace_over = [i for i in range(N) if i not in keep]

    # reshape to 2N indices: (i1..iN ; j1..jN)
    resh = rho.reshape(*(dims + dims))
    # trace out from the back so axis numbering stays valid
    for t in sorted(trace_over, reverse=True):
        resh = np.trace(resh, axis1=t, axis2=t+len(dims))
        dims.pop(t)
    d_keep = int(np.prod(dims))
    return resh.reshape(d_keep, d_keep)

# ---------- distances & figures of merit ----------
def fidelity_uhlmann(rho, sigma):
    # F_uhl(ρ,σ) = [Tr sqrt( sqrt(ρ) σ sqrt(ρ) )]^2
    rt = sqrtm(rho)
    inner = rt @ sigma @ rt
    ev = LA.eigvalsh((inner + inner.conj().T)/2)  # force Hermitian numerics
    ev = np.clip(ev, 0, None)
    return float((np.sum(np.sqrt(ev)))**2)

def bures_angle(rho, sigma):
    # A = arccos sqrt(F_uhl)
    F = fidelity_uhlmann(rho, sigma)
    return float(np.arccos(np.sqrt(np.clip(F, 0, 1))))

def effective_dimension(rho):
    tr2 = float(np.real(np.trace(rho @ rho)))
    tr2 = max(tr2, 1e-15)
    return 1.0/tr2

def fidelity_to_initial_pure(rho, psi0):
    return float(np.real(dag(psi0) @ rho @ psi0))

# ---------- Pauli and couplings ----------
I2 = np.eye(2, dtype=complex)
sx = np.array([[0,1],[1,0]], complex)
sy = np.array([[0,-1j],[1j,0]], complex)
sz = np.array([[1,0],[0,-1]], complex)

def kron(*ops):
    out = np.array([[1]], complex)
    for op in ops:
        out = np.kron(out, op)
    return out

def two_qubit_H_dephasing():
    return kron(sz, sz)

def two_qubit_H_partial_swap():
    return (kron(sx,sx) + kron(sy,sy) + kron(sz,sz))/2.0

def random_two_qubit_H(rng):
    # random traceless Hermitian in su(4)
    A = rng.normal(size=(4,4)) + 1j*rng.normal(size=(4,4))
    H = A + A.conj().T
    H -= np.trace(H)/4 * np.eye(4)
    return H

def embed_two_body_on_4q(H, pair):
    """Embed 4x4 Hamiltonian H on 4 qubits (dims 2,2,2,2) acting on `pair` (e.g., (1,2))"""
    ops = [I2, I2, I2, I2]
    # Expand H in Pauli basis to place it — or simpler: build unitary via controlled kron map
    # We’ll map basis |ab> of the pair to H, identity elsewhere.
    # Implement as projector-sum:
    # U = sum_{x} (|x><x| on env-other) ⊗ exp(-i κ H) on the pair.
    # But for *Hamiltonian* we only need embedding as H_total = I⊗...⊗H_{pair}⊗...⊗I.
    # Do that via kron on the right slots.
    a,b = pair
    order = [0,1,2,3]
    # Build via tensor reordering:
    # To place H on (a,b), we permute axes so (a,b) are consecutive, kron H, then unpermute.
    # For Hamiltonians it’s simpler to form as a sum over basis ops:
    # Construct basis on pair and lift with kron.
    Htot = np.zeros((16,16), complex)
    # basis on the pair
    basis = [np.array([[1,0],[0,0]], complex),
             np.array([[0,1],[0,0]], complex),
             np.array([[0,0],[1,0]], complex),
             np.array([[0,0],[0,1]], complex)]
    # But direct lifting via identity kronecker on the correct slots is easier:
    # H_pair lifted:
    left = [I2]*4
    left[a] = np.array([[1]], complex)  # placeholders not needed actually
    # Correct way: kron over slots, replacing the 2-slot block by H via reshape-trick:
    slots = [I2, I2, I2, I2]
    slots[a] = None
    slots[b] = None
    # Build Kron in correct order: we’ll create an operator that acts as H on (a,b).
    # We can use index shuffling with reshape:
    H4 = np.zeros((16,16), complex)
    # Indices: (q0,q1,q2,q3) — put H on (a,b)
    # Create permutation that brings (a,b) to the last two positions
    perm = [0,1,2,3]
    rest = [i for i in perm if i not in (a,b)]
    new_order = rest + [a,b]
    # reshape, permute, kron, unpermute:
    # Identity on "rest" (size 2^(len(rest))) ⊗ H on last two qubits
    d_rest = 2**len(rest)
    P = np.eye(16).reshape([2]*8)  # permutation tensor, we’ll instead build with swap matrices
    # To avoid over-engineering, use simple recipe:
    # Build all-zeros, then for every computational basis |i> apply H on the (a,b) bits.
    for x in range(16):
        xb = [(x>>k) & 1 for k in range(4)][::-1]  # bits q0..q3
        for y in range(16):
            yb = [(y>>k) & 1 for k in range(4)][::-1]
            if all(xb[k]==yb[k] for k in range(4) if k not in (a,b)):
                # same on untouched qubits; act with H on pair-subspace
                idx_pair_x = xb[a]*2 + xb[b]
                idx_pair_y = yb[a]*2 + yb[b]
                H4[x,y] = H[idx_pair_x, idx_pair_y]
    return H4

def make_4q_H(kind, rng):
    if kind == "dephasing":
        H_pair = two_qubit_H_dephasing()
        return embed_two_body_on_4q(H_pair, pair=(1,2))  # couple S's last qubit to E's first
    elif kind == "pswap":
        H_pair = two_qubit_H_partial_swap()
        return embed_two_body_on_4q(H_pair, pair=(1,2))
    elif kind == "random":
        H_pair = random_two_qubit_H(rng)
        return embed_two_body_on_4q(H_pair, pair=(1,2))
    else:
        raise ValueError("unknown 4q Hamiltonian kind")

# ---------- one-shot metrics ----------
def metrics_one_shot(U, psiS, psiE):
    psiSE0 = np.kron(psiS, psiE)
    psiSE  = U @ psiSE0
    rhoSE  = dm(psiSE)
    # reduce to S
    nS = int(np.log2(len(psiS)))
    dims = [2]* (nS + int(np.log2(len(psiE))))
    rhoS = partial_trace(rhoSE, keep=list(range(nS)), dims=dims)
    # refs
    rhoS0 = dm(psiS)
    Sbits = von_neumann_entropy_bits(rhoS)
    Ibits = 2.0*Sbits
    A2    = bures_angle(rhoS, rhoS0)**2
    F     = fidelity_to_initial_pure(rhoS, psiS)
    deff  = effective_dimension(rhoS)
    return Ibits, A2, F, deff

# ---------- data collection ----------
def collect(kind, nS_qubits, nE_qubits, kappa, n_trials=600, seed=0):
    rng = np.random.default_rng(seed)
    outI, outA2, outF, outDe = [], [], [], []
    if nS_qubits==1 and nE_qubits==1:
        # 2 qubits total
        if kind=="dephasing":
            H = two_qubit_H_dephasing()
        elif kind=="pswap":
            H = two_qubit_H_partial_swap()
        elif kind=="random":
            H = random_two_qubit_H(rng)
        else:
            raise ValueError
        U = expm(-1j*kappa*H)
        for _ in range(n_trials):
            psiS = haar_state(1, rng)
            psiE = haar_state(1, rng)
            I,A2,F,de = metrics_one_shot(U, psiS, psiE)
            outI.append(I); outA2.append(A2); outF.append(F); outDe.append(de)
    else:
        # 4 qubits: S(2) + E(2)
        H = make_4q_H(kind, rng)
        U = expm(-1j*kappa*H)
        for _ in range(n_trials):
            psiS = haar_state(nS_qubits, rng)
            psiE = haar_state(nE_qubits, rng)
            I,A2,F,de = metrics_one_shot(U, psiS, psiE)
            outI.append(I); outA2.append(A2); outF.append(F); outDe.append(de)

    return np.array(outI), np.array(outA2), np.array(outF), np.array(outDe)

# ---------- fitting ----------
def linfit_loglog(x, y):
    x = np.asarray(x); y = np.asarray(y)
    msk = (x>0) & (y>0)
    xx = np.log(x[msk]); yy = np.log(y[msk])
    if len(xx) < 5:
        return np.nan, np.nan, msk
    a,b = np.polyfit(xx, yy, 1)  # y = a x + b
    pred = a*xx + b
    ss_res = np.sum((yy-pred)**2)
    ss_tot = np.sum((yy-np.mean(yy))**2)
    R2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan
    C = np.exp(b)
    alpha = a
    return C, alpha, R2

def R2_from_model(x, y, yhat):
    m = (y>0) & np.isfinite(yhat)
    if m.sum()<5: return np.nan
    ss_res = np.sum((np.log(y[m]) - np.log(yhat[m]))**2)
    ss_tot = np.sum((np.log(y[m]) - np.mean(np.log(y[m])))**2)
    return 1 - ss_res/ss_tot if ss_tot>0 else np.nan

# ---------- plotting ----------
def scatter_with_fit(x, y, C, alpha, label_x, label_y, title):
    eps=1e-12
    xx = np.logspace(np.log10(max(min(x[x>0]),1e-6)), np.log10(max(x)), 200)
    yy = C * (xx**alpha)
    plt.figure(figsize=(5.0,3.6))
    plt.loglog(x+eps, y+eps, '.', alpha=0.35)
    plt.loglog(xx, yy, 'r-', lw=2, label=f'fit slope ≈ {alpha:.3f}')
    plt.xlabel(label_x); plt.ylabel(label_y); plt.title(title); plt.legend(); plt.grid(True, which='both', ls='--', alpha=0.3)
    plt.show()

# ---------- main unified test ----------
def run_unification(kind_list=("dephasing","pswap","random"),
                    sizes=((1,1),(2,2)),
                    kappas=(0.2,0.6,1.0),
                    n_trials=800, seed=1):
    print("=== Unified curvature–information–dimension law test ===")
    for nS,nE in sizes:
        print(f"\n--- System size: S={nS} qubit(s), E={nE} qubit(s) ---")
        for kind in kind_list:
            print(f"\n[{kind}]")
            for kappa in kappas:
                I,A2,F,de = collect(kind, nS, nE, kappa, n_trials=n_trials, seed=seed+int(100*kappa))
                # unified target:  A^2 / I  ~  K * (d_eff - 1)^(-beta)
                eps = 1e-9
                X = np.clip(de-1.0, eps, None)
                Y = np.clip(A2/(I+eps), eps, None)
                C, alpha, R2 = linfit_loglog(X, Y)  # Y ~ C * X^alpha   (want alpha ~ -0.5)
                print(f"kappa={kappa:.2f} | beta≈{-alpha:.3f}  (target 0.5)  | K≈{C:.3g}  | R²={R2:.3f}  | "
                      f"<I>={I.mean():.3f}  <F>={F.mean():.3f}  <d_eff>={de.mean():.3f}")

                title = f"{kind}, κ={kappa:.2f}, S={nS},E={nE}\nY=A²/I vs (d_eff-1): slope≈{alpha:.3f} (want -0.5)"
                scatter_with_fit(X, Y, C, alpha, "d_eff - 1", "A² / I", title)

# ----------------- run -----------------
run_unification(
    kind_list=("dephasing","pswap","random"),
    sizes=((1,1),(2,2)),         # test 2-qubit total and 4-qubit total
    kappas=(0.2, 0.6, 1.0),
    n_trials=900,
    seed=7,
)
