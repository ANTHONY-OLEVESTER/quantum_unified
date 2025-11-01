# ============================================================================
# Curvature–Information Universality: Formalize, Generalize, Relate (Full Test)
# ============================================================================
# - Defines Y = sqrt(deff-1) * A^2 / I
# - Models: random 2-body unitary (chaotic), partial-swap (structured),
#           dephasing (structured), depolarizing (qudit CPTP), amplitude damping (CPTP)
# - Options: + unitary twirl (pre/post local Haar) for channels to restore isotropy
# - Outputs: per-model slope α (log Y vs log(deff-1)), bootstrap CIs, Kendall τ,
#            and a CSV summary for all sweeps.
#
# Pure NumPy; safe in Colab. No plotting here (keeps it fast & headless).
# ============================================================================

import numpy as np
import csv, math
from collections import defaultdict

# -------------------------- LA / info-geo helpers ---------------------------
def dm(psi):
    psi = psi.reshape(-1,1)
    return psi @ psi.conj().T

def dagger(A): return A.conj().T

def eigh_psd(M):
    H = (M + dagger(M))/2
    w, V = np.linalg.eigh(H)
    w = np.clip(np.real(w), 0.0, None)
    return w, V

def sqrtm_psd(M):
    w, V = eigh_psd(M)
    return V @ np.diag(np.sqrt(w)) @ dagger(V)

def fidelity_uhlmann(rho, sigma):
    s = sqrtm_psd(rho) @ sigma @ sqrtm_psd(rho)
    return (np.trace(sqrtm_psd(s)).real)**2

def vn_entropy_bits(rho, tol=1e-12):
    w, _ = eigh_psd(rho)
    w = w[w>tol]
    return float((-w*np.log2(w)).sum()) if w.size else 0.0

def purity(rho): return float(np.real(np.trace(rho @ rho)))

def partial_trace(rho, keep, dims):
    """Trace out all subsystems not in keep (indices 0..n-1).
       rho: (D,D); dims: [d1,...,dn]; keep: sorted indices to keep."""
    dims = list(dims)
    n = len(dims)
    keep = sorted(keep)
    trace_over = [i for i in range(n) if i not in keep]
    resh = rho.reshape(*dims, *dims)  # (i1..in, j1..jn)
    for t in sorted(trace_over, reverse=True):
        resh = np.trace(resh, axis1=t, axis2=t+n)
        dims.pop(t)
        n -= 1
    d_keep = int(np.prod(dims)) if dims else 1
    return resh.reshape(d_keep, d_keep)

# -------------------------- random states/unitaries --------------------------
I2 = np.eye(2, dtype=complex)
sx = np.array([[0,1],[1,0]], complex)
sy = np.array([[0,-1j],[1j,0]], complex)
sz = np.array([[1,0],[0,-1]], complex)
paulis = [sx, sy, sz]

def kron(*ops):
    out = np.array(1.0 + 0j)
    for A in ops:
        out = np.kron(out, A)
    return out

def embed_one_qubit(op, idx, total):
    """Embed a single-qubit operator `op` at index `idx` within `total` qubits."""
    ops = [I2] * total
    ops[idx] = op
    return kron(*ops)

def embed_pair(op_a, idx_a, op_b, idx_b, total):
    """Embed op_a on idx_a and op_b on idx_b within `total` qubits."""
    if idx_a == idx_b:
        raise ValueError("Pair embedding requires distinct indices.")
    ops = [I2] * total
    ops[idx_a] = op_a
    ops[idx_b] = op_b
    return kron(*ops)

def haar_state(n_qudits, d=2, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    D = d**n_qudits
    x = (rng.normal(size=D)+1j*rng.normal(size=D))/np.sqrt(2)
    x /= np.linalg.norm(x)
    return x

def haar_unitary(D, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    X = (rng.normal(size=(D,D))+1j*rng.normal(size=(D,D)))/np.sqrt(2)
    Q, R = np.linalg.qr(X)
    d = np.diag(R)
    Q *= d/np.abs(d)
    return Q

def random_2body_H(nS, nE, kappa, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    total = nS + nE
    dim = 2 ** total
    H = np.zeros((dim, dim), complex)
    scale = 1.0 / np.sqrt(max(1, nS * nE))
    for i in range(nS):
        for j in range(nE):
            idx_s = i
            idx_e = nS + j
            for a in paulis:
                for b in paulis:
                    coeff = scale * rng.normal()
                    H += coeff * embed_pair(a, idx_s, b, idx_e, total)
    H *= kappa
    return H

def unitary_from_H(H):
    # exact via spectral decomposition (H Hermitian)
    w, V = np.linalg.eigh((H+dagger(H))/2)
    return V @ np.diag(np.exp(-1j*w)) @ dagger(V)

# ------------------------------ Channels (CPTP) ------------------------------
def weyl_operators(d):
    W=[]
    omega=np.exp(2j*np.pi/d)
    X = np.roll(np.eye(d, dtype=complex),1,axis=1)
    Z = np.diag([omega**k for k in range(d)])
    for a in range(d):
        Xa = np.linalg.matrix_power(X,a)
        for b in range(d):
            Zb = np.linalg.matrix_power(Z,b)
            W.append((Xa@Zb)/np.sqrt(d))
    return W

def kraus_depolarizing_qudit(d, lam):
    # Φ(ρ)=lam ρ + (1-lam) I/d
    W = weyl_operators(d)
    K=[]
    # identity element is W[0]; set probabilities so total TP
    q = (1-lam)/(d*d-1 + 1e-12)
    K.append(np.sqrt(lam) * np.eye(d, dtype=complex))
    for k in range(1, d*d):
        K.append(np.sqrt(q) * (np.sqrt(d)*W[k]))  # scale back to unitary (no 1/sqrt d)
    return K

def kraus_amplitude_damping(p):
    # Qubit AD with prob p: K0 = |0><0|+sqrt(1-p)|1><1|, K1 = sqrt(p)|0><1|
    K0 = np.array([[1,0],[0,np.sqrt(1-p)]], complex)
    K1 = np.array([[0,np.sqrt(p)],[0,0]], complex)
    return [K0, K1]

def stinespring_isometry(K):
    dS = K[0].shape[0]; m = len(K)
    V = np.zeros((dS*m, dS), complex)
    for k, Ak in enumerate(K):
        V[k*dS:(k+1)*dS,:] = Ak
    return V, dS, m

def apply_channel_Kraus(K, rho):
    out = np.zeros_like(rho)
    for A in K: out += A @ rho @ dagger(A)
    return out

def twirl_channel_sample(K, rho, rng=None):
    """1-sample unitary twirl U Φ(U† rho U) U†"""
    rng = np.random.default_rng() if rng is None else rng
    d = rho.shape[0]
    U = haar_unitary(d, rng)
    rho_in = dagger(U) @ rho @ U
    rho_out = apply_channel_Kraus(K, rho_in)
    return U @ rho_out @ dagger(U)

# ------------------------------- Metrics layer -------------------------------
def metrics_after_unitary(U, psiS, psiE, nS, nE):
    dims = [2]* (nS+nE)
    psiSE0 = kron(psiS, psiE).reshape(-1)
    rhoSE0 = dm(psiSE0)
    rhoSE1 = dm(U @ psiSE0)
    rhoS0 = partial_trace(rhoSE0, keep=list(range(nS)), dims=dims)
    rhoS1 = partial_trace(rhoSE1, keep=list(range(nS)), dims=dims)
    F = fidelity_uhlmann(rhoS0, rhoS1)
    A2 = float(np.arccos(np.sqrt(np.clip(F,0,1)))**2)
    Sbits = vn_entropy_bits(rhoS1)
    Ibits = 2*Sbits  # global pure
    deff = 1.0/purity(rhoS1)
    return Ibits, A2, deff

def metrics_after_kraus(K, rhoS0):
    # Stinespring with |0>_E initializes environment
    V, dS, m = stinespring_isometry(K)
    rhoSE1 = V @ rhoS0 @ dagger(V)   # E initialized in |0>
    rhoS1  = partial_trace(rhoSE1, keep=[0], dims=[dS,m])
    rhoE1  = partial_trace(rhoSE1, keep=[1], dims=[dS,m])
    F = fidelity_uhlmann(rhoS0, rhoS1)
    A2 = float(np.arccos(np.sqrt(np.clip(F,0,1)))**2)
    Ibits = vn_entropy_bits(rhoS1) + vn_entropy_bits(rhoE1) - vn_entropy_bits(rhoSE1)
    deff = 1.0/purity(rhoS1)
    return Ibits, A2, deff

# ------------------------------ Collapse observable --------------------------
def collect_unitary(model, kappa, nS, nE, n_trials, seed):
    rng = np.random.default_rng(seed)
    X=[]; Y=[]
    if model=='random2body':
        H = random_2body_H(nS, nE, kappa, rng)
        U = unitary_from_H(H)
    elif model in {'pswap', 'dephasing'}:
        total = nS + nE
        dim = 2 ** total
        H = np.zeros((dim, dim), complex)
        for i in range(min(nS, nE)):
            idx_s = i
            idx_e = nS + i
            if model == 'dephasing':
                H += embed_pair(sz, idx_s, sz, idx_e, total)
            else:  # pswap -> isotropic Heisenberg exchange
                H += 0.25 * (
                    embed_pair(sx, idx_s, sx, idx_e, total)
                    + embed_pair(sy, idx_s, sy, idx_e, total)
                    + embed_pair(sz, idx_s, sz, idx_e, total)
                )
        if np.allclose(H, 0):
            U = np.eye(dim, dtype=complex)
        else:
            U = unitary_from_H(kappa * H)
    else:
        raise ValueError("unknown unitary model")
    for _ in range(n_trials):
        psiS = haar_state(nS, d=2, rng=rng)
        psiE = haar_state(nE, d=2, rng=rng)
        Ibits, A2, deff = metrics_after_unitary(U, psiS, psiE, nS, nE)
        if Ibits<=1e-12: 
            continue
        x = max(deff-1.0, 1e-12)
        y = np.sqrt(x)*A2/Ibits
        X.append(x); Y.append(y)
    return np.array(X), np.array(Y)

def collect_channel(name, d, strength, n_trials, seed, twirl=False):
    rng = np.random.default_rng(seed)
    if name=='depolarizing':
        lam = strength  # 0..1
        K = kraus_depolarizing_qudit(d, lam)
    elif name=='amp_damp':
        p = strength     # 0..1 for qubit
        K = kraus_amplitude_damping(p)
        d = 2
    else:
        raise ValueError("unknown channel")
    X=[]; Y=[]
    for _ in range(n_trials):
        psi = haar_state(1, d=d, rng=rng)
        rho0 = dm(psi)
        if twirl:
            rho1 = twirl_channel_sample(K, rho0, rng)
            # Mutual information via Stinespring: use same K (averaged by twirl in expectation).
            Ibits, A2, deff = metrics_after_kraus(K, rho0)  # geometric A2 uses rho0->rho1 implicitly only via F; consistent in expectation
            # Replace A2 with the twirled step's A2 for the sampled U
            F = fidelity_uhlmann(rho0, rho1)
            A2 = float(np.arccos(np.sqrt(np.clip(F,0,1)))**2)
        else:
            Ibits, A2, deff = metrics_after_kraus(K, rho0)
        if Ibits<=1e-12:
            continue
        x = max(deff-1.0, 1e-12)
        y = np.sqrt(x)*A2/Ibits
        X.append(x); Y.append(y)
    return np.array(X), np.array(Y)

# ------------------------------- statistics ----------------------------------
def fit_slope_loglog(X, Y):
    mask = (X>0) & (Y>0)
    X = X[mask]; Y = Y[mask]
    lx, ly = np.log10(X), np.log10(Y)
    A = np.vstack([lx, np.ones_like(lx)]).T
    m, b = np.linalg.lstsq(A, ly, rcond=None)[0]
    yhat = m*lx + b
    ss_res = np.sum((ly - yhat)**2)
    ss_tot = np.sum((ly - ly.mean())**2) + 1e-12
    R2 = 1 - ss_res/ss_tot
    return float(m), float(R2), len(lx)

def kendall_tau(x, y):
    # simple O(n^2) Kendall τ-b for moderate n
    n = len(x); c = d = 0
    for i in range(n):
        for j in range(i+1, n):
            s = np.sign((x[i]-x[j])*(y[i]-y[j]))
            if s>0: c += 1
            elif s<0: d += 1
    denom = c+d
    return 0.0 if denom==0 else (c-d)/denom

def bootstrap_slope(X, Y, B=800, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    mask = (X>0) & (Y>0)
    X = X[mask]; Y = Y[mask]
    lx, ly = np.log10(X), np.log10(Y)
    n = len(lx)
    slopes=[]
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        A = np.vstack([lx[idx], np.ones_like(lx[idx])]).T
        m, _ = np.linalg.lstsq(A, ly[idx], rcond=None)[0]
        slopes.append(float(m))
    slopes = np.array(slopes)
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return slopes.mean(), lo, hi

# ------------------------------- sweep runner --------------------------------
def run_sweep(write_csv=True, seed=1):
    rng = np.random.default_rng(seed)
    rows=[]
    # Unitary models across sizes
    sizes = [(1,1),(1,2),(2,1),(2,2),(3,1)]
    for model in ['random2body','pswap','dephasing']:
        X=[]; Y=[]
        for (nS,nE) in sizes:
            Xi, Yi = collect_unitary(model, kappa=0.60, nS=nS, nE=nE, n_trials=300, seed=rng.integers(1e9))
            X.append(Xi); Y.append(Yi)
        X = np.concatenate(X); Y = np.concatenate(Y)
        a, R2, n = fit_slope_loglog(X,Y)
        am, lo, hi = bootstrap_slope(X,Y, B=600, rng=rng)
        tau = kendall_tau(np.log10(X), np.log10(Y))
        rows.append(['unitary', model, 'd=2', 'kappa=0.6', n, a, lo, hi, R2, tau])

# Channels: depolarizing (qudit), amplitude damping (qubit) with optional twirl
    for d in [2,3,5]:
        X,Y = collect_channel('depolarizing', d=d, strength=0.9, n_trials=1500, seed=rng.integers(1e9), twirl=False)
        a, R2, n = fit_slope_loglog(X,Y); am, lo, hi = bootstrap_slope(X,Y, B=600, rng=rng); tau = kendall_tau(np.log10(X), np.log10(Y))
        rows.append(['channel','depolarizing', f'd={d}','lam=0.9', n, a, lo, hi, R2, tau])

    for tw in [False, True]:
        X,Y = collect_channel('amp_damp', d=2, strength=0.3, n_trials=2000, seed=rng.integers(1e9), twirl=tw)
        a, R2, n = fit_slope_loglog(X,Y); am, lo, hi = bootstrap_slope(X,Y, B=600, rng=rng); tau = kendall_tau(np.log10(X), np.log10(Y))
        rows.append(['channel', f'amp_damp{"_twirl" if tw else ""}', 'd=2','p=0.3', n, a, lo, hi, R2, tau])

    # Print summary
    print('=== Universality sweep (alpha ~ 0, CI includes 0, tau ~ 0 => PASS) ===')
    hdr = ["kind","model","dim","param","n","alpha","alpha_lo","alpha_hi","R2","kendall_tau"]
    print('kind     model            dim      param        n  alpha  [95% CI]        R^2     tau')
    for r in rows:
        kind, model, dim, param, n, a, lo, hi, R2, tau = r
        print(f"{kind:8s} {model:16s} {dim:8s} {param:10s} {n:6d}  {a:+6.3f}  [{lo:+6.3f},{hi:+6.3f}]  {R2:5.3f}  {tau:+6.3f}")

    # CSV
    if write_csv:
        with open("universality_sweep.csv","w",newline="") as f:
            w=csv.writer(f); w.writerow(hdr); w.writerows(rows)
        print("\nSaved: universality_sweep.csv")

if __name__=="__main__":
    run_sweep(write_csv=False, seed=42)
