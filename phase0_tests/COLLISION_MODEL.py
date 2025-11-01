#!/usr/bin/env python3
# quantum_projection_arrow.py
# Clean collision model: channel on S (project-each) + pure-state joint keep + exact reverse.

import numpy as np
from numpy import kron
from math import pi
from scipy.linalg import expm, eigvalsh
from scipy.optimize import minimize

# ================== Config ==================
INIT_STATE = "plus"   # "plus" -> |+>, "zero" -> |0>
KAPPA_POINTS = 11
OPT_RESTARTS = 6

# collision demo params
COLLISION_STEPS = 10        # keep <=10; joint pure-state grows as 2^(steps+1)
COLLISION_KAPPAS = [0.2, 0.5, 0.8]

# ================== LinAlg helpers ==================
def dag(A): return A.conj().T
def ket(v):
    v = np.asarray(v, dtype=complex).reshape(-1,1)
    n = np.linalg.norm(v)
    return v if n == 0 else v/n
def dm(psi): return psi @ dag(psi)

I2 = np.eye(2, dtype=complex)
sx = np.array([[0,1],[1,0]], dtype=complex)
sy = np.array([[0,-1j],[1j,0]], dtype=complex)
sz = np.array([[1,0],[0,-1]], dtype=complex)

# ---------- robust partial trace ----------
def ptrace(rho, dims, keep):
    dims = list(dims); n = len(dims)
    keep = sorted(keep)
    R = rho.reshape(*(dims + dims))
    not_keep = [i for i in range(n) if i not in keep]
    for s in reversed(not_keep):
        R = np.trace(R, axis1=s, axis2=s+n)
    out_dim = int(np.prod([dims[i] for i in keep])) if keep else 1
    return R.reshape(out_dim, out_dim)

def ptrace_rho_SE_to_S(rho_SE):  # two-qubit helper (S,E)
    return ptrace(rho_SE, [2,2], [0])

# ---------- metrics ----------
def fidelity(rho, sigma):
    evals, evecs = np.linalg.eigh(rho)
    sqrt_eval = np.sqrt(np.clip(evals, 0, None))
    sqrt_rho = (evecs * sqrt_eval) @ dag(evecs)
    M = sqrt_rho @ sigma @ sqrt_rho
    Mevals = np.linalg.eigvalsh((M+dag(M))/2)
    Mevals = np.clip(Mevals, 0, None)
    return float((np.sum(np.sqrt(Mevals)))**2)

def von_neumann_entropy(rho):
    ev = np.clip(eigvalsh((rho + dag(rho))/2), 1e-16, 1.0)
    return float(-np.sum(ev*np.log2(ev)))

# ---------- couplings ----------
def U_dephasing(kappa):
    return expm(-1j * kappa * kron(sz, sz))
def U_partial_swap(kappa):
    H = 0.5*(kron(sx,sx) + kron(sy,sy) + kron(sz,sz))
    return expm(-1j * kappa * H)

# ---------- initial states ----------
def initial_states():
    if INIT_STATE.lower()=="plus":
        S = ket([1,1])   # |+>
    elif INIT_STATE.lower()=="zero":
        S = ket([1,0])   # |0>
    else:
        raise ValueError("INIT_STATE must be 'plus' or 'zero'")
    E = ket([1,0])       # |0>
    return dm(S), dm(E), S, E

# ---------- sanity check ----------
def _ptrace_selfcheck():
    zero = ket([1,0]); one = ket([0,1])
    bell = (kron(zero, zero)+kron(one, one))/np.sqrt(2)
    rho_S = ptrace(kron(dm(zero),dm(zero)) + kron(dm(one),dm(one)) + kron(zero@dag(one),zero@dag(one)) + kron(one@dag(zero),one@dag(zero)), [2,2], [0])  # not used further; just structure
    rho_S = ptrace(dm(bell), [2,2], [0])
    assert np.allclose(rho_S, 0.5*I2, atol=1e-12), "ptrace failed (Bell)."

# ---------- baseline single-step experiment ----------
def run_single(kappa, U_builder):
    rho_S0, rho_E0, _, _ = initial_states()
    rho_SE0 = kron(rho_S0, rho_E0)
    U = U_builder(kappa)

    rho_SE_fwd = U @ rho_SE0 @ dag(U)
    rho_S_fwd  = ptrace_rho_SE_to_S(rho_SE_fwd)

    rho_SE_rev_joint = dag(U) @ rho_SE_fwd @ U
    rho_S_joint = ptrace_rho_SE_to_S(rho_SE_rev_joint)

    return {
        "F_joint": fidelity(rho_S0, rho_S_joint),
        "S_joint": von_neumann_entropy(rho_S_joint),
        "F_proj":  fidelity(rho_S0, rho_S_fwd),
        "S_proj":  von_neumann_entropy(rho_S_fwd),
    }

def sweep(U_builder, label, kappas):
    print(f"\n=== Sweep: {label}  (INIT_STATE={INIT_STATE}) ===")
    rows=[]
    for k in kappas:
        r=run_single(k,U_builder)
        rows.append({"kappa":k, **r})
    print("kappa\tF_joint\t\tF_proj\t\tS_joint\t\tS_proj")
    for r in rows:
        print(f"{r['kappa']:.3f}\t{r['F_joint']:.6f}\t{r['F_proj']:.6f}\t{r['S_joint']:.6f}\t{r['S_proj']:.6f}")
    return rows

# ---------- best local undo (as before) ----------
def U3(theta, phi, lam):
    ct, st = np.cos(theta/2), np.sin(theta/2)
    return np.array([[ct, -np.exp(1j*lam)*st],
                     [np.exp(1j*phi)*st, np.exp(1j*(phi+lam))*ct]], dtype=complex)

def best_local_undo_for_kappa(kappa, U_builder, restarts=6):
    rho_S0, rho_E0, _, _ = initial_states()
    U = U_builder(kappa)
    rho_S_fwd = ptrace_rho_SE_to_S(U @ kron(rho_S0, rho_E0) @ dag(U))
    def negF(x):
        th, ph, la = x
        V = U3(th, ph, la)
        return -fidelity(rho_S0, V @ rho_S_fwd @ dag(V))
    bounds=[(0,2*pi)]*3
    best=-1.0; best_x=None
    for _ in range(restarts):
        x0 = np.random.rand(3)*2*pi
        res = minimize(negF, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter":600})
        val = -res.fun
        if val>best: best, best_x = val, res.x
    return {"kappa":kappa, "F_best":float(best), "angles":tuple(map(float,best_x))}

def sweep_best_local_undo(U_builder, label, kappas, restarts=6):
    print(f"\n=== Best local undo sweep: {label}  (INIT_STATE={INIT_STATE}) ===")
    rows=[best_local_undo_for_kappa(k,U_builder,restarts) for k in kappas]
    print("kappa\tF_best\t\t(theta, phi, lam)")
    for r in rows:
        th,ph,la = r["angles"]
        print(f"{r['kappa']:.3f}\t{r['F_best']:.6f}\t({th:.3f}, {ph:.3f}, {la:.3f})")
    return rows

# ================== COLLISION MODEL ==================
# (A) Project-after-each: channel on S only (density matrix 2x2)
def channel_step_S_only(rho_S, U2):
    zero = ket([1,0]); rho_E0 = dm(zero)
    rho_SE = U2 @ kron(rho_S, rho_E0) @ dag(U2)
    return ptrace_rho_SE_to_S(rho_SE)

# (B) Keep-all: pure-state joint evolution with exact reverse
def apply_gate_on_S_last_state(psi, U2):
    """
    psi: state vector over n qubits ordered [S, E1, ..., E_t]
    U2: 4x4 applied on (S, last)
    Implementation: reshape to (2, 2^(n-2), 2), move to (S,last,mid)->(4,mid),
    matmul, then reshape back.
    """
    n = int(np.log2(psi.size))
    assert 2**n == psi.size
    mid = 1 << (n-2) if n>=2 else 1
    psi3 = psi.reshape(2, mid, 2)          # (S, mid, last)
    psi_flat = np.transpose(psi3, (0,2,1)).reshape(4, mid)  # (S,last)->row
    psi_new = (U2 @ psi_flat).reshape(2,2,mid)
    psi_new = np.transpose(psi_new, (0,2,1)).reshape(-1,1)
    return psi_new

def collision_model(U_builder, kappa, N=10):
    rho_S0, _, S0, _ = initial_states()
    U2 = U_builder(kappa)
    # --- Project-after-each path (2x2 density matrix) ---
    rho_S = rho_S0.copy()
    fid_proj=[]; ent_proj=[]
    for _ in range(N):
        rho_S = channel_step_S_only(rho_S, U2)
        fid_proj.append(fidelity(rho_S0, rho_S))
        ent_proj.append(von_neumann_entropy(rho_S))
    # --- Keep-all path (pure state vector) ---
    zero = ket([1,0])
    psi = kron(S0, zero)            # start with S + first ancilla
    fid_joint=[]; ent_joint=[]
    # measure S reduced from pure state quickly: reshape psi to (2, 2^t) and partial trace
    def reduced_S_from_state(psi_vec):
        n = int(np.log2(psi_vec.size))     # qubits = 1 + t
        rest = 1 << (n-1)
        mat = psi_vec.reshape(2, rest)     # (S, rest)
        rhoS = mat @ mat.conj().T
        return rhoS
    # First collision already added ancilla; apply gate each step (including first)
    for step in range(N):
        # ensure we have last ancilla present
        if step>0:
            psi = kron(psi, zero)
        psi = apply_gate_on_S_last_state(psi, U2)
        rhoS = reduced_S_from_state(psi)
        fid_joint.append(fidelity(rho_S0, rhoS))
        ent_joint.append(von_neumann_entropy(rhoS))
    # --- Exact global reverse ---
    for _ in range(N):
        psi = apply_gate_on_S_last_state(psi, dag(U2))
        # pop last ancilla by just reshaping; we don't need to actually remove it for the check
    rhoS_end = reduced_S_from_state(psi)
    F_end = fidelity(rho_S0, rhoS_end)
    S_end = von_neumann_entropy(rhoS_end)
    return {
        "proj":{"fid":np.array(fid_proj),"ent":np.array(ent_proj)},
        "joint":{"fid":np.array(fid_joint),"ent":np.array(ent_joint)},
        "final_global_recovery":{"F":F_end,"S":S_end},
        "N":N, "kappa":kappa
    }

def run_collision_demo():
    for label, Ub in [("Dephasing", U_dephasing), ("Partial-SWAP", U_partial_swap)]:
        print(f"\n=== Collision Model: {label}  (INIT_STATE={INIT_STATE}) ===")
        for k in COLLISION_KAPPAS:
            out = collision_model(Ub, k, N=COLLISION_STEPS)
            print(f"kappa={k:.2f} | Global reverse end: F={out['final_global_recovery']['F']:.6f}, S={out['final_global_recovery']['S']:.6f}")
            steps = np.arange(1, out["N"]+1)
            print("step\tF_proj\tS_proj\tF_joint\tS_joint")
            for i in range(out["N"]):
                print(f"{i+1}\t{out['proj']['fid'][i]:.6f}\t{out['proj']['ent'][i]:.6f}\t{out['joint']['fid'][i]:.6f}\t{out['joint']['ent'][i]:.6f}")
            try:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(steps, out["proj"]["fid"], marker='s', label="Fidelity (project-after-each)")
                plt.plot(steps, out["joint"]["fid"], marker='o', label="Fidelity (keep-all)")
                plt.xlabel("Collision #"); plt.ylabel("Fidelity to initial ρ_S")
                plt.title(f"{label}: Fidelity vs collisions — κ={k} — INIT={INIT_STATE}")
                plt.ylim(0,1.05); plt.grid(True, alpha=0.3); plt.legend()

                plt.figure()
                plt.plot(steps, out["proj"]["ent"], marker='s', label="Entropy S (project-after-each)")
                plt.plot(steps, out["joint"]["ent"], marker='o', label="Entropy S (keep-all)")
                plt.xlabel("Collision #"); plt.ylabel("von Neumann entropy S(ρ_S)")
                plt.title(f"{label}: Entropy vs collisions — κ={k} — INIT={INIT_STATE}")
                plt.ylim(0,1.05); plt.grid(True, alpha=0.3); plt.legend()
            except Exception as e:
                print("(plot skipped:", e, ")")
    try:
        import matplotlib.pyplot as plt
        plt.show()
    except Exception:
        pass

# ================== Main ==================
def main():
    _ptrace_selfcheck()
    kappas = np.linspace(0.0, 1.0, KAPPA_POINTS)

    # Baseline single-step sweeps
    rows_deph = sweep(U_dephasing,   "Dephasing U = exp(-i κ σz⊗σz)", kappas)
    rows_ps   = sweep(U_partial_swap,"Partial-SWAP U = exp(-i κ/2 Σ σi⊗σi)", kappas)

    # Best local undo (optional but useful)
    sweep_best_local_undo(U_dephasing,   "Dephasing", kappas, restarts=OPT_RESTARTS)
    sweep_best_local_undo(U_partial_swap,"Partial-SWAP", kappas, restarts=OPT_RESTARTS)

    # Collision model demo
    run_collision_demo()

if __name__ == "__main__":
    main()
