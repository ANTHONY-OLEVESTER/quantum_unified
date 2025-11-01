#!/usr/bin/env python3
# quantum_projection_arrow.py
# Robust partial trace + sanity checks + best local undo optimizer.

import numpy as np
from numpy import kron
from math import pi
from scipy.linalg import expm, eigvalsh
from scipy.optimize import minimize

# ------------------------- Config -------------------------
INIT_STATE = "plus"   # "plus" -> |+>, "zero" -> |0>
KAPPA_POINTS = 11     # points from 0.0 to 1.0 inclusive
OPT_RESTARTS = 8      # random restarts for optimizer

# ------------------------- LinAlg helpers -----------------
def dag(A): return A.conj().T
def ket(v):
    v = np.asarray(v, dtype=complex).reshape(-1, 1)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n
def dm(psi): return psi @ dag(psi)

# Pauli
I2 = np.eye(2, dtype=complex)
sx = np.array([[0,1],[1,0]], dtype=complex)
sy = np.array([[0,-1j],[1j,0]], dtype=complex)
sz = np.array([[1,0],[0,-1]], dtype=complex)

# ---------- Robust partial trace ----------
def ptrace(rho, dims, keep):
    """
    Partial trace of a density matrix 'rho' over a multipartite system.

    Args:
        rho  : (D x D) density matrix where D = prod(dims)
        dims : list of subsystem dimensions, e.g., [2,2] for two qubits
        keep : list of subsystem indices to keep (e.g., [0] keeps the first)

    Returns:
        Reduced density matrix over the kept subsystems, in the same order.
    """
    dims = list(dims)
    n = len(dims)
    keep = sorted(list(keep))
    # Reshape to (d0,d1,...,d_{n-1}, d0,d1,...,d_{n-1})
    R = rho.reshape(*(dims + dims))
    # Trace out subsystems not in 'keep' by pairing axis s with its bra axis s+n
    not_keep = [i for i in range(n) if i not in keep]
    # Trace from the highest index downward so axis numbering stays valid
    for s in reversed(not_keep):
        R = np.trace(R, axis1=s, axis2=s + R.ndim//2)
    # Now R has shape prod(dims[k] for k in keep) for ket and bra parts
    out_dim = int(np.prod([dims[i] for i in keep])) if keep else 1
    return R.reshape(out_dim, out_dim)

def ptrace_rho_SE_to_S(rho_SE):
    # Two qubits with ordering S (left), E (right); keep the left one.
    return ptrace(rho_SE, dims=[2,2], keep=[0])

# ---------- Metrics ----------
def fidelity(rho, sigma):
    # Uhlmann fidelity F(rho,sigma)
    evals, evecs = np.linalg.eigh(rho)
    sqrt_eval = np.sqrt(np.clip(evals, 0, None))
    sqrt_rho = (evecs * sqrt_eval) @ dag(evecs)
    M = sqrt_rho @ sigma @ sqrt_rho
    Mevals = np.linalg.eigvalsh((M + dag(M)) / 2)
    Mevals = np.clip(Mevals, 0, None)
    return float((np.sum(np.sqrt(Mevals)))**2)

def von_neumann_entropy(rho):
    ev = np.clip(eigvalsh((rho + dag(rho)) / 2), 1e-16, 1.0)
    return float(-np.sum(ev * np.log2(ev)))

# ---------- Unitaries (couplings) ----------
def U_dephasing(kappa):
    # U = exp(-i * kappa * sz ⊗ sz)
    H = kron(sz, sz)
    return expm(-1j * kappa * H)

def U_partial_swap(kappa):
    # Heisenberg exchange ~ partial SWAP: H = (sx⊗sx + sy⊗sy + sz⊗sz)/2
    H = 0.5 * (kron(sx, sx) + kron(sy, sy) + kron(sz, sz))
    return expm(-1j * kappa * H)

# ---------- Initial states ----------
def initial_states():
    if INIT_STATE.lower() == "plus":
        S = ket([1,1])      # |+>
    elif INIT_STATE.lower() == "zero":
        S = ket([1,0])      # |0>
    else:
        raise ValueError("INIT_STATE must be 'plus' or 'zero'")
    E = ket([1,0])          # environment starts in |0>
    return dm(S), dm(E)

# ---------- Sanity checks ----------
def _ptrace_selfcheck():
    # Build Bell state via kron to nail ordering: (|00> + |11>)/√2
    zero = ket([1,0]); one = ket([0,1])
    bell = (kron(zero, zero) + kron(one, one)) / np.sqrt(2)
    rho_SE = dm(bell)
    rho_S = ptrace_rho_SE_to_S(rho_SE)
    if not np.allclose(rho_S, 0.5 * I2, atol=1e-12):
        # Dump for debugging instead of a blind assert
        print("Selfcheck FAIL: Tr_E(|Φ+><Φ+|) != I/2")
        print("rho_S:\n", rho_S)
        raise AssertionError("Partial trace sanity check failed (Bell).")

    # Product state |+>⊗|0> → Tr_E gives |+><+|
    plus = ket([1,1]); rho_prod = dm(kron(plus, zero))
    rho_S2 = ptrace_rho_SE_to_S(rho_prod)
    if not np.allclose(rho_S2, dm(plus), atol=1e-12):
        print("Selfcheck FAIL: Tr_E(|+0><+0|) != |+><+|")
        print("rho_S2:\n", rho_S2)
        raise AssertionError("Partial trace sanity check failed (product).")

# ---------- Forward / reverse experiment ----------
def run_single(kappa, U_builder):
    rho_S0, rho_E0 = initial_states()
    rho_SE0 = kron(rho_S0, rho_E0)
    U = U_builder(kappa)

    # Forward
    rho_SE_fwd = U @ rho_SE0 @ dag(U)
    rho_S_fwd  = ptrace_rho_SE_to_S(rho_SE_fwd)

    # Case A: reverse S+E jointly (should recover exactly)
    rho_SE_rev_joint = dag(U) @ rho_SE_fwd @ U
    rho_S_joint = ptrace_rho_SE_to_S(rho_SE_rev_joint)
    F_joint = fidelity(rho_S0, rho_S_joint)
    S_joint = von_neumann_entropy(rho_S_joint)

    # Case B: project (trace E), then "reverse S only" (do nothing here)
    rho_S_proj_rev = rho_S_fwd
    F_proj = fidelity(rho_S0, rho_S_proj_rev)
    S_proj = von_neumann_entropy(rho_S_proj_rev)

    return {
        "kappa": kappa,
        "F_joint": F_joint, "S_joint": S_joint,
        "F_proj":  F_proj,  "S_proj":  S_proj
    }

def sweep(U_builder, label, kappas):
    print(f"\n=== Sweep: {label}  (INIT_STATE={INIT_STATE}) ===")
    rows = [run_single(k, U_builder) for k in kappas]
    print("kappa\tF_joint\t\tF_proj\t\tS_joint\t\tS_proj")
    for r in rows:
        print(f"{r['kappa']:.3f}\t{r['F_joint']:.6f}\t{r['F_proj']:.6f}\t{r['S_joint']:.6f}\t{r['S_proj']:.6f}")
    return rows

# ---------- Best local undo (single-qubit unitary on S) ----------
def U3(theta, phi, lam):
    ct, st = np.cos(theta/2), np.sin(theta/2)
    return np.array([
        [ct, -np.exp(1j*lam)*st],
        [np.exp(1j*phi)*st, np.exp(1j*(phi+lam))*ct]
    ], dtype=complex)

def best_local_undo_for_kappa(kappa, U_builder, restarts=8):
    rho_S0, rho_E0 = initial_states()
    rho_SE0 = kron(rho_S0, rho_E0)
    U = U_builder(kappa)
    rho_SE_fwd = U @ rho_SE0 @ dag(U)
    rho_S_fwd  = ptrace_rho_SE_to_S(rho_SE_fwd)

    def negF(x):
        th, ph, la = x
        V = U3(th, ph, la)
        trial = V @ rho_S_fwd @ dag(V)
        return -fidelity(rho_S0, trial)

    bounds = [(0, 2*pi), (0, 2*pi), (0, 2*pi)]
    best_val, best_x = -1.0, None
    for _ in range(restarts):
        x0 = np.random.rand(3) * 2*pi
        res = minimize(negF, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 600})
        val = -res.fun
        if val > best_val:
            best_val, best_x = val, res.x
    return {"kappa": kappa, "F_best": float(best_val), "angles": tuple(map(float, best_x))}

def sweep_best_local_undo(U_builder, label, kappas, restarts=8):
    print(f"\n=== Best local undo sweep: {label}  (INIT_STATE={INIT_STATE}) ===")
    rows = [best_local_undo_for_kappa(k, U_builder, restarts) for k in kappas]
    print("kappa\tF_best\t\t(theta, phi, lam)")
    for r in rows:
        th, ph, la = r["angles"]
        print(f"{r['kappa']:.3f}\t{r['F_best']:.6f}\t({th:.3f}, {ph:.3f}, {la:.3f})")
    return rows

# ------------------------- Main ---------------------------
def main():
    _ptrace_selfcheck()

    kappas = np.linspace(0.0, 1.0, KAPPA_POINTS)

    # Baseline sweeps
    rows_deph  = sweep(U_dephasing,   "Dephasing U = exp(-i κ σz⊗σz)", kappas)
    rows_ps    = sweep(U_partial_swap,"Partial-SWAP U = exp(-i κ/2 Σ σi⊗σi)", kappas)

    # Optimizer sweeps
    rows_best_deph = sweep_best_local_undo(U_dephasing,   "Dephasing", kappas, restarts=OPT_RESTARTS)
    rows_best_ps   = sweep_best_local_undo(U_partial_swap,"Partial-SWAP", kappas, restarts=OPT_RESTARTS)

    # Plots
    try:
        import matplotlib.pyplot as plt

        def plot_baseline(rows, title):
            ks = [r["kappa"] for r in rows]
            Fj = [r["F_joint"] for r in rows]
            Fp = [r["F_proj"]  for r in rows]
            plt.figure()
            plt.plot(ks, Fj, marker='o', label="Fidelity (joint reverse)")
            plt.plot(ks, Fp, marker='s', label="Fidelity (projected reverse)")
            plt.xlabel("κ (coupling strength)")
            plt.ylabel("Fidelity to initial ρ_S")
            plt.ylim(0, 1.05)
            plt.title(f"{title} — INIT={INIT_STATE}")
            plt.grid(True, alpha=0.3)
            plt.legend()

        def plot_best(rows, title):
            ks = [r["kappa"] for r in rows]
            Fb = [r["F_best"] for r in rows]
            plt.figure()
            plt.plot(ks, Fb, marker='^', label="Best local undo (F*)")
            plt.xlabel("κ (coupling strength)")
            plt.ylabel("Max fidelity after local undo")
            plt.ylim(0, 1.05)
            plt.title(f"Best Local Undo — {title} — INIT={INIT_STATE}")
            plt.grid(True, alpha=0.3)
            plt.legend()

        plot_baseline(rows_deph, "Dephasing Coupling")
        plot_baseline(rows_ps,   "Partial-SWAP Coupling")
        plot_best(rows_best_deph, "Dephasing")
        plot_best(rows_best_ps,   "Partial-SWAP")

        plt.show()
    except Exception as e:
        print("\n(Plotting skipped:", e, ")")

if __name__ == "__main__":
    main()
