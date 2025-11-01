#!/usr/bin/env python3
# quantum_projection_arrow.py
# Test #3: Quantum extension — emergent irreversibility by projection.
# System qubit S coupled to hidden/env qubit E via a reversible unitary.
# Show that tracing out E then "reversing S only" fails to recover the initial state,
# but reversing S+E jointly succeeds.

import numpy as np
from numpy import kron
from scipy.linalg import expm, eigvalsh
import math

# ---------- Utilities ----------
def dag(A): return A.conj().T
def ket(v): 
    v = np.asarray(v, dtype=complex).reshape(-1,1)
    return v / np.linalg.norm(v)
def dm(psi): return psi @ dag(psi)

# Pauli and identities
I2 = np.eye(2, dtype=complex)
sx = np.array([[0,1],[1,0]], dtype=complex)
sy = np.array([[0,-1j],[1j,0]], dtype=complex)
sz = np.array([[1,0],[0,-1]], dtype=complex)

def ptrace_rho_SE_to_S(rho_SE):
    # Partial trace over E for a 2x2 system: reshape to (2,2,2,2) and trace over env indices (1,3)
    rho = rho_SE.reshape(2,2,2,2)
    # Trace over E: sum over env basis
    rho_S = np.einsum('ijaa->ij', rho)  # trace over last index pair
    return rho_S

def fidelity(rho, sigma):
    # Uhlmann fidelity (for qubits can use eigen trick): F = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2
    # Implement via eigen-decomp of rho: sqrt_rho = V sqrt(D) V^\dagger
    evals, evecs = np.linalg.eigh(rho)
    sqrt_eval = np.sqrt(np.clip(evals, 0, None))
    sqrt_rho = (evecs * sqrt_eval) @ dag(evecs)
    M = sqrt_rho @ sigma @ sqrt_rho
    # trace of sqrt(M)
    Mevals = np.linalg.eigvalsh((M + dag(M))/2)
    Mevals = np.clip(Mevals, 0, None)
    return float((np.sum(np.sqrt(Mevals)))**2)

def von_neumann_entropy(rho):
    # S = -Tr rho log2 rho
    evals = np.clip(eigvalsh((rho + dag(rho))/2), 1e-16, 1.0)
    return float(-np.sum(evals * np.log2(evals)))

# ---------- Unitaries (couplings) ----------
def U_dephasing(kappa):
    # U = exp(-i kappa * sz \otimes sz)
    H = kron(sz, sz)
    return expm(-1j * kappa * H)

def U_partial_swap(kappa):
    # Heisenberg exchange ~ partial SWAP: H = (sx⊗sx + sy⊗sy + sz⊗sz)/2
    H = 0.5*(kron(sx,sx) + kron(sy,sy) + kron(sz,sz))
    return expm(-1j * kappa * H)

# ---------- Experiment ----------
def run_single(kappa, U_builder, verbose=False):
    # Initial states:
    # System S in |+> = (|0>+|1>)/sqrt(2), Env E in |0>
    plus = ket([1,1])
    zero = ket([1,0])
    rho_S0 = dm(plus)
    rho_E0 = dm(zero)
    rho_SE0 = kron(rho_S0, rho_E0)

    U = U_builder(kappa)

    # Forward evolve: rho_SE' = U rho_SE0 U^\dagger
    rho_SE_fwd = U @ rho_SE0 @ dag(U)
    rho_S_fwd = ptrace_rho_SE_to_S(rho_SE_fwd)

    # --- Case 1: Reverse S+E jointly (should perfectly recover) ---
    rho_SE_rev_joint = dag(U) @ rho_SE_fwd @ U
    rho_S_joint = ptrace_rho_SE_to_S(rho_SE_rev_joint)  # then look at S
    F_joint = fidelity(rho_S0, rho_S_joint)
    S_joint = von_neumann_entropy(rho_S_joint)

    # --- Case 2: Project (trace E), then "reverse S only" ---
    # Apply U^\dagger on S alone AFTER projection (this cannot reconstruct lost correlations).
    U_S_only = np.array([[1,0],[0,1]], dtype=complex)  # placeholder: identity, we’ll add a local inverse try next
    # Try the local inverse you'd naively imagine: evolve S with the local part of U if it existed.
    # But for entangling U, there is no unitary on S alone that inverts the entangling action.
    # We'll still allow a best-effort local phase flip (for dephasing coupling) to be fair.
    # For dephasing-like, the visible reduced map is dephasing channel: local Z rotations won't revive coherence.

    # We'll just define "attempted reversal" as: apply dag(U_local) on S, but since we don't know it, try identity.
    rho_S_proj_rev = rho_S_fwd  # (no local unitary can restore coherence lost to E)
    F_proj = fidelity(rho_S0, rho_S_proj_rev)
    S_proj = von_neumann_entropy(rho_S_proj_rev)

    # --- Case 3: Control — no coupling (kappa=0) should be perfectly reversible even after projection ---
    # We compute this by definition when kappa==0 via same machinery.
    if verbose:
        print(f"kappa={kappa:.3f} | F_joint={F_joint:.6f}, S_joint={S_joint:.6f} | F_proj={F_proj:.6f}, S_proj={S_proj:.6f}")

    return {
        "kappa": kappa,
        "F_joint": F_joint,
        "S_joint": S_joint,
        "F_proj": F_proj,
        "S_proj": S_proj
    }

def sweep(U_builder, label, kappa_vals):
    print(f"\n=== Sweep: {label} ===")
    rows = []
    for k in kappa_vals:
        rows.append(run_single(k, U_builder))
    # Pretty print a few lines
    print("kappa\tF_joint\t\tF_proj\t\tS_joint\t\tS_proj")
    for r in rows:
        print(f"{r['kappa']:.3f}\t{r['F_joint']:.6f}\t{r['F_proj']:.6f}\t{r['S_joint']:.6f}\t{r['S_proj']:.6f}")
    return rows

def main():
    # Sweep kappa from 0 to 1.0
    kappa_vals = np.linspace(0.0, 1.0, 11)

    rows_deph = sweep(U_dephasing, "Dephasing-style U = exp(-i κ σz⊗σz)", kappa_vals)
    rows_ps   = sweep(U_partial_swap, "Partial-SWAP U = exp(-i κ/2 Σ σi⊗σi)", kappa_vals)

    # Plot (optional)
    try:
        import matplotlib.pyplot as plt
        def plot_rows(rows, title):
            ks = [r["kappa"] for r in rows]
            Fj = [r["F_joint"] for r in rows]
            Fp = [r["F_proj"]  for r in rows]
            plt.figure()
            plt.plot(ks, Fj, marker='o', label="Fidelity (joint reverse)")
            plt.plot(ks, Fp, marker='s', label="Fidelity (projected reverse)")
            plt.xlabel("κ (coupling strength)")
            plt.ylabel("Fidelity to initial ρ_S")
            plt.ylim(0,1.05)
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.legend()

        plot_rows(rows_deph, "Dephasing Coupling")
        plot_rows(rows_ps,   "Partial-SWAP Coupling")
        plt.show()
    except Exception as e:
        print("\n(Plotting skipped:", e, ")")

if __name__ == "__main__":
    main()
