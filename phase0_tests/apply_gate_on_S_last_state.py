#!/usr/bin/env python3
# ==============================================================
#  Reversible S+E model → emergent arrow under projection
#  - Single-shot tests
#  - Collision model (project-after-each vs keep-all)
#  - Robustness sweeps with error bands
#  - Loschmidt echo with correct ancilla popping (exact reverse)
#  - Dephasing analytics overlay
# ==============================================================

import numpy as np
from numpy import kron
from math import pi
from scipy.linalg import expm, eigvalsh
from scipy.optimize import minimize

# --- plotting QoL (you can comment these if not plotting) ---
import matplotlib as mpl
mpl.rcParams['figure.max_open_warning'] = 200

# ================== Config ==================
INIT_STATE      = "plus"     # "plus" or "zero"
KAPPA_POINTS    = 11
OPT_RESTARTS    = 6

# collision demo params (plots + prints)
COLLISION_STEPS = 10
COLLISION_KAPPAS = [0.2, 0.5, 0.8]

# ================== LinAlg helpers ==================
def dag(A): return A.conj().T

def ket(v):
    v = np.asarray(v, dtype=np.complex128).reshape(-1,1)
    n = np.linalg.norm(v)
    return v if n == 0 else v/n

def dm(psi): return psi @ dag(psi)

I2 = np.eye(2, dtype=np.complex128)
sx = np.array([[0,1],[1,0]], dtype=np.complex128)
sy = np.array([[0,-1j],[1j,0]], dtype=np.complex128)
sz = np.array([[1,0],[0,-1]], dtype=np.complex128)

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
    # Uhlmann fidelity F(rho,sigma)
    evals, evecs = np.linalg.eigh((rho + dag(rho))/2)  # ensure Hermitian
    evals = np.clip(evals, 0, None)
    sqrt_eval = np.sqrt(evals)
    sqrt_rho = (evecs * sqrt_eval) @ dag(evecs)
    M = sqrt_rho @ sigma @ sqrt_rho
    Mevals = np.linalg.eigvalsh((M + dag(M))/2)
    Mevals = np.clip(Mevals, 0, None)
    return float((np.sum(np.sqrt(Mevals)))**2)

def von_neumann_entropy(rho):
    ev = np.clip(eigvalsh((rho + dag(rho))/2), 1e-16, 1.0)
    return float(-np.sum(ev*np.log2(ev)))

def purity(rho):
    return float(np.real(np.trace(rho @ rho)))

def bloch_vector(rho):
    rx = np.real(np.trace(rho @ sx))
    ry = np.real(np.trace(rho @ sy))
    rz = np.real(np.trace(rho @ sz))
    return np.array([rx, ry, rz])

# ---------- couplings ----------
def U_dephasing(kappa):
    return expm(-1j * kappa * kron(sz, sz))

def U_partial_swap(kappa):
    H = 0.5*(kron(sx,sx) + kron(sy,sy) + kron(sz,sz))
    return expm(-1j * kappa * H)

# ---------- ancillas & initial states ----------
def ancilla_state(variant="plus", seed=None):
    rng = np.random.default_rng(seed)
    if variant == "plus":
        return ket([1,1])             # |+>
    if variant == "zero":
        return ket([1,0])             # |0>
    if variant == "rand":
        v = rng.normal(size=2) + 1j*rng.normal(size=2)
        return ket(v)
    raise ValueError("variant must be 'plus', 'zero', or 'rand'")

def initial_states():
    if INIT_STATE.lower()=="plus":
        S = ket([1,1])   # |+>
    elif INIT_STATE.lower()=="zero":
        S = ket([1,0])   # |0>
    else:
        raise ValueError("INIT_STATE must be 'plus' or 'zero'")
    E = ket([1,0])       # default |0> (overridable elsewhere)
    return dm(S), dm(E), S, E

# ---------- sanity check ----------
def _ptrace_selfcheck():
    zero = ket([1,0]); one = ket([0,1])
    bell = (kron(zero, zero)+kron(one, one))/np.sqrt(2)
    rho_S = ptrace(dm(bell), [2,2], [0])
    assert np.allclose(rho_S, 0.5*I2, atol=1e-12), "ptrace failed (Bell)."

# ---------- single-shot experiment ----------
def run_single(kappa, U_builder, anc="plus"):
    rho_S0, _rho_E0, _, _ = initial_states()
    E0 = ancilla_state(anc)
    U  = U_builder(kappa)
    rho_SE0 = kron(rho_S0, dm(E0))
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

def sweep(U_builder, label, kappas, anc="plus"):
    print(f"\n=== Sweep: {label}  (INIT_STATE={INIT_STATE}, anc={anc}) ===")
    rows=[]
    for k in kappas:
        r=run_single(k,U_builder,anc)
        rows.append({"kappa":k, **r})
    print("kappa\tF_joint\t\tF_proj\t\tS_joint\t\tS_proj")
    for r in rows:
        print(f"{r['kappa']:.3f}\t{r['F_joint']:.6f}\t{r['F_proj']:.6f}\t{r['S_joint']:.6f}\t{r['S_proj']:.6f}")
    return rows

# ---------- best local undo (optional) ----------
def U3(theta, phi, lam):
    ct, st = np.cos(theta/2), np.sin(theta/2)
    return np.array([[ct, -np.exp(1j*lam)*st],
                     [np.exp(1j*phi)*st, np.exp(1j*(phi+lam))*ct]], dtype=np.complex128)

def best_local_undo_for_kappa(kappa, U_builder, anc="plus", restarts=6):
    rho_S0, _rho_E0, _, _ = initial_states()
    U = U_builder(kappa)
    rho_S_fwd = ptrace_rho_SE_to_S(U @ kron(rho_S0, dm(ancilla_state(anc))) @ dag(U))
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

def sweep_best_local_undo(U_builder, label, kappas, anc="plus", restarts=6):
    print(f"\n=== Best local undo: {label} (INIT={INIT_STATE}, anc={anc}) ===")
    rows=[best_local_undo_for_kappa(k,U_builder,anc,restarts) for k in kappas]
    print("kappa\tF_best\t\t(theta, phi, lam)")
    for r in rows:
        th,ph,la = r["angles"]
        print(f"{r['kappa']:.3f}\t{r['F_best']:.6f}\t({th:.3f}, {ph:.3f}, {la:.3f})")
    return rows

# ================== COLLISION MODEL ==================
# (A) Project-after-each: channel on S only (density matrix 2x2)
def channel_step_S_only(rho_S, U2, ancilla="plus", seed=None):
    e = ancilla_state(ancilla, seed)
    rho_SE = U2 @ kron(rho_S, dm(e)) @ dag(U2)
    return ptrace_rho_SE_to_S(rho_SE)

# (B) Keep-all: pure-state joint evolution with exact reverse
def apply_gate_on_S_last_state(psi, U2):
    """
    Apply 2-qubit U2 to (S, last) on an n-qubit pure state |psi>.
    Order: [S, E1, ..., E_t].
    """
    n = int(np.log2(psi.size))
    assert 2**n == psi.size and n >= 2
    mid = 1 << (n-2)
    psi3 = psi.reshape(2, mid, 2)                 # (S, mid, last)
    psi_sl = np.transpose(psi3, (0,2,1)).reshape(4, mid)   # rows=(S,last)
    psi_sl_new = U2 @ psi_sl
    psi3_new = np.transpose(psi_sl_new.reshape(2,2,mid), (0,2,1))
    return psi3_new.reshape(-1,1)

def pop_last_ancilla(psi, e):
    """
    Contract the last qubit with <e|, returning an (n-1)-qubit state.
    """
    n = int(np.log2(psi.size))
    assert 2**n == psi.size and n >= 2
    rest = 1 << (n-1)
    psi3 = psi.reshape(rest, 2)          # (rest, last)
    out  = psi3 @ e.conj()
    return out.reshape(-1,1) / np.linalg.norm(out)

def collision_model(U_builder, kappa, N=10, ancilla="plus", rand_seed=0):
    rho_S0, _, S0, _ = initial_states()
    U2 = U_builder(kappa)

    # --- Project-after-each path ---
    rng = np.random.default_rng(rand_seed)
    rho_S = rho_S0.copy()
    fid_proj=[]; ent_proj=[]
    for _ in range(N):
        rho_S = channel_step_S_only(rho_S, U2, ancilla=ancilla, seed=rng.integers(1e9))
        fid_proj.append(fidelity(rho_S0, rho_S))
        ent_proj.append(von_neumann_entropy(rho_S))

    # --- Keep-all path (pure state) with exact reverse using ancilla popping ---
    rng = np.random.default_rng(rand_seed)
    e_hist = []
    psi = kron(S0, ancilla_state(ancilla, rng.integers(1e9)))
    e_hist.append(psi[-2:,:].copy())  # we overwrite with the sampled ket below for consistency
    e_hist = [ancilla_state(ancilla, rng.integers(1e9))]
    psi = kron(S0, e_hist[0])

    fid_joint=[]; ent_joint=[]

    def rhoS(psi_vec):
        n = int(np.log2(psi_vec.size)); rest = 1 << (n-1)
        M = psi_vec.reshape(2, rest)
        return M @ M.conj().T

    for t in range(N):
        if t>0:
            e_hist.append(ancilla_state(ancilla, rng.integers(1e9)))
            psi = kron(psi, e_hist[-1])
        psi = apply_gate_on_S_last_state(psi, U2)
        rs = rhoS(psi)
        fid_joint.append(fidelity(rho_S0, rs))
        ent_joint.append(von_neumann_entropy(rs))

    # exact global reverse: U† then pop <e_t|
    for t in range(N-1, -1, -1):
        psi = apply_gate_on_S_last_state(psi, dag(U2))
        psi = pop_last_ancilla(psi, e_hist[t])

    rs_end = rhoS(psi)
    F_end = fidelity(rho_S0, rs_end)
    S_end = von_neumann_entropy(rs_end)

    return {
        "proj":{"fid":np.array(fid_proj),"ent":np.array(ent_proj)},
        "joint":{"fid":np.array(fid_joint),"ent":np.array(ent_joint)},
        "final_global_recovery":{"F":F_end,"S":S_end},
        "N":N, "kappa":kappa
    }

# ================== ROBUSTNESS / ANALYTICS PACK ==================
def loschmidt_echo_collisions(U_builder, kappa, S0, E_mode="rand", N=10, seed=0):
    rng = np.random.default_rng(seed)
    U2 = U_builder(kappa)
    # forward
    e_list = [ancilla_state(E_mode, rng.integers(1e9))]
    psi = kron(S0, e_list[0])
    L_keep = []
    def rhoS(psi_vec):
        n = int(np.log2(psi_vec.size)); rest = 1 << (n-1)
        M = psi_vec.reshape(2, rest); return M @ M.conj().T
    for t in range(N):
        if t>0:
            e_list.append(ancilla_state(E_mode, rng.integers(1e9)))
            psi = kron(psi, e_list[-1])
        psi = apply_gate_on_S_last_state(psi, U2)
        L_keep.append(fidelity(dm(S0), rhoS(psi)))
    # reverse
    for t in range(N-1, -1, -1):
        psi = apply_gate_on_S_last_state(psi, dag(U2))
        psi = pop_last_ancilla(psi, e_list[t])
    L_end = fidelity(dm(S0), rhoS(psi))
    return np.array(L_keep), L_end

def mutual_info_S_E(rho_SE):
    rho_S = ptrace(rho_SE, [2,2], [0])
    rho_E = ptrace(rho_SE, [2,2], [1])
    return (von_neumann_entropy(rho_S)
            + von_neumann_entropy(rho_E)
            - von_neumann_entropy(rho_SE))

def analytic_dephasing_collisions(kappa, n, S_mode="plus", E_mode="plus"):
    # For S=|+>, E=|+>, U=exp(-i k σz⊗σz): coherence factor per collision c=cos(2k)
    # Fidelity to |+><+| after n collisions: F = (1 + c^n)/2
    if S_mode=="plus" and E_mode in ("plus","rand"):
        c = np.cos(2*kappa)
        return 0.5*(1 + c**n)
    return None

def sweep_random(
    U_builder,
    kappas=np.linspace(0.1, 1.0, 10),
    steps=12,
    S_mode="rand",
    E_mode="rand",
    trials=64,
    seed=42,
):
    rng = np.random.default_rng(seed)
    out = {}
    for k in kappas:
        U2 = U_builder(k)
        F_proj = np.zeros((trials, steps))
        S_proj = np.zeros((trials, steps))
        Purity  = np.zeros((trials, steps))
        Bloch   = np.zeros((trials, steps, 3))
        L_keep  = np.zeros((trials, steps))
        L_end   = np.zeros(trials)
        for t in range(trials):
            # random S
            v = rng.normal(size=2) + 1j*rng.normal(size=2)
            S0 = ket(v) if S_mode=="rand" else ancilla_state(S_mode, rng.integers(1e9))
            rhoS = dm(S0)
            # project-after-each
            for s in range(steps):
                e = ancilla_state(E_mode, rng.integers(1e9))
                rho_SE = U2 @ kron(rhoS, dm(e)) @ dag(U2)
                rhoS   = ptrace_rho_SE_to_S(rho_SE)
                F_proj[t,s] = fidelity(dm(S0), rhoS)
                S_proj[t,s] = von_neumann_entropy(rhoS)
                Purity[t,s] = purity(rhoS)
                Bloch[t,s]  = bloch_vector(rhoS)
            # keep-all Loschmidt
            L_keep[t], L_end[t] = loschmidt_echo_collisions(U_builder, k, S0, E_mode, N=steps, seed=rng.integers(1e9))
        out[k] = {
            "F_proj_mean": F_proj.mean(0), "F_proj_std": F_proj.std(0),
            "S_proj_mean": S_proj.mean(0), "S_proj_std": S_proj.std(0),
            "purity_mean": Purity.mean(0), "purity_std": Purity.std(0),
            "bloch_mean":  Bloch.mean(0),  "bloch_std":  Bloch.std(0),
            "L_keep_mean": L_keep.mean(0), "L_keep_std": L_keep.std(0),
            "L_end_mean":  L_end.mean(),   "L_end_std":  L_end.std(),
        }
    return out

# ---------- plotting helpers (safe to inline) ----------
def plot_with_bands(x, y, ystd, label):
    import matplotlib.pyplot as plt
    plt.plot(x, y, label=label)
    lo, hi = y-ystd, y+ystd
    plt.fill_between(x, lo, hi, alpha=0.15)

def run_bulletproof_suite():
    import matplotlib.pyplot as plt

    steps = 12
    kappas = [0.2, 0.5, 0.8]

    # 1) Random ancillas + random initial states
    for label, Ub in [("Dephasing", U_dephasing), ("Partial-SWAP", U_partial_swap)]:
        print(f"\n### Robustness sweep: {label} (random S, random ancillas)")
        res = sweep_random(Ub, kappas=kappas, steps=steps, S_mode="rand", E_mode="rand", trials=64)
        x = np.arange(1, steps+1)
        for k in kappas:
            R = res[k]
            # Fidelity: project vs keep-all mean
            plt.figure()
            plot_with_bands(x, R["F_proj_mean"], R["F_proj_std"], "F (project)")
            plot_with_bands(x, R["L_keep_mean"], R["L_keep_std"], "F (keep-all)")
            plt.xlabel("Collision #"); plt.ylabel("Fidelity to initial ρ_S")
            plt.ylim(0,1.05); plt.grid(True, alpha=0.3)
            plt.title(f"{label}: Robustness — κ={k} (rand S & ancillas)")
            plt.legend()

            # Entropy & Purity
            plt.figure()
            plot_with_bands(x, R["S_proj_mean"], R["S_proj_std"], "Entropy S(ρ_S)")
            plot_with_bands(x, R["purity_mean"], R["purity_std"], "Purity Tr(ρ_S^2)")
            plt.xlabel("Collision #"); plt.ylabel("Value")
            plt.grid(True, alpha=0.3); plt.title(f"{label}: S and Purity — κ={k}")
            plt.legend()

            # Bloch shrinkage norm ||r||
            rnorm = np.linalg.norm(R["bloch_mean"], axis=1)
            rstd  = np.linalg.norm(R["bloch_std"], axis=1)
            plt.figure()
            plot_with_bands(x, rnorm, rstd, "‖Bloch r‖")
            plt.xlabel("Collision #"); plt.ylabel("Bloch norm")
            plt.grid(True, alpha=0.3); plt.title(f"{label}: Bloch shrinkage — κ={k}")
            plt.legend()

            print(f"Loschmidt final (keep-all) mean±std for κ={k}: {R['L_end_mean']:.6f} ± {R['L_end_std']:.6f}")

    # 2) Analytics overlay for dephasing with S=|+>, E=|+>
    print("\n### Analytics overlay: dephasing, S=|+>, E=|+>")
    x = np.arange(1, steps+1)
    for k in [0.2, 0.5, 0.8]:
        rho_S0, _, S0, _ = initial_states()
        U2 = U_dephasing(k)
        rhoS = rho_S0.copy()
        F = []
        for n in range(1, steps+1):
            e = ket([1,1])  # |+>
            rho_SE = U2 @ kron(rhoS, dm(e)) @ dag(U2)
            rhoS = ptrace_rho_SE_to_S(rho_SE)
            F.append(fidelity(rho_S0, rhoS))
        F = np.array(F)
        Fan = np.array([analytic_dephasing_collisions(k, n, "plus", "plus") for n in range(1, steps+1)])
        plt.figure()
        plt.plot(x, F, marker='o', label="numeric")
        if Fan[0] is not None:
            plt.plot(x, Fan, marker='s', label="analytic")
        plt.xlabel("Collision #"); plt.ylabel("Fidelity to initial ρ_S")
        plt.title(f"Dephasing collisions — κ={k} — numeric vs analytic")
        plt.ylim(0,1.05); plt.grid(True, alpha=0.3); plt.legend()

    # 3) Single-shot mutual information (dephasing)
    print("\n### Single-shot mutual information (dephasing)")
    for k in np.linspace(0,1.0,6):
        S0 = ket([1,1]); E0 = ket([1,1])
        rho_SE0 = kron(dm(S0), dm(E0))
        U2 = U_dephasing(k)
        rho_SEf = U2 @ rho_SE0 @ dag(U2)
        I_SE = mutual_info_S_E(rho_SEf)
        FS   = fidelity(dm(S0), ptrace_rho_SE_to_S(rho_SEf))
        print(f"k={k:.2f} | I(S:E)={I_SE:.6f} | F_proj={FS:.6f}")

    try:
        import matplotlib.pyplot as plt
        plt.show()
    except Exception:
        pass

# ================== Main ==================
def main():
    _ptrace_selfcheck()
    kappas = np.linspace(0.0, 1.0, KAPPA_POINTS)

    # Baseline single-shot sweeps (ancilla in |+> to show dephasing decoherence)
    sweep(U_dephasing,    "Dephasing U = exp(-i κ σz⊗σz)", kappas, anc="plus")
    sweep(U_partial_swap, "Partial-SWAP U = exp(-i κ/2 Σ σi⊗σi)", kappas, anc="plus")

    # Best local undo (optional)
    sweep_best_local_undo(U_dephasing,    "Dephasing", kappas, anc="plus", restarts=OPT_RESTARTS)
    sweep_best_local_undo(U_partial_swap, "Partial-SWAP", kappas, anc="plus", restarts=OPT_RESTARTS)

    # Collision model demo with exact global reverse (prints & plots)
    for label, Ub in [("Dephasing", U_dephasing), ("Partial-SWAP", U_partial_swap)]:
        print(f"\n=== Collision Model: {label}  (INIT_STATE={INIT_STATE}, anc=rand) ===")
        for k in COLLISION_KAPPAS:
            out = collision_model(Ub, k, N=COLLISION_STEPS, ancilla="rand", rand_seed=1234)
            print(f"kappa={k:.2f} | Global reverse end: F={out['final_global_recovery']['F']:.12f}, S={out['final_global_recovery']['S']:.12f}")

            # quick plots
            try:
                import matplotlib.pyplot as plt
                steps = np.arange(1, out["N"]+1)
                plt.figure()
                plt.plot(steps, out["proj"]["fid"], marker='s', label="F (project-after-each)")
                plt.plot(steps, out["joint"]["fid"], marker='o', label="F (keep-all)")
                plt.xlabel("Collision #"); plt.ylabel("Fidelity to initial ρ_S")
                plt.title(f"{label}: Fidelity vs collisions — κ={k} — INIT={INIT_STATE}")
                plt.ylim(0,1.05); plt.grid(True, alpha=0.3); plt.legend()

                plt.figure()
                plt.plot(steps, out["proj"]["ent"], marker='s', label="S (project)")
                plt.plot(steps, out["joint"]["ent"], marker='o', label="S (keep-all)")
                plt.xlabel("Collision #"); plt.ylabel("von Neumann entropy S(ρ_S)")
                plt.title(f"{label}: Entropy vs collisions — κ={k} — INIT={INIT_STATE}")
                plt.ylim(0,1.05); plt.grid(True, alpha=0.3); plt.legend()
            except Exception:
                pass

    # Robustness + analytics pack
    run_bulletproof_suite()

if __name__ == "__main__":
    main()
