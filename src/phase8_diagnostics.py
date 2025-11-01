#!/usr/bin/env python3
# Phase VIII diagnostics:
#   * Generates Haar-isometry or Haar-unitary samples (GPU-accelerated when --device cuda)
#   * Or ingests an existing per-D CSV (Phase VI/Phase VII output)
#   * Tests several D choices and Y normalisations to reconcile intercept/variance exponents
#   * Produces CSV summaries + comparison plots

from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import cupy as cp  # type: ignore

    _HAS_CUPY = True
except Exception:  # pragma: no cover
    cp = None
    _HAS_CUPY = False

try:
    from scipy.stats import kurtosis as _kurt
    from scipy.stats import skew as _skew
    from scipy.stats import theilslopes

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

    def _skew(x):
        x = np.asarray(x)
        m = x.mean()
        s = x.std() + 1e-12
        return float(np.mean(((x - m) / s) ** 3))

    def _kurt(x):
        x = np.asarray(x)
        m = x.mean()
        s = x.std() + 1e-12
        return float(np.mean(((x - m) / s) ** 4) - 3.0)

EPS = 1e-12
RSEED = 0xBADA55


# ---------------------- helper for backend ---------------------- #

def _to_numpy(arr):
    if _HAS_CUPY and cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def _make_rngs(device: str, seed: int):
    rng_cpu = np.random.default_rng(seed)
    rng_gpu = None
    xp = np
    if device == "cuda":
        if not _HAS_CUPY:
            raise RuntimeError("CuPy not available. Install cupy-cuda11x (or matching wheel).")
        xp = cp  # type: ignore
        rng_gpu = cp.random.default_rng(seed)  # type: ignore
    return xp, rng_cpu, rng_gpu


def _complex_gaussian(shape, xp, rng_cpu, rng_gpu):
    if xp is np:
        return rng_cpu.normal(size=shape) + 1j * rng_cpu.normal(size=shape)
    rng = rng_gpu if rng_gpu is not None else xp.random.default_rng()
    return rng.standard_normal(size=shape) + 1j * rng.standard_normal(size=shape)


# ---------------------- math helpers ---------------------- #

def binary_entropy_bits(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    A = np.column_stack((x, np.ones_like(x)))
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


def bootstrap_line(x: np.ndarray, y: np.ndarray, B: int = 2000, seed: int = RSEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    slopes = np.empty(B)
    intercepts = np.empty(B)
    ones = np.ones(n)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        A = np.column_stack((x[idx], ones))
        slopes[i], intercepts[i] = np.linalg.lstsq(A, y[idx], rcond=None)[0]
    slo, shi = np.percentile(slopes, [2.5, 97.5])
    ilo, ihi = np.percentile(intercepts, [2.5, 97.5])
    return (float(slopes.mean()), float(intercepts.mean())), (float(slo), float(shi)), (float(ilo), float(ihi))


def bootstrap_slope_loglog(Dvals: np.ndarray, vY: np.ndarray, B: int = 2000, seed: int = RSEED):
    rng = np.random.default_rng(seed)
    lx = np.log10(Dvals)
    ly = np.log10(np.maximum(vY, EPS))
    n = len(lx)
    slopes = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        slopes[b], _ = linear_fit(lx[idx], ly[idx])
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return float(slopes.mean()), (float(lo), float(hi))


# ---------------------- sampling ---------------------- #

def sample_random_states(nS: int, env_dim_log: int, trials: int, device: str, seed: int):
    """
    Draw random pure states on C^{dS * env_dim} with dS = 2^{nS}, env_dim = 2^{env_dim_log}.
    Returns numpy arrays (X, A2, Ibits).
    """
    xp, rng_cpu, rng_gpu = _make_rngs(device, seed)
    dS = 2 ** nS
    env_dim = 2 ** env_dim_log
    z = _complex_gaussian((trials, dS * env_dim), xp, rng_cpu, rng_gpu)
    norms = xp.linalg.norm(z, axis=1, keepdims=True) + 1e-12
    psi = (z / norms).reshape(trials, dS, env_dim)
    rhoS = xp.einsum("tse,tje->tsj", psi, xp.conj(psi))  # (trials, dS, dS)
    rhoS_np = _to_numpy(rhoS)

    a = rhoS_np[:, 0, 0].real
    b = rhoS_np[:, 0, 1]
    purity = a * a + (1.0 - a) * (1.0 - a) + 2.0 * (b.real * b.real + b.imag * b.imag)
    purity = np.clip(purity, 1.0 / dS, 1.0)
    deff = 1.0 / np.maximum(purity, EPS)
    X = np.maximum(deff - 1.0, EPS)

    s = np.sqrt(np.maximum(2.0 * purity - 1.0, 0.0))
    lam1 = (1.0 + s) * 0.5
    Ibits = 2.0 * binary_entropy_bits(lam1)

    F = np.clip(a, 0.0, 1.0)
    A2 = np.arccos(np.sqrt(F)) ** 2

    # release GPU workspace
    if xp is not np:
        cp.get_default_memory_pool().free_all_blocks()  # type: ignore

    return X.astype(float), A2.astype(float), np.maximum(Ibits, EPS)


# ---------------------- Y variants ---------------------- #

def compute_Y_variants(X: np.ndarray, A2: np.ndarray, Ibits: np.ndarray, dS: int) -> Dict[str, np.ndarray]:
    Y0 = np.sqrt(np.maximum(X, EPS)) * A2 / np.maximum(Ibits, EPS)
    Y1 = A2 / np.maximum(Ibits, EPS)
    Y2 = np.sqrt(np.maximum(X, EPS) / max(dS - 1, 1)) * A2 / np.maximum(Ibits, EPS)
    return {"orig": Y0, "no_sqrt": Y1, "scaled_ds": Y2}


def alpha_from_cloud(X: np.ndarray, Y: np.ndarray):
    mask = (X > 0) & (Y > 0)
    if mask.sum() < 3:
        return float("nan"), float("nan"), float("nan"), int(mask.sum())
    lx = np.log10(X[mask])
    ly = np.log10(Y[mask])
    a, b = linear_fit(lx, ly)
    if _HAS_SCIPY:
        slope_ts, intercept_ts, _, _ = theilslopes(ly, lx)
    else:
        slopes = []
        for i in range(len(lx)):
            dx = lx[i + 1 :] - lx[i]
            dy = ly[i + 1 :] - ly[i]
            mask_nonzero = dx != 0
            slopes.extend((dy[mask_nonzero] / dx[mask_nonzero]).tolist())
        slope_ts = float(np.median(slopes)) if slopes else a
        intercept_ts = float(np.median(ly - slope_ts * lx))
    return float(a), float(slope_ts), float(b), int(mask.sum())


# ---------------------- record ---------------------- #


@dataclass
class PerSize:
    model: str
    dS: int
    env_param: int
    D_total: int
    D_stinespring: int
    D_env_kraus: int
    invD_total: float
    invD_stine: float
    invD_envk: float
    alpha_ols: float
    alpha_theilsen: float
    varY: float
    meanY: float
    skewY: float
    kurtY: float
    y_norm: str
    samples: int
    device: str


# ---------------------- generation pipeline ---------------------- #


def run_generate(mode: str, nS: int, params: List[int], trials: int, seed: int, outdir: Path, device: str):
    records: List[PerSize] = []
    for param in params:
        local_seed = seed ^ (param * 0x9E3779B1)
        if mode == "isometry":
            X, A2, Ibits = sample_random_states(nS, param, trials, device, local_seed)
            dS = 2 ** nS
            env_dim = 2 ** param
            D_total = D_stine = dS * env_dim
            D_envk = env_dim
            label = "haar_isometry"
        else:
            X, A2, Ibits = sample_random_states(nS, param, trials, device, local_seed)
            dS = 2 ** nS
            dE = 2 ** param
            D_total = dS * dE
            D_stine = dS * dE
            D_envk = dE * (2 ** param)  # treat Kraus rank as 2^{param}
            label = "haar_unitary"

        Y_variants = compute_Y_variants(X, A2, Ibits, dS)
        for yname, Y in Y_variants.items():
            a_ols, a_ts, b, n = alpha_from_cloud(X, Y)
            records.append(
                PerSize(
                    model=label,
                    dS=dS,
                    env_param=param,
                    D_total=D_total,
                    D_stinespring=D_stine,
                    D_env_kraus=D_envk,
                    invD_total=1.0 / max(D_total, 1),
                    invD_stine=1.0 / max(D_stine, 1),
                    invD_envk=1.0 / max(D_envk, 1),
                    alpha_ols=a_ols,
                    alpha_theilsen=a_ts,
                    varY=float(np.var(Y)),
                    meanY=float(np.mean(Y)),
                    skewY=_skew(Y),
                    kurtY=_kurt(Y),
                    y_norm=yname,
                    samples=n,
                    device=device,
                )
            )

    if records:
        per_path = outdir / "data" / "phase8_per_size.csv"
        with per_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
            w.writeheader()
            for r in records:
                w.writerow(asdict(r))
    return records


def run_ingest(perD_csv: Path) -> List[PerSize]:
    records: List[PerSize] = []
    with perD_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            D = int(float(row.get("D", "0")))
            invD = float(row.get("invD", 1.0 / D if D > 0 else 0.0))
            alpha = float(row.get("alpha", "nan"))
            varY = float(row.get("varY", "nan"))
            meanY = float(row.get("meanY", "nan"))
            n = int(float(row.get("n", "0")))
            records.append(
                PerSize(
                    model="ingested",
                    dS=-1,
                    env_param=-1,
                    D_total=D,
                    D_stinespring=D,
                    D_env_kraus=D,
                    invD_total=invD,
                    invD_stine=invD,
                    invD_envk=invD,
                    alpha_ols=alpha,
                    alpha_theilsen=alpha,
                    varY=varY,
                    meanY=meanY,
                    skewY=float("nan"),
                    kurtY=float("nan"),
                    y_norm="orig",
                    samples=n,
                    device="n/a",
                )
            )
    return records


# ---------------------- aggregation ---------------------- #


def aggregate(records: List[PerSize], outdir: Path, title_tag: str):
    if not records:
        print("[phase8] No records to aggregate.")
        return

    combos = [
        ("D_total", lambda r: (r.D_total, r.invD_total)),
        ("D_stinespring", lambda r: (r.D_stinespring, r.invD_stine)),
        ("D_env_kraus", lambda r: (r.D_env_kraus, r.invD_envk)),
    ]
    y_norms = sorted({r.y_norm for r in records})

    alpha_rows = []
    var_rows = []

    figA, axA = plt.subplots(figsize=(7.0, 4.6))
    figV, axV = plt.subplots(figsize=(7.6, 4.8))
    color_map = {"orig": "C0", "no_sqrt": "C1", "scaled_ds": "C2"}

    for yname in y_norms:
        rs_y = [r for r in records if r.y_norm == yname]
        if not rs_y:
            continue

        for dlabel, getter in combos:
            Ds = np.array([getter(r)[0] for r in rs_y], float)
            invD = np.array([getter(r)[1] for r in rs_y], float)
            abs_alpha = np.array([abs(r.alpha_ols) for r in rs_y], float)
            varY = np.array([r.varY for r in rs_y], float)

            if len(Ds) < 3:
                continue

            # Intercept metrics
            slope, intercept = linear_fit(invD, abs_alpha)
            (_, _), (slo, shi), (ilo, ihi) = bootstrap_line(invD, abs_alpha, B=1500)
            if _HAS_SCIPY:
                slope_ts, intercept_ts, _, _ = theilslopes(abs_alpha, invD)
            else:
                slope_ts, intercept_ts = slope, intercept

            alpha_rows.append(
                {
                    "y_norm": yname,
                    "D_choice": dlabel,
                    "slope_ols": slope,
                    "slope_lo": slo,
                    "slope_hi": shi,
                    "intercept_ols": intercept,
                    "intercept_lo": ilo,
                    "intercept_hi": ihi,
                    "slope_theilsen": slope_ts,
                    "intercept_theilsen": intercept_ts,
                    "num_points": len(Ds),
                    "device": rs_y[0].device,
                }
            )

            xs = np.linspace(0, invD.max() * 1.05, 200)
            axA.scatter(invD, abs_alpha, s=35, alpha=0.75, label=f"{yname}/{dlabel}")
            axA.plot(xs, slope * xs + intercept, "--", color=color_map.get(yname, "C0"), alpha=0.6)

            # variance scaling
            slopeV, (vlo, vhi) = bootstrap_slope_loglog(Ds, varY, B=1500)
            best_p = 1 if abs(slopeV + 1.0) < abs(slopeV + 2.0) else 2
            var_rows.append(
                {
                    "y_norm": yname,
                    "D_choice": dlabel,
                    "slope_loglog": slopeV,
                    "slope_lo": vlo,
                    "slope_hi": vhi,
                    "best_flatten_p_in_{1,2}": best_p,
                    "device": rs_y[0].device,
                }
            )

            if dlabel == "D_total":
                gx = np.logspace(np.log10(Ds.min() * 0.9), np.log10(Ds.max() * 1.1), 200)
                aV, bV = linear_fit(np.log10(Ds), np.log10(np.maximum(varY, EPS)))
                gy = 10 ** (aV * np.log10(gx) + bV)
                axV.scatter(Ds, varY, s=45, alpha=0.8, label=f"{yname} (slope {slopeV:+.2f})")
                axV.plot(gx, gy, "--", color=color_map.get(yname, "C0"), alpha=0.6)

    axA.set_title(r"$|\alpha|$ vs $1/D$ — intercept audit (" + title_tag + ")")
    axA.set_xlabel(r"$1/D$")
    axA.set_ylabel(r"$|\alpha|$")
    handles, labels = axA.get_legend_handles_labels()
    if handles:
        axA.legend(handles, labels, fontsize=8, ncol=2)
    figA.tight_layout()
    figA.savefig(outdir / "figures" / "phase8_alpha_intercepts.png", dpi=160)
    plt.close(figA)

    axV.set_title(r"Variance scaling of $Y$ vs $D$ — normalisation audit (" + title_tag + ")")
    axV.set_xlabel(r"$D$")
    axV.set_ylabel(r"$\mathrm{Var}(Y)$")
    axV.set_xscale("log")
    axV.set_yscale("log")
    gx = np.logspace(2, 5, 200)
    axV.plot(gx, (gx / gx[0]) ** (-1), ":", color="gray", alpha=0.6, label=r"$D^{-1}$ ref")
    axV.plot(gx, (gx / gx[0]) ** (-2), ":", color="silver", alpha=0.6, label=r"$D^{-2}$ ref")
    handles_v, labels_v = axV.get_legend_handles_labels()
    if handles_v:
        axV.legend(handles_v, labels_v, fontsize=8)
    figV.tight_layout()
    figV.savefig(outdir / "figures" / "phase8_var_scaling.png", dpi=160)
    plt.close(figV)

    if alpha_rows:
        a_path = outdir / "data" / "phase8_alpha_intercepts.csv"
        with a_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(alpha_rows[0].keys()))
            w.writeheader()
            w.writerows(alpha_rows)

    if var_rows:
        v_path = outdir / "data" / "phase8_var_scaling.csv"
        with v_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(var_rows[0].keys()))
            w.writeheader()
            w.writerows(var_rows)

    print(
        "[phase8] Wrote:\n"
        f"  {outdir/'figures'/'phase8_alpha_intercepts.png'}\n"
        f"  {outdir/'figures'/'phase8_var_scaling.png'}"
    )


# ---------------------- CLI ---------------------- #


def main():
    ap = argparse.ArgumentParser(description="Phase VIII — intercept/variance diagnostics")
    ap.add_argument("--mode", choices=["isometry", "unitary", "ingest"], default="isometry")
    ap.add_argument("--nS", type=int, default=1, help="system qubits (nS=1 fastest)")
    ap.add_argument("--m-min", type=int, default=6, help="min log2 env dim for isometry")
    ap.add_argument("--m-max", type=int, default=14, help="max log2 env dim for isometry")
    ap.add_argument("--nE-min", type=int, default=6, help="min env qubits for unitary mode")
    ap.add_argument("--nE-max", type=int, default=12, help="max env qubits for unitary mode")
    ap.add_argument("--trials", type=int, default=4000, help="trials per size")
    ap.add_argument("--seed", type=int, default=RSEED, help="base RNG seed")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="backend for generation")
    ap.add_argument("--perD", type=str, default="", help="Per-D CSV path (for --mode ingest)")
    ap.add_argument("--outdir", type=str, default="phase8-out")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "data").mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    if args.mode == "ingest":
        if not args.perD:
            raise SystemExit("Provide --perD path when using --mode ingest.")
        records = run_ingest(Path(args.perD))
        title = "ingested CSV"
    elif args.mode == "isometry":
        params = list(range(args.m_min, args.m_max + 1))
        records = run_generate("isometry", args.nS, params, args.trials, args.seed, outdir, args.device)
        title = f"Haar isometry ({args.device})"
    else:  # unitary
        params = list(range(args.nE_min, args.nE_max + 1))
        records = run_generate("unitary", args.nS, params, args.trials, args.seed, outdir, args.device)
        title = f"Haar unitary ({args.device})"

    aggregate(records, outdir, title)


if __name__ == "__main__":
    main()


