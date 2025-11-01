#!/usr/bin/env python3
"""Figure build orchestrator for the Quantum Formula project."""

from __future__ import annotations

import argparse
import ast
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import types

import numpy as np

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "figs"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.35,
        "figure.figsize": (6.0, 4.0),
        "savefig.dpi": 300,
    }
)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def ensure_fig_dir() -> None:
    FIGDIR.mkdir(exist_ok=True)


def savefig(fig: matplotlib.figure.Figure, slug: str) -> List[Path]:
    """Save a matplotlib figure to PDF and PNG inside figs/."""
    ensure_fig_dir()
    pdf_path = FIGDIR / f"fig_{slug}.pdf"
    png_path = FIGDIR / f"fig_{slug}.png"
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [pdf_path, png_path]


def save_ok(path: Path) -> bool:
    """Return True when either a PDF or PNG exists for the figure."""
    return path.with_suffix(".pdf").exists() or path.with_suffix(".png").exists()


# ---------------------------------------------------------------------------
# Sanitised module loader
# ---------------------------------------------------------------------------

_MODULE_CACHE: Dict[str, types.ModuleType] = {}


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    left = test.left
    right = test.comparators[0] if test.comparators else None
    if not isinstance(left, ast.Name) or left.id != "__name__":
        return False
    if isinstance(right, ast.Constant) and right.value == "__main__":
        return True
    return False


def load_module_clean(name: str, filename: str) -> types.ModuleType:
    """Load a Python module after stripping top-level run()/main() calls."""
    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]

    path = ROOT / filename
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))

    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    new_body = []
    for node in tree.body:
        if isinstance(node, ast.If) and _is_main_guard(node):
            continue  # skip guarded main blocks
        if isinstance(node, (ast.For, ast.While, ast.Try, ast.With)):
            continue  # skip executable blocks at module scope
        if isinstance(node, ast.Assign):
            value = node.value
            if isinstance(value, ast.Call):
                func = value.func
                if isinstance(func, ast.Name) and (
                    func.id == "main"
                    or func.id.startswith("run_")
                    or func.id in function_names
                ):
                    new_body.append(
                        ast.Assign(targets=node.targets, value=ast.Constant(value=None))
                    )
                    continue
            new_body.append(node)
            continue
        if isinstance(node, ast.AnnAssign):
            value = node.value
            if isinstance(value, ast.Call):
                func = value.func
                if isinstance(func, ast.Name) and (
                    func.id == "main"
                    or func.id.startswith("run_")
                    or func.id in function_names
                ):
                    node.value = ast.Constant(value=None)
                    new_body.append(node)
                    continue
            new_body.append(node)
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and (
                func.id == "main"
                or func.id.startswith("run_")
                or func.id in function_names
            ):
                continue
        new_body.append(node)

    tree.body = new_body
    ast.fix_missing_locations(tree)
    code = compile(tree, str(path), "exec")

    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(code, module.__dict__)
    except Exception:
        sys.modules.pop(name, None)
        raise
    _MODULE_CACHE[name] = module
    return module


# ---------------------------------------------------------------------------
# Figure registry
# ---------------------------------------------------------------------------

REGISTRY: Dict[str, Callable[[], Sequence[Path]]] = {}


def register(slug: str) -> Callable[[Callable[..., Sequence[Path]]], Callable[..., Sequence[Path]]]:
    """Decorator to register figure builders."""

    def decorator(func: Callable[..., Sequence[Path]]) -> Callable[..., Sequence[Path]]:
        if slug in REGISTRY:
            raise ValueError(f"Duplicate slug registered: {slug}")
        REGISTRY[slug] = func
        return func

    return decorator


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _configure_axes(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _collect_projection_data(
    coupling_fn: Callable[[float], np.ndarray],
    runner: Callable[..., Dict[str, float]],
    kappas: np.ndarray,
) -> Dict[str, np.ndarray]:
    rows = [runner(float(k), coupling_fn) for k in kappas]
    return {
        "kappa": kappas,
        "F_joint": np.array([r["F_joint"] for r in rows]),
        "F_proj": np.array([r["F_proj"] for r in rows]),
        "S_joint": np.array([r["S_joint"] for r in rows]),
        "S_proj": np.array([r["S_proj"] for r in rows]),
    }


# ---------------------------------------------------------------------------
# Registered builders
# ---------------------------------------------------------------------------


@register("projection-arrow")
def build_projection_arrow() -> Sequence[Path]:
    mod = load_module_clean("quantum_projection_arrow", "phase0_tests/quantum_projection_arrow.py")
    kappas = np.linspace(0.0, 1.0, 11)
    couplings = [
        ("Dephasing coupling", mod.U_dephasing),
        ("Partial-SWAP coupling", mod.U_partial_swap),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (label, fn) in zip(axes, couplings):
        data = _collect_projection_data(fn, mod.run_single, kappas)
        ax.plot(data["kappa"], data["F_joint"], marker="o", label="Fidelity (joint reverse)")
        ax.plot(data["kappa"], data["F_proj"], marker="s", label="Fidelity (projected reverse)")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlim(kappas.min(), kappas.max())
        _configure_axes(ax, r"$\kappa$", "Fidelity to $|\\psi_S^0\\rangle$", label)
        ax.legend(loc="lower left")

    fig.suptitle("Projection-induced irreversibility")
    return savefig(fig, "projection-arrow")


def _information_fidelity_panels(ax, points, title, plot_kind: str, fit_fn) -> None:
    I_all = np.concatenate([p["I_all"] for p in points])
    F_all = np.concatenate([p["F_all"] for p in points])

    if plot_kind == "power":
        x = np.clip(I_all, 1e-4, None)
        y = np.clip(1.0 - F_all, 1e-4, None)
        ax.loglog(x, y, ".", alpha=0.4)
        fit = fit_fn(I_all, F_all)
        if fit:
            grid = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
            fit_y = fit["c"] * (grid ** fit["alpha"])
            ax.loglog(grid, fit_y, "r-", lw=2, label=f"slope = {fit['alpha']:.3f}")
            ax.legend()
        _configure_axes(ax, r"$I(S{:}E)$ [bits]", r"$1 - F_{\mathrm{proj}}$", title)
    else:
        x = I_all
        y = -np.log(np.clip(F_all, 1e-12, 1.0))
        ax.scatter(x, y, s=18, alpha=0.4)
        fit = fit_fn(I_all, F_all)
        if fit:
            grid = np.linspace(0.0, x.max(), 200)
            fit_y = fit["a"] * grid + fit["b"]
            ax.plot(grid, fit_y, "r-", lw=2, label=f"slope = {fit['a']:.3f}")
            ax.legend()
        _configure_axes(ax, r"$I(S{:}E)$ [bits]", r"$-\ln F_{\mathrm{proj}}$", title)


@register("information-fidelity")
def build_information_fidelity() -> Sequence[Path]:
    mod = load_module_clean(
        "information_fidelity_experiment", "phase0_tests/information_fidelity_experiment.py"
    )
    kappas = np.linspace(0.05, 1.0, 12)
    n_trials = 64

    dep = mod.one_collision_IF_points(mod.U_dephasing, kappas, n_trials=n_trials, seed=1234)
    psw = mod.one_collision_IF_points(mod.U_partial_swap, kappas, n_trials=n_trials, seed=5678)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    _information_fidelity_panels(
        axes[0, 0],
        dep,
        "Dephasing — power law",
        "power",
        mod.fit_power_law,
    )
    _information_fidelity_panels(
        axes[1, 0],
        dep,
        "Dephasing — exponential",
        "exp",
        mod.fit_exp,
    )
    _information_fidelity_panels(
        axes[0, 1],
        psw,
        "Partial-SWAP — power law",
        "power",
        mod.fit_power_law,
    )
    _information_fidelity_panels(
        axes[1, 1],
        psw,
        "Partial-SWAP — exponential",
        "exp",
        mod.fit_exp,
    )

    fig.suptitle("Information fidelity relationships across couplings")
    return savefig(fig, "information-fidelity")


@register("fourqubit-summary")
def build_fourqubit_summary() -> Sequence[Path]:
    mod = load_module_clean("four_qubit_experiment", "phase0_tests/four_qubit_experiment.py")
    kappas = {"Dephasing": 0.60, "Partial-SWAP": 0.60}
    trials = 400
    seed_base = 42

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))

    for idx, (label, kappa) in enumerate(kappas.items()):
        H_fn = getattr(mod, f"H_{'dephasing' if 'Dephasing' in label else 'partial_swap'}_4q")
        H = H_fn()
        I, F, A, deff = mod.collect_samples(H, kappa, n_trials=trials, seed=seed_base + idx)

        gamma, R2_geo = mod.fit_through_origin(I, A ** 2)
        C, alpha, R2_dim = mod.fit_power_law(np.clip(deff - 1.0, 1e-9, None), np.clip(1.0 - F, 1e-12, None))

        ax_geo = axes[0, idx]
        ax_dim = axes[1, idx]

        ax_geo.scatter(I, A ** 2, s=12, alpha=0.35)
        grid = np.linspace(0.0, max(I.max(), 1e-6), 200)
        ax_geo.plot(grid, gamma * grid, "r-", lw=2, label=f"slope = {gamma:.2f}; $R^2$={R2_geo:.3f}")
        _configure_axes(ax_geo, r"$I(S{:}E)$ [bits]", r"$A^2$", f"{label}: geometric law")
        ax_geo.legend()

        x = np.clip(deff - 1.0, 1e-9, None)
        y = np.clip(1.0 - F, 1e-12, None)
        ax_dim.loglog(x, y, ".", alpha=0.35)
        grid = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
        ax_dim.loglog(
            grid,
            C * grid ** alpha,
            "r-",
            lw=2,
            label=f"slope = {alpha:.3f}; $R^2$={R2_dim:.3f}",
        )
        _configure_axes(ax_dim, r"$d_{\mathrm{eff}} - 1$", r"$1 - F$", f"{label}: dimensional law")
        ax_dim.legend()

    fig.suptitle("Four-qubit information–geometry–dimension checks")
    return savefig(fig, "fourqubit-summary")


@register("unification-summary")
def build_unification_summary() -> Sequence[Path]:
    mod = load_module_clean("Quantum_unification", "Quantum_unification.py")
    kappas = [0.2, 0.6]
    seeds = {0.2: 101, 0.6: 305}
    trials = 400

    fig, axes = plt.subplots(len(kappas), 3, figsize=(12.0, 6.4), sharex=True, sharey=True)
    if len(kappas) == 1:
        axes = np.array([axes])

    kinds = ["dephasing", "pswap", "random"]

    for row, kappa in enumerate(kappas):
        for col, kind in enumerate(kinds):
            I, A2, F, de = mod.collect(kind, 1 if kappa == 0.2 else 2, 1 if kappa == 0.2 else 2, kappa, n_trials=trials, seed=seeds[kappa] + col)
            X = np.clip(de - 1.0, 1e-8, None)
            Y = np.clip(A2 / np.clip(I, 1e-8, None), 1e-8, None)

            C, alpha, R2 = mod.linfit_loglog(X, Y)
            ax = axes[row, col]
            ax.loglog(X, Y, ".", alpha=0.35)
            grid = np.logspace(np.log10(X.min()), np.log10(X.max()), 200)
            ax.loglog(grid, C * grid ** alpha, "r-", lw=2, label=f"slope={alpha:.3f}, $R^2$={R2:.3f}")
            if row == len(kappas) - 1:
                ax.set_xlabel(r"$d_{\mathrm{eff}} - 1$")
            if col == 0:
                ax.set_ylabel(r"$A^2/I$")
            ax.set_title(f"{kind} — $\\kappa={kappa:.1f}$")
            ax.legend()

    fig.suptitle("Unified curvature–information–dimension scaling")
    return savefig(fig, "unification-summary")


@register("collapse-k0p6")
def build_collapse_k06() -> Sequence[Path]:
    mod = load_module_clean("Collapse_test", "phase0_tests/Collapse_test.py")
    kappa = 0.60
    models = ["dephasing", "pswap", "random2body"]
    trials = 250

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), sharey=True)

    for ax, model in zip(axes, models):
        X, Y = mod.sample_collapse_data(model, kappa, sizes=[(1, 1), (2, 2), (2, 3)], n_trials=trials, seed=1234)
        alpha, R2 = mod.fit_slope_loglog(X, Y)
        ax.loglog(X, Y, ".", alpha=0.35)
        grid = np.logspace(np.log10(X.min()), np.log10(X.max()), 200)
        C = np.exp(np.mean(np.log(Y)) - alpha * np.mean(np.log(X)))
        ax.loglog(grid, C * grid ** alpha, "r-", lw=2, label=f"slope={alpha:.3f}, $R^2$={R2:.3f}")
        _configure_axes(ax, r"$d_{\mathrm{eff}} - 1$", r"$Y$", model.capitalize())
        ax.legend()

    fig.suptitle("Collapse invariant across coupling models ($\\kappa=0.6$)")
    return savefig(fig, "collapse-k0p6")


@register("beta-vs-dimension")
def build_beta_vs_dimension() -> Sequence[Path]:
    mod = load_module_clean("beta_vs_dimension", "phase0_tests/beta_vs_dimension.py")
    models = ("dephasing", "pswap", "random2body")
    NS_LIST = [1, 2]
    NE_LIST = [1, 2, 3]
    results = {}

    for model in models:
        for nS in NS_LIST:
            for nE in NE_LIST:
                seed = 700 + 50 * (nS * 10 + nE)
                out = mod.collect_and_fit(
                    model,
                    nS,
                    nE,
                    kappa=0.60,
                    shots=400,
                    scrambleU=True,
                    twirlE=False,
                    min_I=1e-3,
                    use_binning=True,
                    bins=16,
                    min_per_bin=8,
                    seed=seed,
                )
                results[(model, nS, nE)] = out

    fig, axes = plt.subplots(1, len(models), figsize=(13.5, 4.3), sharey=True)
    for ax, model in zip(axes, models):
        xs, ys, err = [], [], []
        for nS in NS_LIST:
            for nE in NE_LIST:
                key = (model, nS, nE)
                res = results[key]
                xs.append(2 ** nE)
                ys.append(res["beta"])
                err.append(res["ci"])
        xs = np.array(xs)
        ys = np.array(ys)
        err = np.array(err)
        order = np.argsort(xs)
        xs, ys, err = xs[order], ys[order], err[order]

        ax.errorbar(xs, ys, yerr=err, fmt="o-", capsize=3)
        ax.axhline(0.5, linestyle="--", color="tab:red", alpha=0.6, label=r"target $\beta=0.5$")
        _configure_axes(ax, r"$d_E$", r"$\beta$", f"{model} couplings")
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.legend()

    fig.suptitle(r"Scaling exponent $\beta$ versus environment dimension ($\kappa=0.6$)")
    return savefig(fig, "beta-vs-dimension")


# ---------------------------------------------------------------------------
# TeX discovery & CLI plumbing
# ---------------------------------------------------------------------------


def parse_tex_figures(tex_path: Path) -> List[Tuple[str, Optional[str], Optional[str]]]:
    if not tex_path.exists():
        return []
    text = tex_path.read_text(encoding="utf-8")
    blocks = []
    tree = ast.parse("pass")  # placeholder to satisfy type checkers
    del tree

    entries: List[Tuple[str, Optional[str], Optional[str]]] = []
    import re

    figure_re = re.compile(r"\\begin\{figure\}.*?\\end\{figure\}", re.S)
    include_re = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}")
    caption_re = re.compile(r"\\caption\{([^}]*)\}")
    label_re = re.compile(r"\\label\{([^}]*)\}")

    for block in figure_re.findall(text):
        caption = None
        label = None
        cap_match = caption_re.search(block)
        lab_match = label_re.search(block)
        if cap_match:
            caption = cap_match.group(1)
        if lab_match:
            label = lab_match.group(1)
        for path in include_re.findall(block):
            entries.append((path, caption, label))
    return entries


def slug_from_tex_path(path: str) -> str:
    name = Path(path).name
    if name.startswith("fig_"):
        name = name[len("fig_") :]
    if "." in name:
        name = name.split(".", 1)[0]
    return name.lower().replace("_", "-")


def discover_slugs_from_tex(tex_path: Path) -> List[str]:
    entries = parse_tex_figures(tex_path)
    slugs = []
    for path, *_ in entries:
        slugs.append(slug_from_tex_path(path))
    return slugs


def try_heuristic(slug: str) -> Optional[str]:
    slug = slug.lower()
    heuristic_map = [
        ("projection", "projection-arrow"),
        ("arrow", "projection-arrow"),
        ("if", "information-fidelity"),
        ("information", "information-fidelity"),
        ("four", "fourqubit-summary"),
        ("fourqubit", "fourqubit-summary"),
        ("collapse", "collapse-k0p6"),
        ("beta", "beta-vs-dimension"),
        ("dimension", "beta-vs-dimension"),
        ("unification", "unification-summary"),
    ]
    for key, target in heuristic_map:
        if key in slug and target in REGISTRY:
            return target
    return None


def run_one(slug: str) -> Tuple[str, List[str]]:
    if slug in REGISTRY:
        try:
            paths = REGISTRY[slug]()
            results = []
            for path in paths:
                stem = path if isinstance(path, Path) else Path(path)
                if save_ok(stem):
                    results.append(stem)
            if results:
                return "generated", [f"Saved {', '.join(str(p) for p in results)}"]
            return "failed", [f"Builder for '{slug}' completed but outputs missing."]
        except Exception as exc:  # pragma: no cover - defensive
            traceback.print_exc()
            return "failed", [f"Exception while building '{slug}': {exc}"]

    mapped = try_heuristic(slug)
    if mapped and mapped != slug:
        return run_one(mapped)
    return "failed", [f"No builder registered for slug '{slug}'."]  # pragma: no cover


def resolve_slugs(args_only: Optional[str], run_all: bool) -> List[str]:
    if args_only:
        return [s.strip() for s in args_only.split(",") if s.strip()]
    if run_all:
        tex_slugs = discover_slugs_from_tex(ROOT / "docs" / "main.tex")
        if tex_slugs:
            return tex_slugs
        return sorted(REGISTRY.keys())
    raise ValueError("Specify --only or --all")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build figures for the Quantum Formula paper.")
    parser.add_argument("--only", type=str, help="Comma-separated list of slugs to build.")
    parser.add_argument("--all", action="store_true", help="Build all figures referenced in TeX (fallback: all builders).")
    args = parser.parse_args(argv)

    try:
        slugs = resolve_slugs(args.only, args.all)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    report: Dict[str, Tuple[str, List[str]]] = {}
    for slug in slugs:
        status, notes = run_one(slug)
        report[slug] = (status, notes)

    for slug, (status, notes) in report.items():
        symbol = {"generated": "OK", "failed": "ERR", "skipped": "SKIP"}.get(status, status)
        print(f"[{symbol}] {slug}")
        for note in notes:
            print(f"    - {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
