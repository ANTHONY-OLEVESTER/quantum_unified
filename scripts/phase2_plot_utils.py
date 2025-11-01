# phase2_plot_utils.py
#
# Shared helpers to convert Phase-II universality sweep data into reusable
# plotting caches and publish-ready figures.

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    return value


def build_plot_cache(
    pooled_data: Mapping[str, Dict[str, Any]],
    size_data: Mapping[str, Dict[str, Dict[str, Any]]],
    rows: Sequence[Any],
    pooled_rows: Sequence[Mapping[str, Any]],
    alpha_per_size: Sequence[Mapping[str, Any]] | None = None,
    gamma_rows: Sequence[Mapping[str, Any]] | None = None,
    var_y_samples: Sequence[Mapping[str, Any]] | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    # Backwards compatibility: older callers used slightly different kwarg names.
    if 'alpha_per_size_rows' in kwargs and alpha_per_size is None:
        alpha_per_size = kwargs.pop('alpha_per_size_rows')
    if 'var_y_rows' in kwargs and var_y_samples is None:
        var_y_samples = kwargs.pop('var_y_rows')
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"build_plot_cache() got unexpected keyword arguments: {unexpected}")
    cache: Dict[str, Any] = {
        "pooled_data": {},
        "size_data": {},
        "rows": [],
        "pooled_rows": [],
        "alpha_per_size": [],
        "gamma_rows": [],
        "var_y_by_D": [],
    }

    for model, data in pooled_data.items():
        cache["pooled_data"][model] = {
            "kind": data.get("kind"),
            "X": [np.asarray(arr, dtype=float).tolist() for arr in data.get("X", [])],
            "Y": [np.asarray(arr, dtype=float).tolist() for arr in data.get("Y", [])],
        }

    for model, sizes in size_data.items():
        cache["size_data"][model] = {}
        for size_label, payload in sizes.items():
            cache["size_data"][model][size_label] = {
                "D": float(payload.get("D", float("nan"))),
                "X": [np.asarray(arr, dtype=float).tolist() for arr in payload.get("X", [])],
                "Y": [np.asarray(arr, dtype=float).tolist() for arr in payload.get("Y", [])],
            }

    cache["rows"] = [_sanitize_dict(asdict(row)) for row in rows]
    cache["pooled_rows"] = [_sanitize_dict(row) for row in pooled_rows]
    if alpha_per_size is not None:
        cache["alpha_per_size"] = [_sanitize_dict(row) for row in alpha_per_size]
    if gamma_rows is not None:
        cache["gamma_rows"] = [_sanitize_dict(row) for row in gamma_rows]
    if var_y_samples is not None:
        cache["var_y_by_D"] = [_sanitize_dict(row) for row in var_y_samples]
    return cache


def _sanitize_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: _to_serializable(val) for key, val in data.items()}


def _concat(list_of_lists: Iterable[Iterable[float]]) -> np.ndarray:
    arrays = []
    for lst in list_of_lists:
        arr = np.asarray(lst, dtype=float)
        if arr.size:
            arrays.append(arr)
    if not arrays:
        return np.array([], dtype=float)
    return np.concatenate(arrays)


def _fit_slope_loglog(X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
    mask = (X > 0) & (Y > 0)
    X = X[mask]
    Y = Y[mask]
    if X.size < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "R2": float("nan")}
    lx = np.log10(X)
    ly = np.log10(Y)
    A = np.column_stack((lx, np.ones_like(lx)))
    slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
    yhat = slope * lx + intercept
    ss_res = np.sum((ly - yhat) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2) + 1e-12
    R2 = 1.0 - ss_res / ss_tot
    return {"slope": float(slope), "intercept": float(intercept), "R2": float(R2)}


def plot_phase2_results(cache: Dict[str, Any], output_dir: Path | str) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    models_order = [row["model"] for row in cache.get("pooled_rows", [])]
    pooled_stats = {row["model"]: row for row in cache.get("pooled_rows", [])}

    if cache.get("pooled_data"):
        saved.extend(
            _plot_collapse_panels(cache.get("pooled_data", {}), pooled_stats, models_order, output_dir)
        )
    saved.extend(
        _plot_finite_size_scaling(cache.get("size_data", {}), output_dir)
    )
    alpha_rows = cache.get("alpha_per_size", [])
    gamma_stats = {row["model"]: row for row in cache.get("gamma_rows", cache.get("pooled_rows", []))}
    saved.extend(
        plot_alpha_vs_invD(alpha_rows, output_dir)
    )
    saved.extend(
        plot_gamma_from_alpha(alpha_rows, gamma_stats, output_dir)
    )
    saved.extend(
        plot_variance_scaling(cache.get("var_y_by_D", []), output_dir)
    )
    saved.extend(
        _plot_twirl_restoration(cache.get("rows", []), output_dir)
    )
    return saved


def _save_figure(fig: plt.Figure, stem: str, output_dir: Path) -> List[Path]:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def _plot_collapse_panels(
    pooled_data: Mapping[str, Dict[str, Any]],
    pooled_stats: Mapping[str, Mapping[str, Any]],
    models_order: Sequence[str],
    output_dir: Path,
) -> List[Path]:
    if not models_order:
        return []
    n_models = len(models_order)
    cols = min(4, n_models)
    rows_fig = int(math.ceil(n_models / cols))
    fig, axes = plt.subplots(rows_fig, cols, figsize=(4.2 * cols, 3.4 * rows_fig), squeeze=False)
    axes_iter = iter(axes.flat)

    for model in models_order:
        ax = next(axes_iter)
        pdata = pooled_data.get(model)
        stats = pooled_stats.get(model, {})
        if not pdata:
            ax.set_visible(False)
            continue
        X = _concat(pdata.get("X", []))
        Y = _concat(pdata.get("Y", []))
        if X.size < 2 or Y.size < 2:
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        slope = float(stats.get("alpha", float("nan")))
        intercept = float(stats.get("intercept", float("nan")))

        ax.scatter(X, Y, s=12, alpha=0.35, color="#1f77b4")
        if np.isfinite(slope) and np.isfinite(intercept):
            xs = np.logspace(np.log10(X.min()), np.log10(X.max()), 200)
            logy = slope * np.log10(xs) + intercept
            ax.plot(xs, 10 ** logy, color="#d62728", lw=1.6, label=f"$\\alpha={slope:+.3f}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.set_xlabel(r"$d_{\mathrm{eff}}-1$")
        ax.set_ylabel(r"$Y$")
        kind = pdata.get("kind", "")
        title = f"{model} ({kind})" if kind else model
        ax.set_title(title)
        if ax.get_legend() is None and np.isfinite(slope):
            ax.legend(frameon=False, loc="best", fontsize=9)

    for ax in axes_iter:
        ax.set_visible(False)

    return _save_figure(fig, "phase2_collapse_panels", output_dir)


def _plot_finite_size_scaling(size_data: Mapping[str, Dict[str, Dict[str, Any]]], output_dir: Path) -> List[Path]:
    if not size_data:
        return []
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    all_D: List[float] = []
    all_abs: List[float] = []
    for idx, (model, sizes) in enumerate(size_data.items()):
        Ds = []
        abs_alphas = []
        for payload in sizes.values():
            X = _concat(payload.get("X", []))
            Y = _concat(payload.get("Y", []))
            if X.size < 2 or Y.size < 2:
                continue
            fit = _fit_slope_loglog(X, Y)
            if not np.isfinite(fit["slope"]):
                continue
            Ds.append(float(payload.get("D", float("nan"))))
            abs_alphas.append(abs(fit["slope"]))
        if len(Ds) < 2:
            continue
        Ds = np.array(Ds, dtype=float)
        abs_alphas = np.array(abs_alphas, dtype=float)
        order = np.argsort(Ds)
        Ds = Ds[order]
        abs_alphas = abs_alphas[order]
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None
        ax.plot(Ds, abs_alphas, "o-", label=model, color=color)
        fit = _fit_slope_loglog(Ds, abs_alphas)
        if np.isfinite(fit["slope"]) and np.isfinite(fit["intercept"]):
            xs = np.logspace(np.log10(Ds.min()), np.log10(Ds.max()), 200)
            logy = fit["slope"] * np.log10(xs) + fit["intercept"]
            ax.plot(xs, 10 ** logy, "--", color=color, alpha=0.6)
        all_D.extend(Ds.tolist())
        all_abs.extend(abs_alphas.tolist())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$D = d_S d_E$")
    ax.set_ylabel(r"$|\alpha|$")
    ax.set_title("Finite-size scaling")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    if all_D and all_abs:
        xs = np.logspace(np.log10(min(all_D)), np.log10(max(all_D)), 200)
        if xs.size:
            ref_scale = max(all_abs)
            ref = ref_scale * (xs / xs[0]) ** -1
            ax.plot(xs, ref, "k--", alpha=0.5, label=r"$D^{-1}$ reference")
    if ax.get_lines():
        ax.legend(frameon=False, fontsize=9)
    return _save_figure(fig, "phase2_finite_size_scaling", output_dir)


def plot_alpha_vs_invD(
    alpha_rows: Sequence[Mapping[str, Any]],
    output_dir: Path | str,
) -> List[Path]:
    output_dir = Path(output_dir)
    if not alpha_rows:
        return []
    grouped: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row in alpha_rows:
        try:
            model = row.get("model", "")
            D = float(row.get("D", float("nan")))
            abs_alpha = float(row.get("abs_alpha", float("nan")))
        except (TypeError, ValueError):
            continue
        if not (model and np.isfinite(D) and np.isfinite(abs_alpha) and D > 0):
            continue
        grouped[model].append((D, abs_alpha))
    if not grouped:
        return []

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for idx, (model, data) in enumerate(sorted(grouped.items())):
        data.sort(key=lambda pair: pair[0])
        Ds = np.array([pair[0] for pair in data], dtype=float)
        abs_alphas = np.array([pair[1] for pair in data], dtype=float)
        invD = 1.0 / Ds
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None
        ax.scatter(invD, abs_alphas, color=color, s=40, alpha=0.85)
        if invD.size >= 2:
            coeffs = np.polyfit(invD, abs_alphas, 1)
            xs = np.linspace(invD.min(), invD.max(), 200)
            ax.plot(xs, np.polyval(coeffs, xs), linestyle="--", color=color)
            intercept = coeffs[1]
            label = f"{model}: intercept={intercept:+.3f}"
        else:
            label = f"{model}"
        ax.plot([], [], color=color, linestyle="--", label=label)
    ax.set_xlabel(r"$1/D$")
    ax.set_ylabel(r"$|\alpha|$")
    ax.set_title(r"$|\alpha|$ versus $1/D$")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(frameon=False, fontsize=9)
    return _save_figure(fig, "phase3_alpha_vs_invD", output_dir)


def plot_variance_scaling(
    var_rows: Sequence[Mapping[str, Any]],
    output_dir: Path | str,
) -> List[Path]:
    output_dir = Path(output_dir)
    if not var_rows:
        return []
    grouped: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in var_rows:
        try:
            model = row.get("model", "")
            D = float(row.get("D", float("nan")))
            y_val = float(row.get("y_value", float("nan")))
        except (TypeError, ValueError):
            continue
        if not (model and np.isfinite(D) and np.isfinite(y_val) and D > 0):
            continue
        grouped[model][D].append(y_val)
    if not grouped:
        return []

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for idx, (model, d_map) in enumerate(sorted(grouped.items())):
        Ds = []
        variances = []
        for D, values in sorted(d_map.items()):
            arr = np.asarray(values, dtype=float)
            if arr.size < 2:
                continue
            Ds.append(D)
            variances.append(float(np.var(arr, ddof=1)))
        if len(Ds) < 2:
            continue
        Ds = np.array(Ds, dtype=float)
        variances = np.maximum(np.array(variances, dtype=float), 1e-12)
        order = np.argsort(Ds)
        Ds = Ds[order]
        variances = variances[order]
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None
        ax.scatter(Ds, variances, color=color, s=45, alpha=0.85)
        fit = _fit_slope_loglog(Ds, variances)
        slope = fit["slope"]
        if np.isfinite(slope) and np.isfinite(fit["intercept"]):
            xs = np.logspace(np.log10(Ds.min()), np.log10(Ds.max()), 200)
            logy = slope * np.log10(xs) + fit["intercept"]
            ax.plot(xs, 10 ** logy, linestyle="--", color=color)
            label = f"{model}: slope={slope:+.3f}"
        else:
            label = model
        ax.plot([], [], color=color, linestyle="--", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$D$")
    ax.set_ylabel(r"$\mathrm{Var}(Y)$")
    ax.set_title(r"Variance scaling of $Y$")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(frameon=False, fontsize=9)
    return _save_figure(fig, "phase3_varY_scaling", output_dir)


def plot_gamma_from_alpha(
    alpha_rows: Sequence[Mapping[str, Any]],
    gamma_stats: Mapping[str, Mapping[str, Any]],
    output_dir: Path | str,
) -> List[Path]:
    output_dir = Path(output_dir)
    if not alpha_rows:
        return []

    grouped: Dict[str, List[Tuple[float, float]]] = {}
    for row in alpha_rows:
        model = row.get("model")
        if model is None:
            continue
        try:
            D = float(row.get("D", float("nan")))
            abs_alpha = float(row.get("abs_alpha", float("nan")))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(D) and math.isfinite(abs_alpha) and D > 0 and abs_alpha > 0):
            continue
        grouped.setdefault(str(model), []).append((D, abs_alpha))

    if not grouped:
        return []

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for idx, (model, data) in enumerate(sorted(grouped.items())):
        if not data:
            continue
        data.sort(key=lambda pair: pair[0])
        Ds = np.array([pair[0] for pair in data], dtype=float)
        abs_alphas = np.array([pair[1] for pair in data], dtype=float)
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None
        ax.scatter(Ds, abs_alphas, color=color, s=40, alpha=0.85)

        stats = gamma_stats.get(model, {})
        gamma = float(stats.get("gamma", float("nan")))
        gamma_lo = float(stats.get("gamma_lo", float("nan")))
        gamma_hi = float(stats.get("gamma_hi", float("nan")))
        log_c = float(stats.get("log_c", float("nan")))

        if np.isfinite(gamma) and np.isfinite(log_c):
            xs = np.logspace(np.log10(Ds.min()), np.log10(Ds.max()), 200)
            logy = log_c - gamma * np.log10(xs)
            ax.plot(xs, 10 ** logy, color=color, linestyle="--", linewidth=2.0)

        if np.isfinite(gamma) and np.isfinite(gamma_lo) and np.isfinite(gamma_hi):
            label = f"{model}: gamma_hat={gamma:.3f} [{gamma_lo:.3f},{gamma_hi:.3f}]"
        elif np.isfinite(gamma):
            label = f"{model}: gamma_hat={gamma:.3f}"
        else:
            label = f"{model}: gamma_hat=nan"
        ax.plot([], [], color=color, linestyle="--", linewidth=2.0, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$D = d_S d_E$")
    ax.set_ylabel(r"$|\alpha|$")
    ax.set_title(r"Finite-size exponent fits on $|\alpha|$")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(frameon=False, fontsize=9)
    return _save_figure(fig, "phase2_finite_size_gamma", output_dir)


def _plot_twirl_restoration(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> List[Path]:
    depth_map: Dict[float, Dict[int, float]] = {}
    for row in rows:
        model = row.get("model", "")
        if not isinstance(model, str) or not model.startswith("amp_damp_twirl"):
            continue
        try:
            depth = int(model.replace("amp_damp_twirl", ""))
        except ValueError:
            continue
        param = row.get("param", "")
        if isinstance(param, str) and "=" in param:
            try:
                p_val = float(param.split("=")[1])
            except ValueError:
                p_val = float("nan")
        else:
            p_val = float("nan")
        if not math.isfinite(p_val):
            continue
        depth_map.setdefault(p_val, {})[depth] = abs(float(row.get("alpha", float("nan"))))

    if not depth_map:
        return []

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for p_val in sorted(depth_map):
        depths = sorted(depth_map[p_val])
        alphas = [depth_map[p_val][d] for d in depths]
        ax.plot(depths, alphas, marker="o", label=f"p={p_val:.2f}")
    ax.set_xlabel("Twirl depth $m$")
    ax.set_ylabel(r"$|\alpha|$")
    ax.set_title("Twirl restoration (amplitude damping)")
    ax.set_xticks(sorted({d for depths in depth_map.values() for d in depths}))
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(frameon=False, fontsize=9)
    return _save_figure(fig, "phase2_twirl_restoration", output_dir)
