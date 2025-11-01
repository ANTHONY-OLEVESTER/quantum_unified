#!/usr/bin/env python3
import pathlib
import sys

import numpy as np
import pandas as pd


def fit_intercept_alpha(df: pd.DataFrame):
    x = None
    for candidate in ("invD", "1/D"):
        if candidate in df.columns:
            x = df[candidate].to_numpy(dtype=float)
            x_name = candidate
            break
    y = None
    for candidate in ("abs_alpha", "|alpha|", "abs(|alpha|)"):
        if candidate in df.columns:
            y = df[candidate].to_numpy(dtype=float)
            y_name = candidate
            break
    if x is None or y is None:
        raise RuntimeError("phase4_alpha_vs_invD.csv must have columns [model, invD(or 1/D), abs_alpha].")

    if "model" not in df.columns:
        raise RuntimeError("phase4_alpha_vs_invD.csv missing 'model' column.")

    results = []
    for model, group in df.groupby("model"):
        gx = group[x_name].to_numpy(dtype=float)
        gy = group[y_name].to_numpy(dtype=float)
        A = np.vstack([gx, np.ones_like(gx)]).T
        slope, intercept = np.linalg.lstsq(A, gy, rcond=None)[0]
        results.append((model, float(intercept), float(slope)))
    return sorted(results, key=lambda t: t[0])


def fit_loglog_slope(df: pd.DataFrame):
    if "model" not in df.columns:
        raise RuntimeError("phase4_varY_by_D.csv missing 'model' column.")
    if "D" not in df.columns:
        raise RuntimeError("phase4_varY_by_D.csv must include 'D' column.")

    var_col = None
    for candidate in ("varY", "VarY", "var_y", "mean_varY"):
        if candidate in df.columns:
            var_col = candidate
            break
    if var_col is None:
        raise RuntimeError("phase4_varY_by_D.csv must have columns [model, D, varY].")

    output = []
    for model, group in df.groupby("model"):
        logD = np.log10(group["D"].to_numpy(dtype=float))
        logV = np.log10(np.maximum(group[var_col].to_numpy(dtype=float), 1e-16))
        A = np.vstack([logD, np.ones_like(logD)]).T
        slope, intercept = np.linalg.lstsq(A, logV, rcond=None)[0]
        output.append((model, float(slope)))
    return sorted(output, key=lambda t: t[0])


def main() -> int:
    out_path = pathlib.Path("paper/phase4_summary.tex")
    alpha_csv = pathlib.Path("data/phase4_alpha_vs_invD.csv")
    var_csv = pathlib.Path("data/phase4_varY_by_D.csv")

    alpha_df = pd.read_csv(alpha_csv)
    var_df = pd.read_csv(var_csv)

    intercepts = fit_intercept_alpha(alpha_df)
    slopes = dict(fit_loglog_slope(var_df))

    lines = [
        r"\subsection*{Phase IV: Large-$D$ trending (summary)}",
        r"We extended sizes and checked two asymptotic signatures:",
        r"\begin{enumerate}",
        r"\item \textbf{$|\alpha|$ vs $1/D$}: near-flat trends with finite intercepts.",
        r"\item \textbf{Variance scaling}: $\mathrm{Var}(Y)$ decays with $D$ (log--log slope).",
        r"\end{enumerate}",
        "",
        r"\begin{center}",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"model & $|\alpha|(1/D\!\to\!0)$ (intercept) & slope of $\log\Var(Y)$ vs $\log D$ \\",
        r"\hline",
    ]
    for model, intercept, slope_alpha in intercepts:
        slope_var = slopes.get(model, float("nan"))
        lines.append(f"{model} & {intercept:.3f} & {slope_var:.3f} \\\\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{center}", "", r"Figures~\ref{fig:phase4-alpha-invD} and~\ref{fig:phase4-vary} show the raw trends."])

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
