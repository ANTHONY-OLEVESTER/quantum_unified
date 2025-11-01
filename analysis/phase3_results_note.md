# Phase III Universality Summary (placeholder)

This note captures the asymptotic scaling evidence once the Phase III sweep is executed.
Populate the tables below after running:

```
python phase2.py --sweep --max-qubits 9 --output-dir phase3-out
```

## Finite-size exponent

| model        | num sizes | gamma | 95% CI | pass? | notes |
|--------------|-----------|-------|--------|-------|-------|
| random2body  | TODO      | TODO  | TODO   | TODO  |       |
| pswap        | TODO      | TODO  | TODO   | TODO  |       |
| dephasing    | TODO      | TODO  | TODO   | TODO  |       |

## Variance law

| model        | slope | 95% CI | num sizes | comment |
|--------------|-------|--------|-----------|---------|
| random2body  | TODO  | TODO   | TODO      | target slope -2 |
| depolarizing | TODO  | TODO   | TODO      | target slope -2 |

## Checklist

- [ ] `finite_size_alpha_per_size.csv` includes $D=256$ and $D=512$ entries.
- [ ] `phase3_varY_by_D.csv` covers at least four distinct $D$ values per model.
- [ ] Figures `phase3_alpha_vs_invD.(png|pdf)` and `phase3_varY_scaling.(png|pdf)` updated.
- [ ] Appendix theorem referenced in `docs/main.tex` reflects the latest constants.

Record qualitative observations about failure modes (integrable, dissipative, topological) here once the structured sweeps are expanded.
