# Curvature–Information Invariant (Phases 1–3)

## How to build the paper
```bash
make           # builds paper/main.pdf using existing data & figures
```

Repo layout:

* `src/` — code (`phase2.py`, utils)
* `data/` — CSV outputs
* `figures/` — PNG figures
* `paper/` — LaTeX sources
## Phase VI (fast isotropic asymptotics)
Artifacts:
- `figures/phase6_alpha_vs_invD.png`
- `figures/phase6_varY_scaling.png`
- `figures/phase6_Y_hist_Dmax.png`
- `data/phase6_theorem_perD.csv`, `data/phase6_summary.txt`

To preview extracted numbers:
```bash
make phase6-numbers
```

> Note: the Phase VI sampler approximates a 2-design; the 
> Haar/4-design concentration proven in Appendix~\ref{app:theorem} 
> will require a deeper random circuit or exact Haar isometries.


## Build the paper

`ash
python scripts/ingest_metrics.py
make
``n
## Create arXiv bundle

`ash
make arxiv
# produces curvature_information_arxiv.tgz
``n
