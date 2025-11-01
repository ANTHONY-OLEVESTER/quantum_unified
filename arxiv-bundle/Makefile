PDF=paper/main.pdf

all: figs data $(PDF)


$(PDF): paper/main.tex paper/sections.tex paper/appendix_theorem.tex paper/macros.tex paper/references.bib paper/phase4_summary.tex
	cd paper && pdflatex -interaction=nonstopmode main.tex
	-cd paper && bibtex main
	cd paper && pdflatex -interaction=nonstopmode main.tex
	cd paper && pdflatex -interaction=nonstopmode main.tex
	-@copy /Y paper\main.pdf Curvature-Information-Principle.pdf 2> NUL || true
	-@cp -f paper/main.pdf Curvature-Information-Principle.pdf 2>/dev/null || true

figs:
	@echo "Using existing figures in ./figures (Phase 2/3)."

data:
	@echo "Using existing CSVs in ./data (Phase 2/3)."



clean:
	rm -f paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out paper/*.toc
# BEGIN PHASE4 RULE
paper/phase4_summary.tex: data/phase4_alpha_vs_invD.csv data/phase4_varY_by_D.csv scripts/phase4_to_tex.py
	@echo "Generating Phase IV LaTeX summary..."
	python scripts/phase4_to_tex.py
# END PHASE4 RULE

phase6-numbers:
	@python scripts/phase6_pull_numbers.py
