PDF=paper/main.pdf
ARXIV=arxiv_bundle

all: assets data-tex $(PDF)

data-tex:
	@python scripts/ingest_metrics.py
	@python scripts/make_tables.py || true

$(PDF): paper/main.tex paper/sections.tex paper/sections_auto.tex paper/appendix_theorem.tex paper/macros.tex paper/references.bib
	cd paper && pdflatex -interaction=nonstopmode main.tex
	-cd paper && bibtex main
	cd paper && pdflatex -interaction=nonstopmode main.tex
	cd paper && pdflatex -interaction=nonstopmode main.tex
	-@copy /Y paper\main.pdf Curvature-Information-Principle.pdf 2> NUL || true
	-@cp -f paper/main.pdf Curvature-Information-Principle.pdf 2>/dev/null || true

assets:
	@echo "Syncing expected figure assets into paper/figures ..."
	-@mkdir paper\figures 2> NUL || true
	-@copy /Y phase8-out\figures\phase8_var_scaling.png paper\figures\phase8_var_scaling.png 2> NUL || true
	-@copy /Y phase9-plus-haar-extend\figures\phase9_varY_vs_D_haar.png paper\figures\phase9_varY_scaling.png 2> NUL || true
	-@copy /Y phase3-out\phase3_alpha_vs_invD.png paper\figures\phase3_alpha_vs_invD.png 2> NUL || true
	-@copy /Y figures\phase2_collapse_panels.png paper\figures\phase2_collapse_panels.png 2> NUL || true

arxiv:
	-@rmdir /S /Q $(ARXIV) 2> NUL || true
	-@rm -rf $(ARXIV) 2>/dev/null || true
	-@mkdir $(ARXIV) 2> NUL || true
	-@xcopy /Y /I /Q paper\*.tex $(ARXIV)\ > NUL 2>&1 || true
	-@xcopy /Y /I /Q paper\*.bib $(ARXIV)\ > NUL 2>&1 || true
	-@copy /Y paper\README_ARXIV.txt $(ARXIV)\README.txt > NUL 2>&1 || true
	-@mkdir $(ARXIV)\figures 2> NUL || true
	-@if exist paper\figures ( xcopy /E /I /Y paper\figures $(ARXIV)\figures\ > NUL 2>&1 ) else ( xcopy /E /I /Y figures $(ARXIV)\figures\ > NUL 2>&1 )
	-@tar -czf curvature_information_arxiv.tgz -C $(ARXIV) .

clean:
	rm -f paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out paper/*.toc paper/*.lof paper/*.lot

.PHONY: all clean arxiv data-tex assets
