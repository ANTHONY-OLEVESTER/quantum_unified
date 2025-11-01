Title: A Universal Curvature–Information Principle: Flatness and D^-1 Concentration under 2-Designs
Author: Anthony Olevester (olevester.joram123@gmail.com)

Contents
- main.tex               (entry point)
- sections.tex           (paper body; includes sections_auto.tex)
- sections_auto.tex      (pre-generated summary; no code execution)
- appendix_theorem.tex   (theorem and proof sketch)
- macros.tex             (macros + theorem env)
- references.bib         (bibliography database)
- figures/*.png          (figures used by main.tex via \graphicspath)

Build Instructions (arXiv-compatible)
1) Run pdfLaTeX once to generate aux files:
   pdflatex -interaction=nonstopmode main.tex
2) Run BibTeX:
   bibtex main
3) Run pdfLaTeX twice more to resolve references:
   pdflatex -interaction=nonstopmode main.tex
   pdflatex -interaction=nonstopmode main.tex

Notes
- No shell-escape (write18) is required.
- Standard packages only: geometry, amsmath, amssymb, amsthm, graphicx, booktabs, hyperref, xcolor.
- Figures are PNGs in figures/; \graphicspath{{figures/}} is set in main.tex.
- The file sections_auto.tex is static text generated ahead of time; no external scripts are invoked by LaTeX.
- If any figure is missing in your builder, verify the figures/ directory is present alongside main.tex.

Tested locally with MiKTeX/pdfTeX and BibTeX; should work with TeX Live on arXiv.

