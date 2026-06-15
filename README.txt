================================================================================
ECONOMETRICA SUBMISSION PACKAGE
"Identifying Rational Types in Unknown Environments"
Author: Humberto Bernal
================================================================================

This folder contains the main manuscript and three self-contained online
appendices for submission to Econometrica.

--------------------------------------------------------------------------------
STRUCTURE
--------------------------------------------------------------------------------

Main/
  Bernal_H.tex              Main manuscript (compile from this directory)
  Bernal_H.pdf              Compiled manuscript (regenerate if needed)
  references.bib            Bibliography
  econsocart.cls/.cfg, econsoc.bst, textcase.sty
  figuras_esp_ordenadas/    19 empirical figures (Fig_1a--Fig_10b)
  figuras_calibracion_journal/  16 calibration figures (ELN / PAR scenarios)

Appendix_1/
  tex/Appendix_1.tex        Macro dynamics: VEC and LLT (Section 2.2)
  matlab/                   Replication scripts and data
  README.txt                Full replication instructions

Appendix_2/
  tex/Appendix_2.tex        Microstructure and competing hazards (Section 2.3)
  tex/tables_mnl_full.tex, tables_cox_full.tex
  stata/                    Stata replication pipeline
  replicate.do              Entry point
  README.txt                Full replication instructions

Appendix_3/
  tex/Appendix_3.tex        Formal proofs and Lean verification
  lean/                     Lean 4 formalization (Appendix3Proofs.lean)
  replicate.sh              Lean build entry point
  README.txt                Full verification instructions

--------------------------------------------------------------------------------
COMPILE MAIN MANUSCRIPT
--------------------------------------------------------------------------------

  cd Main
  pdflatex Bernal_H
  bibtex   Bernal_H
  pdflatex Bernal_H
  pdflatex Bernal_H

--------------------------------------------------------------------------------
COMPILE APPENDICES (optional)
--------------------------------------------------------------------------------

  cd Appendix_1/tex && pdflatex Appendix_1 && bibtex Appendix_1 && pdflatex Appendix_1 && pdflatex Appendix_1
  cd Appendix_2/tex && pdflatex Appendix_2 && bibtex Appendix_2 && pdflatex Appendix_2 && pdflatex Appendix_2
  cd Appendix_3/tex && pdflatex Appendix_3 && bibtex Appendix_3 && pdflatex Appendix_3 && pdflatex Appendix_3

================================================================================
