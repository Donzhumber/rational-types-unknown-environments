================================================================================
REPLICATION PACKAGE
Appendix 2: Microstructure and Competing Hazards in Colombia
Supplement to "Identifying Rational Types in Unknown Environments"
Author: Humberto Bernal (Economics Program, Universidad Colegio Mayor de
Cundinamarca; Ph.D. in Economics, Universidad de los Andes)
================================================================================

This folder contains everything needed to replicate the results reported in
Appendix 2 (Section 2.3 of the main paper): CNMH/SIEVCAC microdata, Stata
estimation scripts, exported Cox curves, and the English LaTeX source. All
scripts, comments, console output, and table notes are in English.

--------------------------------------------------------------------------------
1. CONTENTS
--------------------------------------------------------------------------------

replicate.do              Entry point at the package root (recommended).
README.txt                This file.

stata/
  _setup_paths.do         Locates stata/data/Data_merge.dta and sets paths
                          (included by every script).
  _load_analytical_sample.do
                          Loads the estimation checkpoint or builds it from
                          Data_merge.dta.
  run_all.do              Full pipeline called by replicate.do.
  microstructure_replication.do
                          Sample construction, attrition logit, main MNL,
                          robustness, SUEST IIA test, four cause-specific
                          Cox models, margin exports.
  sample_selection_robustness.do
                          Focused replication of MNL fit, robustness variants,
                          death-rate proportion test, and SUEST IIA test.
  cox_only.do             Standalone Cox replication (four cause-specific
                          equations) with export to cox_appendix_rows.txt.
  export_cox_xlsx.py      Refreshes Survival_Rescate in Datos_Graficas_Cox.xlsx
                          from cumul_rescue_d100.dta (called by
                          microstructure_replication.do).
  data/
    Data_merge.dta        Merged CNMH case-victim file (N = 39,369; ~37 MB).
                          Required for replication.
    analytical_sample.dta Estimation checkpoint (n = 1,125; numeric outcome
                          codes). Used by sample_selection_robustness.do
                          without rerunning the full construction block.
  outputs/
    Datos_Graficas_Cox.xlsx
                          Survival/hazard curves; Survival_Rescate and
                          Cumul_Rescue refreshed on each full run.
    (generated on run)    replication.log (full console output),
                          analytical_sample.dta, cox_appendix_rows.txt,
                          cox_cumul_day100.txt, cumul_rescue_d100.dta, *.log.

tex/
  Appendix_2.tex          LaTeX source of the appendix.
  Appendix_2.pdf          Compiled document (regenerate with Step 5 if missing).
  tables_mnl_full.tex     Full MNL tables (input by Appendix_2.tex).
  tables_cox_full.tex     Full Cox tables (input by Appendix_2.tex).
  references.bib          Bibliography database.
  econsocart.cls,
  econsocart.cfg,
  econsoc.bst,
  textcase.sty            Econometric Society class files needed to compile.

--------------------------------------------------------------------------------
2. REQUIREMENTS
--------------------------------------------------------------------------------

- Stata 19 or later (commands: logit, mlogit, stset, stcox, margins,
  suest, prtest, estimates table).
- A LaTeX distribution (TeX Live 2024 or later) with pdflatex and bibtex,
  to recompile the document (optional).

--------------------------------------------------------------------------------
3. HOW TO REPLICATE
--------------------------------------------------------------------------------

Step 1 -- Run the full pipeline (choose one method):

  Method A -- Stata GUI (set working directory first):
    File > Change Working Directory... > select Appendix_2/
    (must contain replicate.do and stata/ subfolder). Then:

      do replicate.do

  Method B -- Full path (works from any working directory):

      do "/full/path/to/Appendix_2/replicate.do"

  Method C -- Already inside stata/:

      do run_all.do

  Method D -- Batch mode (shell):

      cd Appendix_2
      stata -b -q do replicate.do

  replicate.do locates stata/run_all.do automatically. Estimated runtime:
  10--30 minutes. Full output is saved to stata/outputs/replication.log
  (open in any text editor; Results may truncate very long runs).

Step 2 -- Individual scripts (optional):
  Working directory: Appendix_2/stata/

      do microstructure_replication.do
      do sample_selection_robustness.do
      do cox_only.do

  Each script includes _setup_paths.do and finds data/Data_merge.dta
  automatically. sample_selection_robustness.do and cox_only.do can use
  data/analytical_sample.dta if microstructure_replication.do was already
  run (or rebuild from Data_merge.dta if the checkpoint is missing).

  Expected console output (microstructure_replication.do):
  - Universe N = 39,369; valid event date n = 36,916; extortionate n = 11,270
  - Analytical sample n = 1,125; municipal clusters K = 373
  - Attrition logit: n = 11,253 (of 11,270 extortive); non-metropolitan zones
    OR = 1.9--2.2 (p < 0.001); period 1998--2002 OR = 1.76 (p = 0.001)
  - MNL: Wald chi2(78) = 5,250.34, p < 0.001; Pseudo R2 = 0.18
  - Key RRRs: post-2011 Escape = 11.28; public-sector Escape = 3.92;
    Rescue duration RRR = 0.18 (short), 0.08 (long)
  - IIA: chi2(1) = 0.24, p = 0.625
  - Robustness: common delinquency (Death) beta = 4.15; collective beta = 2.46
  - Death rates: FARC 0.2%, common delinquency 8.0%; prtest z = -5.71
  - Cox subsample N = 461; cumulative rescue by day 98: FARC 0.27, ELN 0.29,
    paramilitaries 0.33; Payment HR paramilitaries = 4.28;
    public-sector Payment = 0.20, Rescue = 0.24; post-2011 Payment = 0.51,
    Rescue = 4.86

Step 3 -- Inspect outputs:
  Open stata/outputs/Datos_Graficas_Cox.xlsx (sheets Survival_Pago,
  Hazard_Pago, Survival_Rescate, Hazard_Rescate, Letalidad, Survival_Fuga,
  Hazard_Fuga). After a full run, check stata/outputs/cox_appendix_rows.txt
  and the log files in stata/.

Step 4 -- Read Section S8 of Appendix_2.pdf (mapping to the mechanism model).

Step 5 -- Recompile the document (optional):
  In tex/, run

      pdflatex Appendix_2
      bibtex   Appendix_2
      pdflatex Appendix_2
      pdflatex Appendix_2

--------------------------------------------------------------------------------
4. DATA PROVENANCE
--------------------------------------------------------------------------------

Primary CNMH exports (not included in this package; available from CNMH):
  CasosSE_2025_09.xlsx    SIEVCAC case export (September 2025)
  VictimasSE_202509.xlsx  SIEVCAC victim export (September 2025)

The package includes Data_merge.dta, the merged master file used in the
author's Stata scripts. The checkpoint analytical_sample.dta reproduces the
post-construction estimation file (n = 1,125 after dropping eight cases with
missing geographic zone).

Outcome coding (Y_Resultado):
  1 = Payment   (PAGO, CANJE, ...)
  3 = Rescue    (RESCATE)
  0 = Escape/release without payment (FUGA, POR PRESIÓN/INTERMEDIACIÓN)
  2 = Death     (MUERTO EN CAUTIVERIO, ASESINADO)

--------------------------------------------------------------------------------
5. NOTES
--------------------------------------------------------------------------------

- Download the entire Appendix_2/ folder. The master file must be at
  stata/data/Data_merge.dta (~37 MB). Without it, no script can run.
- Set the working directory to Appendix_2/ for replicate.do, or to
  stata/ for individual scripts. Do not run from a parent directory
  above Appendix_2/.
- Cox models use DíasdeCautiverio; the Cox subsample (N = 461) excludes
  cases with missing or zero-day captivity duration.
- MNL margins in the appendix match Stata -margins- with unconditional
  predictions; raw outcome shares differ because margins average over all
  covariates.
- Original Spanish scripts: Data/Centro_memoria_Histórica/Modelo_Setlen.do
  and Prueba_Robusto.do. Package scripts are English translations with
  relative paths only.
- No paths outside Appendix_2/ are required for replication once the folder
  is downloaded.

Troubleshooting:
- "Could not find stata/run_all.do"
  -> Use Method B (full path to replicate.do), or Method A with cwd = Appendix_2/.
  Confirm the folder contains both replicate.do and stata/run_all.do.
- "data/Data_merge.dta not found" -> confirm stata/data/Data_merge.dta (~37 MB).
- Empty stata/outputs/ (except Datos_Graficas_Cox.xlsx) -> run replicate.do first.
- Results window shows only part of the output -> open stata/outputs/replication.log.
================================================================================
