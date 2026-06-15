================================================================================
REPLICATION PACKAGE
Appendix 1: Macro Dynamics -- Cointegration and the VEC Foundation
Supplement to "Identifying Rational Types in Unknown Environments"
Author: Humberto Bernal (Economics Program, Universidad Colegio Mayor de
Cundinamarca; Ph.D. in Economics, Universidad de los Andes)
================================================================================

This folder contains everything needed to replicate the results reported in
Appendix 1 (and summarized in Section 2.2 of the main paper): the data, the
MATLAB estimation scripts, the exported trend series, and the LaTeX source of
the document. All scripts, comments, console output, and figure labels are in
English.

--------------------------------------------------------------------------------
1. CONTENTS
--------------------------------------------------------------------------------

tex/
  Appendix_1.tex       LaTeX source of the appendix.
  Appendix_1.pdf       Compiled document (regenerate with Step 4 if missing).
  references.bib       Bibliography database.
  econsocart.cls,
  econsocart.cfg,
  econsoc.bst,
  textcase.sty         Econometric Society class files needed to compile.

matlab/Trend_Difference/
  check_trend_equivalence.m
                       Validates Section S7: reads VEC and LLT trend CSVs,
                       builds d_t = tau^VEC - tau^LLT on the full estimation
                       sample (Jul 1954 - Dec 2025), runs ADF and KPSS, exports
                       trend_difference.csv, plots.
  verify_appendix1.m   Full diagnostic against Appendix_1.tex (optional).
  trend_difference.csv Output of check_trend_equivalence.m (full sample).

matlab/VEC/
  Data.csv             Monthly data, July 1954 - December 2025 (T = 858).
                       Relevant columns: Cantidad_Extorsivo (kidnapping for
                       ransom, K^E), Cantidad_Simple (non-extortive
                       kidnapping, K^NE), BR_r (commercial lending interest
                       rate, r), mes (month, format YYYYmM).
  VEC_2T.m             Full VEC pipeline: Table I unit-root tests (ADF and
                       KPSS in levels and first differences, 12 lags,
                       constant; p-values with tabulated bounds),
                       Johansen trace test with lag-sensitivity analysis,
                       VEC(p=36, r=1, unrestricted constant) estimation,
                       adjustment coefficients with standard errors,
                       error-correction-term diagnostics, residual
                       diagnostics (Ljung-Box, Jarque-Bera, ARCH on VEC
                       residuals; separate ARCH on ECT),
                       Gonzalo-Granger decomposition, impulse responses
                       (Cholesky and Blanchard-Quah), FEVD, and forecasts.
  Tendencias_SE.csv    Output: observed series and VEC-GG long-run trends.
                       Columns: Date, Kidnap_Ransom_Obs, Kidnap_Ransom_Trend,
                       NonExtortionate_Obs, NonExtortionate_Trend.

matlab/SE_Kalman/
  Data.csv             Same data file (identical copy; the script reads it
                       from its own folder).
  LLT_Extorsivo.m      Local Linear Trend state-space model for K^E:
                       stochastic level, AR(8) slope, AR(5) cycle, irregular
                       term. Maximum likelihood via the Kalman filter
                       (ssm/estimate), parameter table with p-values,
                       innovation standard deviations in levels, and
                       smoothed components.
  LLT_decomposition.csv
                       Output: smoothed components of the LLT (Slope AR(8),
                       Cycle AR(5)). Columns: Date, Observed, Trend, Slope,
                       Cycle, Residual. Included in the package; regenerated
                       when LLT_Extorsivo.m is run.

--------------------------------------------------------------------------------
2. REQUIREMENTS
--------------------------------------------------------------------------------

- MATLAB R2025b or later with the Econometrics Toolbox (adftest, kpsstest,
  pptest, jcitest, vecm, ssm) and the Statistics and Machine Learning
  Toolbox.
- A LaTeX distribution (TeX Live 2024 or later) with pdflatex and bibtex,
  to recompile the document.

--------------------------------------------------------------------------------
3. HOW TO REPLICATE
--------------------------------------------------------------------------------

Step 1 -- VEC results (Sections S2-S5 of the appendix):
  In MATLAB, set the working directory to matlab/VEC and run

      VEC_2T

  The script prints every number reported in the appendix: Table I
  (unit-root tests), the Johansen trace test (p = 36, model H1) with its lag-sensitivity
  analysis, the cointegrating vector beta = (0.0374, -0.0722, -0.0490), the
  adjustment vector alpha = (-2.937, 0.358, -0.065) with standard errors,
  the error-correction-term and residual diagnostics, the Gonzalo-Granger
  factors and loading matrix, impulse responses, the FEVD tables, and the
  forecast table. It also regenerates Tendencias_SE.csv. Note that the
  bootstrap standard errors for beta (200 replications) take several
  minutes.

Step 2 -- LLT results (Section S6 of the appendix):
  Set the working directory to matlab/SE_Kalman and run

      LLT_Extorsivo

  The script estimates the state-space model (17 parameters), prints the
  coefficient table with p-values and the innovation standard deviations in
  levels, and regenerates the decomposition CSV.
  Note: quasi-Newton maximum likelihood on a 15-state model can converge to
  slightly different points depending on MATLAB version and starting
  values; innovation standard deviations should reproduce up to small
  numerical differences, with identical qualitative conclusions (all sigma
  parameters significant at p < 0.001).

Step 3 -- Trend equivalence (Section S7 of the appendix):
  Set the working directory to matlab/Trend_Difference and run

      check_trend_equivalence

  The script reads Kidnap_Ransom_Trend from VEC/Tendencias_SE.csv and
  Trend from SE_Kalman/LLT_decomposition.csv, builds
  d_t = tau^VEC - tau^LLT on the full estimation sample (Jul 1954 - Dec 2025,
  T = 858), and prints ADF and KPSS p-values (default bandwidth and lags = 12).
  It also exports trend_difference.csv and produces comparison plots.
  Expected results: ADF p = 0.001; KPSS (lags = 12) p = 0.071.

Step 3b -- Optional full diagnostic (all tables vs Appendix_1.tex):
  From matlab/Trend_Difference, run

      verify_appendix1

Step 4 -- Read Section S8 (relevance for the mechanism):
  Open tex/Appendix_1.pdf, Section S8, and Table "Mapping of Mechanism
  Inputs to Replicated Macro Results." That section explains how the
  macro results connect to the mechanism model and which package files
  support each claim. CSV files (Tendencias_SE.csv, LLT_decomposition.csv,
  trend_difference.csv) allow inspection without rerunning MATLAB.

Step 5 -- Recompile the document (optional):
  In tex/, run

      pdflatex Appendix_1
      bibtex   Appendix_1
      pdflatex Appendix_1
      pdflatex Appendix_1

--------------------------------------------------------------------------------
4. NOTES
--------------------------------------------------------------------------------

- The estimation sample is July 1954 - December 2025 (T = 858). All macro
  models and trend-equivalence tests on d_t use this full sample. Descriptive
  institutional figures in the main paper display 1964-2025.
- Isolated missing values in Data.csv are imputed inside the scripts
  (fillmissing: moving median of order three in the VEC script, linear
  interpolation in the LLT script).
- Relative to the author's working copies, the scripts in this package were
  translated to English and VEC_2T.m was reduced to the single pipeline
  that produces the appendix results (an unrelated auxiliary model block
  and a machine-specific temporary-file path were removed). Estimation
  commands, options, and numerical settings are unchanged.
- Section S8 (Relevance for the mechanism) is self-contained for readers who
  download only this folder: it maps each mechanism-relevant macro result to
  appendix sections and package files (Table in S8). No paths outside
  Appendix_1/ are required.
- check_trend_equivalence.m adapts the trend-comparison logic of
  SE_Kalman/Tendencias.m from the author's working folder, streamlined
  for the replication package: it reads the pre-computed CSV outputs
  instead of re-running LLT and VEC from scratch.
================================================================================
