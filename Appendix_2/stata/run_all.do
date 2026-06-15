* ==============================================================================
* APPENDIX 2 -- RUN ALL REPLICATION SCRIPTS
* Execute from Appendix_2/stata/ or via Appendix_2/replicate.do
* ==============================================================================
version 19
clear all
set more off
set linesize 120

capture confirm file "_setup_paths.do"
if _rc capture cd stata
capture confirm file "_setup_paths.do"
if _rc {
    display as error "Set the working directory to Appendix_2/ or Appendix_2/stata/."
    exit 601
}
include _setup_paths.do

* Results window truncates long runs; save full output and enlarge buffer.
capture set scrollbufsize 500000
set linesize 200
capture log close _all
log using "`outdir'/replication.log", replace text
display as text _n "Logging full output to: `outdir'/replication.log"
display as text "(Open this file if Results does not show everything.)" _n

display as text _n(2) "=== APPENDIX 2: FULL REPLICATION PIPELINE ==="

do "`pkgroot'/microstructure_replication.do"
do "`pkgroot'/sample_selection_robustness.do"
do "`pkgroot'/cox_only.do"

capture log close _all
display as text _n(2) "=== REPLICATION COMPLETE ==="
display as text "Full log: `outdir'/replication.log"
display as text "Outputs:  `outdir'"
