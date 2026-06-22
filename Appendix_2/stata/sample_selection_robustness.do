* ==============================================================================
* APPENDIX 2: SAMPLE SELECTION AND MNL ROBUSTNESS
* Attrition logit, main MNL, robustness variants, SUEST IIA test.
* Run from stata/ after microstructure_replication.do OR standalone from
* data/Data_merge.dta using the same variable construction block.
* ==============================================================================
version 19
clear all
set more off

capture confirm file "_setup_paths.do"
if _rc capture cd stata
capture confirm file "_setup_paths.do"
if _rc {
    display as error "Set the working directory to Appendix_2/ or Appendix_2/stata/."
    exit 601
}
include _setup_paths.do

include _load_analytical_sample.do

* --------------------------------------------------------------------------
* ATTRITION (re-estimated on extortive stratum from master --- see README)
* --------------------------------------------------------------------------
display as text _n ">>> For full attrition logit on n=11,270 extortive cases,"
display as text "    run microstructure_replication.do from the top. <<<"

* --------------------------------------------------------------------------
* MAIN MNL
* --------------------------------------------------------------------------
display as text _n ">>> MAIN MULTINOMIAL LOGIT (n=`=_N', base = Payment) <<<"
mlogit Y_Outcome i.Captor_Group i.Victim_Profile i.Geographic_Zone ///
       i.Modus_Operandi i.Kidnap_Structure i.GAULA_Intervention ///
       i.Victim_Sex i.Duration_Categ ln_Victims i.Historical_Period, ///
       vce(cluster Municipality_ID) base(1) rrr
estimates store Main

display as text "Wald chi2 = " %9.2f e(chi2) ", df = " e(df) ", p = " %6.4f e(p)
quietly mlogit Y_Outcome i.Captor_Group i.Victim_Profile i.Geographic_Zone ///
       i.Modus_Operandi i.Kidnap_Structure i.GAULA_Intervention ///
       i.Victim_Sex i.Duration_Categ ln_Victims i.Historical_Period, ///
       base(1)
display as text "Pseudo R2 = " %5.2f (1 - e(ll)/e(ll_0))
estimates restore Main

* --------------------------------------------------------------------------
* ROBUSTNESS
* --------------------------------------------------------------------------
quietly mlogit Y_Outcome i.Captor_Group i.Victim_Profile i.Geographic_Zone ///
       i.Modus_Operandi i.Kidnap_Structure i.GAULA_Intervention ///
       i.Victim_Sex i.Duration_Categ ln_Victims i.Historical_Period, ///
       vce(robust) base(1)
estimates store RobSE

quietly mlogit Y_Outcome i.Captor_Group i.Victim_Profile i.Geographic_Zone ///
       i.Modus_Operandi i.Kidnap_Structure i.GAULA_Intervention ///
       i.Victim_Sex ln_Victims i.Historical_Period, ///
       vce(cluster Municipality_ID) base(1)
estimates store NoDur

display as text _n ">>> ROBUSTNESS TABLE (Death equation — β and p-value) <<<"
file open fh using "`outdir'/mnl_robustness_death.txt", write replace
foreach mdl in Main RobSE NoDur {
    quietly estimates restore `mdl'
    foreach coef in "4.Captor_Group" "2.Kidnap_Structure" {
        local b  = _b[Death:`coef']
        local se = _se[Death:`coef']
        local p  = 2*(1-normal(abs(`b'/`se')))
        display as text "`mdl' `coef': b=" %8.4f (`b') " p=" %6.4f (`p')
        file write fh "`mdl' `coef' " %8.4f (`b') " " %6.4f (`p') _n
    }
}
file close fh
display as text "(written to mnl_robustness_death.txt)"
estimates restore Main

* Death proportion test: FARC vs common delinquency
gen Death_Dummy = (Y_Outcome == 2)
display as text _n ">>> PROPORTION TEST: DEATH RATE FARC vs COMMON DELINQUENCY <<<"
prtest Death_Dummy if inlist(Captor_Group, 1, 4), by(Captor_Group)

* --------------------------------------------------------------------------
* SUEST IIA
* --------------------------------------------------------------------------
display as text _n ">>> SUEST IIA TEST <<<"
quietly {
    mlogit Y_Outcome i.Captor_Group i.Victim_Profile i.Geographic_Zone ///
           i.Modus_Operandi i.Kidnap_Structure i.GAULA_Intervention ///
           i.Victim_Sex i.Duration_Categ ln_Victims i.Historical_Period, base(1)
    estimates store A
    mlogit Y_Outcome i.Captor_Group i.Victim_Profile i.Geographic_Zone ///
           i.Modus_Operandi i.Kidnap_Structure i.GAULA_Intervention ///
           i.Victim_Sex i.Duration_Categ ln_Victims i.Historical_Period ///
           if Y_Outcome != 2, base(1)
    estimates store B
    suest A B, cluster(Municipality_ID) noomitted
}
test [A_Escape_or_Release]_b[4.Historical_Period] = [B_Escape_or_Release]_b[4.Historical_Period]

display as text _n ">>> SAMPLE SELECTION / ROBUSTNESS COMPLETE <<<"
