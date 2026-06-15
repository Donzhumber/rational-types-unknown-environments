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
* ATTRITION (re-estimated on extortive stratum from master — see README)
* --------------------------------------------------------------------------
display as text _n ">>> For full attrition logit on n=11,270 extortive cases,"
display as text "    run microstructure_replication.do from the top. <<<"

* --------------------------------------------------------------------------
* MAIN MNL
* --------------------------------------------------------------------------
display as text _n ">>> MAIN MULTINOMIAL LOGIT (n=`=_N', base = Payment) <<<"
mlogit Y_Resultado i.Grupo_Responsable i.Perfil_Victima i.Zona_Geografica ///
       i.Modus_Operandi i.Estructura_Secuestro i.Intervencion_GAULA ///
       i.Sexo_Victima i.Duracion_Categ ln_TotalVictimas i.Periodo_Historico, ///
       vce(cluster Municipio_Num) base(1) rrr
estimates store Main

display as text "Wald chi2 = " %9.2f e(chi2) ", df = " e(df) ", p = " %6.4f e(p)
quietly mlogit Y_Resultado i.Grupo_Responsable i.Perfil_Victima i.Zona_Geografica ///
       i.Modus_Operandi i.Estructura_Secuestro i.Intervencion_GAULA ///
       i.Sexo_Victima i.Duracion_Categ ln_TotalVictimas i.Periodo_Historico, ///
       base(1)
display as text "Pseudo R2 = " %5.2f (1 - e(ll)/e(ll_0))
estimates restore Main

* --------------------------------------------------------------------------
* ROBUSTNESS
* --------------------------------------------------------------------------
quietly mlogit Y_Resultado i.Grupo_Responsable i.Perfil_Victima i.Zona_Geografica ///
       i.Modus_Operandi i.Estructura_Secuestro i.Intervencion_GAULA ///
       i.Sexo_Victima i.Duracion_Categ ln_TotalVictimas i.Periodo_Historico, ///
       vce(robust) base(1)
estimates store RobSE

quietly mlogit Y_Resultado i.Grupo_Responsable i.Perfil_Victima i.Zona_Geografica ///
       i.Modus_Operandi i.Estructura_Secuestro i.Intervencion_GAULA ///
       i.Sexo_Victima ln_TotalVictimas i.Periodo_Historico, ///
       vce(cluster Municipio_Num) base(1)
estimates store NoDur

display as text _n ">>> ROBUSTNESS TABLE (Death equation coefficients) <<<"
estimates table Main RobSE NoDur, ///
    keep([Death]4.Grupo_Responsable [Death]2.Estructura_Secuestro) ///
    b(%9.3f) star(.05 .01 .001)

* Death proportion test: FARC vs common delinquency
gen Muerte_Dummy = (Y_Resultado == 2)
display as text _n ">>> PROPORTION TEST: DEATH RATE FARC vs COMMON DELINQUENCY <<<"
prtest Muerte_Dummy if inlist(Grupo_Responsable, 1, 4), by(Grupo_Responsable)

* --------------------------------------------------------------------------
* SUEST IIA
* --------------------------------------------------------------------------
display as text _n ">>> SUEST IIA TEST <<<"
quietly {
    mlogit Y_Resultado i.Grupo_Responsable i.Perfil_Victima i.Zona_Geografica ///
           i.Modus_Operandi i.Estructura_Secuestro i.Intervencion_GAULA ///
           i.Sexo_Victima i.Duracion_Categ ln_TotalVictimas i.Periodo_Historico, base(1)
    estimates store A
    mlogit Y_Resultado i.Grupo_Responsable i.Perfil_Victima i.Zona_Geografica ///
           i.Modus_Operandi i.Estructura_Secuestro i.Intervencion_GAULA ///
           i.Sexo_Victima i.Duracion_Categ ln_TotalVictimas i.Periodo_Historico ///
           if Y_Resultado != 2, base(1)
    estimates store B
    suest A B, cluster(Municipio_Num) noomitted
}
test [A_Escape_or_Release]_b[4.Periodo_Historico] = [B_Escape_or_Release]_b[4.Periodo_Historico]

display as text _n ">>> SAMPLE SELECTION / ROBUSTNESS COMPLETE <<<"
