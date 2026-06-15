* ==============================================================================
* APPENDIX 2 REPLICATION: MICROSTRUCTURE AND COMPETING HAZARDS
* Supplement to "Identifying Rational Types in Unknown Environments"
* Models: sample-selection logit, multinomial logit (MNL), Cox cause-specific
* hazards (Payment, Rescue, Death, Escape/Release).
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

display as text _n(2) "=== APPENDIX 2 REPLICATION ==="
display as text "Data directory: `datadir'"
display as text "Output directory: `outdir'"

use "`datadir'/Data_merge.dta", clear

quietly count
local N_raw = r(N)

* ------------------------------------------------------------------------------
* 1. DATE PARSING
* ------------------------------------------------------------------------------
destring Año Mes Día, replace force
gen Fecha = mdy(Mes, Día, Año)
drop if missing(Fecha)
format Fecha %td
drop if Año == 0

* ------------------------------------------------------------------------------
* 2. VARIABLE CONSTRUCTION (before outcome filter for attrition test)
* ------------------------------------------------------------------------------
replace TipodeLiberación = strtrim(TipodeLiberación)

gen Grupo_Responsable = .
replace Grupo_Responsable = 1 if strpos(DescripciónPresuntoResponsable, "FARC") > 0
replace Grupo_Responsable = 2 if strpos(DescripciónPresuntoResponsable, "ELN") > 0 | DescripciónPresuntoResponsable == "ERG"
replace Grupo_Responsable = 3 if PresuntoResponsable == "GRUPO PARAMILITAR" | PresuntoResponsable == "GRUPO POSDESMOVILIZACIÓN"
replace Grupo_Responsable = 4 if Grupo_Responsable == .

replace Ocupación = strtrim(Ocupación)
gen Perfil_Victima = .
replace Perfil_Victima = 1 if inlist(Ocupación, "GANADERO/HACENDADO", "COMERCIANTE", "EMPRESARIO - INDUSTRIAL")
replace Perfil_Victima = 2 if inlist(Ocupación, "EMPLEADO", "PROFESIONAL", "ESTUDIANTE", "ADMINISTRADOR DE FINCA", "PERSONAL DE SALUD") | Ocupación == "PENSIONADO"
replace Perfil_Victima = 3 if inlist(Ocupación, "CAMPESINO", "CONDUCTOR/MOTORISTA", "AMA DE CASA", "OBRERO", "TRABAJADOR DE FINCA") | inlist(Ocupación, "MINERO", "PESCADOR", "SEGURIDAD PRIVADA", "RELIGIOSO")
replace Perfil_Victima = 4 if inlist(Ocupación, "FUNCIONARIO PÚBLICO", "FUERZA PÚBLICA")
replace Perfil_Victima = 5 if Perfil_Victima == .

replace Departamento = strtrim(Departamento)
replace Municipio = strtrim(Municipio)
gen Zona_Geografica = .
replace Zona_Geografica = 2 if inlist(Departamento, "ANTIOQUIA", "BOYACA", "CALDAS", "CUNDINAMARCA", "HUILA") | inlist(Departamento, "QUINDIO", "RISARALDA", "SANTANDER", "TOLIMA")
replace Zona_Geografica = 3 if inlist(Departamento, "ATLANTICO", "BOLIVAR", "CESAR", "CORDOBA", "LA GUAJIRA") | inlist(Departamento, "MAGDALENA", "SUCRE", "SAN ANDRES")
replace Zona_Geografica = 4 if inlist(Departamento, "CAUCA", "CHOCO", "NARIÑO", "VALLE DEL CAUCA","NORTE DE SANTANDER")
replace Zona_Geografica = 5 if inlist(Departamento, "AMAZONAS", "ARAUCA", "CAQUETA", "CASANARE", "GUAINIA") | inlist(Departamento, "GUAVIARE", "META", "PUTUMAYO", "VAUPES", "VICHADA")
replace Zona_Geografica = 1 if inlist(Municipio, "BOGOTA, D.C.", "SOACHA", "MEDELLIN", "BELLO", "ENVIGADO") | inlist(Municipio, "SABANETA", "SANTIAGO DE CALI", "BARRANQUILLA", "BUCARAMANGA", "CUCUTA")
replace Zona_Geografica = 1 if inlist(Municipio, "FLORIDABLANCA", "CARTAGENA DE INDIAS","PEREIRA", "MANIZALES")

replace Modalidad = strtrim(Modalidad)
gen Modus_Operandi = .
replace Modus_Operandi = 1 if inlist(Modalidad, "INTERCEPTACIÓN", "ASALTO", "PERSECUCIÓN")
replace Modus_Operandi = 2 if inlist(Modalidad, "RETÉN", "PESCA MILAGROSA", "RUTA")
replace Modus_Operandi = 3 if inlist(Modalidad, "INCURSIÓN", "ACCIÓN BÉLICA", "RETENCIÓN/EJECUCIÓN")
replace Modus_Operandi = 4 if inlist(Modalidad, "ENGAÑO", "CITACIÓN", "CANJE/INTERCAMBIO")
replace Modus_Operandi = 5 if Modus_Operandi == .

gen Estructura_Secuestro = (ModalidaddeSecuestro != "INDIVIDUAL") + 1
gen Intervencion_GAULA = (GAULA == "GAULA")
gen Sexo_Victima = 1
replace Sexo_Victima = 2 if Sexo == "MUJER"
replace Sexo_Victima = 3 if Sexo == "SIN INFORMACION"
encode Municipio, gen(Municipio_Num)
gen ln_TotalVictimas = ln(TotaldeVíctimasdelCaso)

gen Periodo_Historico = .
replace Periodo_Historico = 1 if Año <= 1997
replace Periodo_Historico = 2 if Año >= 1998 & Año <= 2002
replace Periodo_Historico = 3 if Año >= 2003 & Año <= 2010
replace Periodo_Historico = 4 if Año >= 2011

label define Lab_Y 0 "Escape_or_Release" 1 "Payment" 2 "Death" 3 "Rescue"
label define Lab_Grupo 1 "FARC" 2 "ELN" 3 "Paramilitaries" 4 "Common_delinquency"
label define Lab_Perfil 1 "High_income" 2 "Middle_class" 3 "Vulnerable" 4 "Public_sector" 5 "Unknown"
label define Lab_Zona 1 "Metropolis" 2 "Andean" 3 "Caribbean" 4 "Pacific" 5 "East_Jungle"
label define Lab_Modus 1 "Direct_attack" 2 "Checkpoint" 3 "Military_incursion" 4 "Deception" 5 "Unknown"
label define Lab_Estructura 1 "Individual" 2 "Collective"
label define Lab_GAULA 0 "No_GAULA" 1 "GAULA"
label define Lab_Sexo 1 "Male" 2 "Female" 3 "Unknown"
label define Lab_Duracion 1 "Express_<=2d" 2 "Short_<=30d" 3 "Long_>30d" 4 "Missing"
label define Lab_Periodo 1 "Pre_1998" 2 "1998_2002" 3 "2003_2010" 4 "Post_2011"

* ------------------------------------------------------------------------------
* 3. SAMPLE COUNTS
* ------------------------------------------------------------------------------
quietly count
local N_valid_date = r(N)
display as text _n ">>> SAMPLE CONSTRUCTION <<<"
display as text "Universe (merged CNMH records): `N_raw'"
display as text "Valid event date: `N_valid_date'"

preserve
keep if TipodeSecuestro == "EXTORSIVO"
quietly count
local N_extortive = r(N)
display as text "Extortionate stratum: `N_extortive'"

gen Missing_Outcome = inlist(TipodeLiberación, "ND", "OTRO", "NA") | missing(TipodeLiberación)

quietly count if Missing_Outcome == 0
local N_observable = r(N)
display as text "Observable outcome (extortive): `N_observable'"

* Attrition logit on full extortive stratum
display as text _n ">>> ATTRITION LOGIT (extortive stratum) <<<"
logit Missing_Outcome i.Grupo_Responsable i.Perfil_Victima i.Zona_Geografica ///
      i.Modus_Operandi i.Intervencion_GAULA ln_TotalVictimas i.Periodo_Historico, ///
      vce(cluster Municipio_Num) or
estimates store Attrition
display as text "Attrition logit N (complete covariates) = " e(N)

* Drop missing outcomes
drop if Missing_Outcome == 1

* Outcome variable
gen Y_Resultado = .
replace Y_Resultado = 1 if inlist(TipodeLiberación, "PAGO", "CANJE", "PAGO - POR PRESIÓN/INTERMEDIACIÓN", "CANJE - PAGO")
replace Y_Resultado = 3 if TipodeLiberación == "RESCATE"
replace Y_Resultado = 0 if inlist(TipodeLiberación, "FUGA", "POR PRESIÓN/INTERMEDIACIÓN")
capture confirm variable SituaciónActualdelaVíctima
if _rc == 0 {
    replace Y_Resultado = 2 if inlist(SituaciónActualdelaVíctima, "MUERTO EN CAUTIVERIO", "ASESINADO")
}
label values Y_Resultado Lab_Y

gen Duracion_Categ = 4
replace Duracion_Categ = 1 if DíasdeCautiverio <= 2
replace Duracion_Categ = 2 if DíasdeCautiverio > 2 & DíasdeCautiverio <= 30
replace Duracion_Categ = 3 if DíasdeCautiverio > 30 & DíasdeCautiverio != .
label values Duracion_Categ Lab_Duracion
label values Grupo_Responsable Lab_Grupo
label values Perfil_Victima Lab_Perfil
label values Zona_Geografica Lab_Zona
label values Modus_Operandi Lab_Modus
label values Estructura_Secuestro Lab_Estructura
label values Intervencion_GAULA Lab_GAULA
label values Sexo_Victima Lab_Sexo
label values Periodo_Historico Lab_Periodo

* Analytical sample: complete covariates for MNL (drops 8 cases with missing zone)
drop if missing(Zona_Geografica) | missing(Y_Resultado) | missing(Grupo_Responsable) ///
    | missing(Perfil_Victima) | missing(Modus_Operandi) | missing(Estructura_Secuestro) ///
    | missing(Intervencion_GAULA) | missing(Sexo_Victima) | missing(Duracion_Categ) ///
    | missing(ln_TotalVictimas) | missing(Periodo_Historico)

quietly count
local N_analytical = r(N)
display as text "Analytical sample (complete covariates): `N_analytical'"

quietly count if !missing(Municipio_Num)
local N_clusters = r(N)
quietly levelsof Municipio_Num, local(muni)
local K_clusters : word count `muni'
display as text "Municipal clusters: `K_clusters'"

* Outcome distribution
tab Y_Resultado, missing

* Save analytical sample
save "`outdir'/analytical_sample.dta", replace
save "`datadir'/analytical_sample.dta", replace

* ------------------------------------------------------------------------------
* 4. MULTINOMIAL LOGIT (MAIN)
* ------------------------------------------------------------------------------
display as text _n ">>> MULTINOMIAL LOGIT (MAIN, base = Payment) <<<"
mlogit Y_Resultado i.Grupo_Responsable i.Perfil_Victima i.Zona_Geografica ///
       i.Modus_Operandi i.Estructura_Secuestro i.Intervencion_GAULA ///
       i.Sexo_Victima i.Duracion_Categ ln_TotalVictimas i.Periodo_Historico, ///
       vce(cluster Municipio_Num) base(1) rrr
estimates store MNL_Main

* Model fit (report from the clustered RRR model; estat gof unavailable here)
scalar wald_chi2 = e(chi2)
scalar wald_df   = e(df)
scalar wald_p    = e(p)
quietly mlogit Y_Resultado i.Grupo_Responsable i.Perfil_Victima i.Zona_Geografica ///
       i.Modus_Operandi i.Estructura_Secuestro i.Intervencion_GAULA ///
       i.Sexo_Victima i.Duracion_Categ ln_TotalVictimas i.Periodo_Historico, ///
       base(1)
scalar pseudo_r2 = 1 - e(ll)/e(ll_0)
estimates restore MNL_Main
display as text "Wald chi2(`=wald_df') = " %9.2f wald_chi2 ", p = " %6.4f wald_p
display as text "Pseudo R2 = " %5.2f pseudo_r2

* Key RRRs for appendix (equation names from value labels; base = Payment)
display as text _n ">>> KEY RELATIVE RISK RATIOS <<<"
lincom [Escape_or_Release]4.Periodo_Historico
lincom [Escape_or_Release]4.Perfil_Victima
lincom [Rescue]2.Duracion_Categ
lincom [Rescue]3.Duracion_Categ

* Margins: rescue by zone, death by group
margins Zona_Geografica, predict(outcome(3))
margins Grupo_Responsable, predict(outcome(2))
margins Duracion_Categ, predict(outcome(3))
margins Perfil_Victima, predict(outcome(0))
margins Intervencion_GAULA, over(Duracion_Categ) predict(outcome(3))

* Export margins
margins Zona_Geografica, predict(outcome(3)) saving("`outdir'/margins_rescue_zone", replace)
margins Grupo_Responsable, predict(outcome(2)) saving("`outdir'/margins_death_group", replace)
margins Duracion_Categ, predict(outcome(3)) saving("`outdir'/margins_rescue_duration", replace)
margins Perfil_Victima, predict(outcome(0)) saving("`outdir'/margins_escape_profile", replace)

* ------------------------------------------------------------------------------
* 5. ROBUSTNESS AND IIA
* ------------------------------------------------------------------------------
display as text _n ">>> ROBUSTNESS: NO CLUSTER <<<"
quietly mlogit Y_Resultado i.Grupo_Responsable i.Perfil_Victima i.Zona_Geografica ///
       i.Modus_Operandi i.Estructura_Secuestro i.Intervencion_GAULA ///
       i.Sexo_Victima i.Duracion_Categ ln_TotalVictimas i.Periodo_Historico, ///
       vce(robust) base(1)
estimates store MNL_Robust

display as text _n ">>> ROBUSTNESS: NO DURATION <<<"
quietly mlogit Y_Resultado i.Grupo_Responsable i.Perfil_Victima i.Zona_Geografica ///
       i.Modus_Operandi i.Estructura_Secuestro i.Intervencion_GAULA ///
       i.Sexo_Victima ln_TotalVictimas i.Periodo_Historico, ///
       vce(cluster Municipio_Num) base(1)
estimates store MNL_NoDur

estimates table MNL_Main MNL_Robust MNL_NoDur, ///
    keep([Death]4.Grupo_Responsable [Death]2.Estructura_Secuestro) ///
    b(%9.3f) star(.05 .01 .001)

* Death rate by group + proportion test
gen Muerte_Dummy = (Y_Resultado == 2)
display as text _n ">>> DEATH RATES BY GROUP <<<"
tabstat Muerte_Dummy, by(Grupo_Responsable) stat(mean count)
prtest Muerte_Dummy if inlist(Grupo_Responsable, 1, 4), by(Grupo_Responsable)

* SUEST IIA test
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

* ------------------------------------------------------------------------------
* 6. COX CAUSE-SPECIFIC MODELS
* ------------------------------------------------------------------------------
local causes "Payment Rescue Death Escape"
local codes  "1 3 2 0"

forvalues k = 1/4 {
    local cause : word `k' of `causes'
    local code  : word `k' of `codes'
    display as text _n ">>> COX MODEL: `cause' (event code `code') <<<"
    stset DíasdeCautiverio, failure(Y_Resultado==`code')
    stcox i.Grupo_Responsable i.Perfil_Victima i.Zona_Geografica ///
          i.Modus_Operandi i.Estructura_Secuestro i.Intervencion_GAULA ///
          i.Sexo_Victima ln_TotalVictimas i.Periodo_Historico, robust
    estimates store Cox_`cause'
}

* Export key HR table via coef table
estimates restore Cox_Payment
estimates table Cox_Payment Cox_Rescue Cox_Death Cox_Escape, ///
    keep(3.Grupo_Responsable 4.Grupo_Responsable 4.Perfil_Victima 4.Periodo_Historico) ///
    b(%9.3f) star(.05 .01 .001)

* Cumulative rescue at day 100 by group
display as text _n ">>> CUMULATIVE RESCUE AT DAY 100 <<<"
estimates restore Cox_Rescue
stset DíasdeCautiverio, failure(Y_Resultado==3)
stcurve, survival at1(Grupo_Responsable=1) at2(Grupo_Responsable=2) ///
    at3(Grupo_Responsable=3) at4(Grupo_Responsable=4) range(0 100) ///
    outfile("`outdir'/cumul_rescue_d100", replace) nodraw

* Export cumulative rescue probabilities at the terminal grid point
preserve
use "`outdir'/cumul_rescue_d100.dta", clear
quietly summarize _t
local tday = r(max)
keep if _t == `tday'
local groups "FARC ELN Paramilitaries CommonDel"
file open fh using "`outdir'/cox_cumul_day100.txt", write replace
file write fh "day `tday'" _n
forvalues g = 1/4 {
    local lab : word `g' of `groups'
    scalar surv = surv`g'[1]
    scalar cumul = 1 - surv
    file write fh "`lab' " %6.4f (cumul) _n
    display as text "`lab' Pr(Rescue by day `tday') = " %6.4f (cumul)
}
file close fh
restore

display as text _n ">>> Updating Datos_Graficas_Cox.xlsx from cumul_rescue_d100.dta <<<"
capture noisily shell python3 "`pkgroot'/export_cox_xlsx.py" "`outdir'"

display as text _n "=== REPLICATION COMPLETE ==="
display as text "Outputs saved in: `outdir'"
