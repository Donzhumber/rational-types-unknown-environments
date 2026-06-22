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

* ------------------------------------------------------------------------------
* 0. RENAME RAW VARIABLES TO ENGLISH
* ------------------------------------------------------------------------------
rename Año                            Year
rename Mes                            Month
rename Día                            Day
rename TipodeLiberación               Release_Type
rename DescripciónPresuntoResponsable Suspect_Description
rename PresuntoResponsable            Suspect_Group
rename Ocupación                      Occupation
rename Departamento                   Department
rename Municipio                      Municipality
rename Modalidad                      Modality
rename ModalidaddeSecuestro           Kidnap_Modality
rename GAULA                          GAULA_Group
rename Sexo                           Sex
rename TotaldeVíctimasdelCaso         Total_Victims
rename TipodeSecuestro                Kidnap_Type
rename DíasdeCautiverio               Days_Captivity
capture rename SituaciónActualdelaVíctima Victim_Status

quietly count
local N_raw = r(N)

* ------------------------------------------------------------------------------
* 1. DATE PARSING
* ------------------------------------------------------------------------------
destring Year Month Day, replace force
gen Event_Date = mdy(Month, Day, Year)
drop if missing(Event_Date)
format Event_Date %td
drop if Year == 0

* ------------------------------------------------------------------------------
* 2. VARIABLE CONSTRUCTION (before outcome filter for attrition test)
* ------------------------------------------------------------------------------
replace Release_Type = strtrim(Release_Type)

gen Captor_Group = .
replace Captor_Group = 1 if strpos(Suspect_Description, "FARC") > 0
replace Captor_Group = 2 if strpos(Suspect_Description, "ELN") > 0 | Suspect_Description == "ERG"
replace Captor_Group = 3 if Suspect_Group == "GRUPO PARAMILITAR" | Suspect_Group == "GRUPO POSDESMOVILIZACIÓN"
replace Captor_Group = 4 if Captor_Group == .

replace Occupation = strtrim(Occupation)
gen Victim_Profile = .
replace Victim_Profile = 1 if inlist(Occupation, "GANADERO/HACENDADO", "COMERCIANTE", "EMPRESARIO - INDUSTRIAL")
replace Victim_Profile = 2 if inlist(Occupation, "EMPLEADO", "PROFESIONAL", "ESTUDIANTE", "ADMINISTRADOR DE FINCA", "PERSONAL DE SALUD") | Occupation == "PENSIONADO"
replace Victim_Profile = 3 if inlist(Occupation, "CAMPESINO", "CONDUCTOR/MOTORISTA", "AMA DE CASA", "OBRERO", "TRABAJADOR DE FINCA") | inlist(Occupation, "MINERO", "PESCADOR", "SEGURIDAD PRIVADA", "RELIGIOSO")
replace Victim_Profile = 4 if inlist(Occupation, "FUNCIONARIO PÚBLICO", "FUERZA PÚBLICA")
replace Victim_Profile = 5 if Victim_Profile == .

replace Department = strtrim(Department)
replace Municipality = strtrim(Municipality)
gen Geographic_Zone = .
replace Geographic_Zone = 2 if inlist(Department, "ANTIOQUIA", "BOYACA", "CALDAS", "CUNDINAMARCA", "HUILA") | inlist(Department, "QUINDIO", "RISARALDA", "SANTANDER", "TOLIMA")
replace Geographic_Zone = 3 if inlist(Department, "ATLANTICO", "BOLIVAR", "CESAR", "CORDOBA", "LA GUAJIRA") | inlist(Department, "MAGDALENA", "SUCRE", "SAN ANDRES")
replace Geographic_Zone = 4 if inlist(Department, "CAUCA", "CHOCO", "NARIÑO", "VALLE DEL CAUCA","NORTE DE SANTANDER")
replace Geographic_Zone = 5 if inlist(Department, "AMAZONAS", "ARAUCA", "CAQUETA", "CASANARE", "GUAINIA") | inlist(Department, "GUAVIARE", "META", "PUTUMAYO", "VAUPES", "VICHADA")
replace Geographic_Zone = 1 if inlist(Municipality, "BOGOTA, D.C.", "SOACHA", "MEDELLIN", "BELLO", "ENVIGADO") | inlist(Municipality, "SABANETA", "SANTIAGO DE CALI", "BARRANQUILLA", "BUCARAMANGA", "CUCUTA")
replace Geographic_Zone = 1 if inlist(Municipality, "FLORIDABLANCA", "CARTAGENA DE INDIAS","PEREIRA", "MANIZALES")

replace Modality = strtrim(Modality)
gen Modus_Operandi = .
replace Modus_Operandi = 1 if inlist(Modality, "INTERCEPTACIÓN", "ASALTO", "PERSECUCIÓN")
replace Modus_Operandi = 2 if inlist(Modality, "RETÉN", "PESCA MILAGROSA", "RUTA")
replace Modus_Operandi = 3 if inlist(Modality, "INCURSIÓN", "ACCIÓN BÉLICA", "RETENCIÓN/EJECUCIÓN")
replace Modus_Operandi = 4 if inlist(Modality, "ENGAÑO", "CITACIÓN", "CANJE/INTERCAMBIO")
replace Modus_Operandi = 5 if Modus_Operandi == .

gen Kidnap_Structure  = (Kidnap_Modality != "INDIVIDUAL") + 1
gen GAULA_Intervention = (GAULA_Group == "GAULA")
gen Victim_Sex = 1
replace Victim_Sex = 2 if Sex == "MUJER"
replace Victim_Sex = 3 if Sex == "SIN INFORMACION"
encode Municipality, gen(Municipality_ID)
gen ln_Victims = ln(Total_Victims)

gen Historical_Period = .
replace Historical_Period = 1 if Year <= 1997
replace Historical_Period = 2 if Year >= 1998 & Year <= 2002
replace Historical_Period = 3 if Year >= 2003 & Year <= 2010
replace Historical_Period = 4 if Year >= 2011

label define Lab_Y         0 "Escape_or_Release" 1 "Payment" 2 "Death" 3 "Rescue"
label define Lab_Group     1 "FARC" 2 "ELN" 3 "Paramilitaries" 4 "Common_delinquency"
label define Lab_Profile   1 "High_income" 2 "Middle_class" 3 "Vulnerable" 4 "Public_sector" 5 "Unknown"
label define Lab_Zone      1 "Metropolis" 2 "Andean" 3 "Caribbean" 4 "Pacific" 5 "East_Jungle"
label define Lab_Modus     1 "Direct_attack" 2 "Checkpoint" 3 "Military_incursion" 4 "Deception" 5 "Unknown"
label define Lab_Structure 1 "Individual" 2 "Collective"
label define Lab_GAULA     0 "No_GAULA" 1 "GAULA"
label define Lab_Sex       1 "Male" 2 "Female" 3 "Unknown"
label define Lab_Duration  1 "Express_<=2d" 2 "Short_<=30d" 3 "Long_>30d" 4 "Missing"
label define Lab_Period    1 "Pre_1998" 2 "1998_2002" 3 "2003_2010" 4 "Post_2011"

* ------------------------------------------------------------------------------
* 3. SAMPLE COUNTS
* ------------------------------------------------------------------------------
quietly count
local N_valid_date = r(N)
display as text _n ">>> SAMPLE CONSTRUCTION <<<"
display as text "Universe (merged CNMH records): `N_raw'"
display as text "Valid event date: `N_valid_date'"

preserve
keep if Kidnap_Type == "EXTORSIVO"
quietly count
local N_extortive = r(N)
display as text "Extortionate stratum: `N_extortive'"

gen Missing_Outcome = inlist(Release_Type, "ND", "OTRO", "NA") | missing(Release_Type)

quietly count if Missing_Outcome == 0
local N_observable = r(N)
display as text "Observable outcome (extortive): `N_observable'"

* Attrition logit on full extortive stratum
display as text _n ">>> ATTRITION LOGIT (extortive stratum) <<<"
logit Missing_Outcome i.Captor_Group i.Victim_Profile i.Geographic_Zone ///
      i.Modus_Operandi i.GAULA_Intervention ln_Victims i.Historical_Period, ///
      vce(cluster Municipality_ID) or
estimates store Attrition
display as text "Attrition logit N (complete covariates) = " e(N)

* Drop missing outcomes
drop if Missing_Outcome == 1

* Outcome variable
gen Y_Outcome = .
replace Y_Outcome = 1 if inlist(Release_Type, "PAGO", "CANJE", "PAGO - POR PRESIÓN/INTERMEDIACIÓN", "CANJE - PAGO")
replace Y_Outcome = 3 if Release_Type == "RESCATE"
replace Y_Outcome = 0 if inlist(Release_Type, "FUGA", "POR PRESIÓN/INTERMEDIACIÓN")
capture confirm variable Victim_Status
if _rc == 0 {
    replace Y_Outcome = 2 if inlist(Victim_Status, "MUERTO EN CAUTIVERIO", "ASESINADO")
}
label values Y_Outcome Lab_Y

gen Duration_Categ = 4
replace Duration_Categ = 1 if Days_Captivity <= 2
replace Duration_Categ = 2 if Days_Captivity > 2 & Days_Captivity <= 30
replace Duration_Categ = 3 if Days_Captivity > 30 & Days_Captivity != .
label values Duration_Categ    Lab_Duration
label values Captor_Group      Lab_Group
label values Victim_Profile    Lab_Profile
label values Geographic_Zone   Lab_Zone
label values Modus_Operandi    Lab_Modus
label values Kidnap_Structure  Lab_Structure
label values GAULA_Intervention Lab_GAULA
label values Victim_Sex        Lab_Sex
label values Historical_Period Lab_Period

* Analytical sample: complete covariates for MNL (drops 8 cases with missing zone)
drop if missing(Geographic_Zone)   | missing(Y_Outcome)       | missing(Captor_Group) ///
    | missing(Victim_Profile)      | missing(Modus_Operandi)  | missing(Kidnap_Structure) ///
    | missing(GAULA_Intervention)  | missing(Victim_Sex)      | missing(Duration_Categ) ///
    | missing(ln_Victims)          | missing(Historical_Period)

quietly count
local N_analytical = r(N)
display as text "Analytical sample (complete covariates): `N_analytical'"

quietly count if !missing(Municipality_ID)
local N_clusters = r(N)
quietly levelsof Municipality_ID, local(muni)
local K_clusters : word count `muni'
display as text "Municipal clusters: `K_clusters'"

* Outcome distribution
tab Y_Outcome, missing

* Save analytical sample
save "`outdir'/analytical_sample.dta", replace
save "`datadir'/analytical_sample.dta", replace

* ------------------------------------------------------------------------------
* 4. MULTINOMIAL LOGIT (MAIN)
* ------------------------------------------------------------------------------
display as text _n ">>> MULTINOMIAL LOGIT (MAIN, base = Payment) <<<"
mlogit Y_Outcome i.Captor_Group i.Victim_Profile i.Geographic_Zone ///
       i.Modus_Operandi i.Kidnap_Structure i.GAULA_Intervention ///
       i.Victim_Sex i.Duration_Categ ln_Victims i.Historical_Period, ///
       vce(cluster Municipality_ID) base(1) rrr
estimates store MNL_Main

* Model fit (report from the clustered RRR model; estat gof unavailable here)
scalar wald_chi2 = e(chi2)
scalar wald_df   = e(df)
scalar wald_p    = e(p)
quietly mlogit Y_Outcome i.Captor_Group i.Victim_Profile i.Geographic_Zone ///
       i.Modus_Operandi i.Kidnap_Structure i.GAULA_Intervention ///
       i.Victim_Sex i.Duration_Categ ln_Victims i.Historical_Period, ///
       base(1)
scalar pseudo_r2 = 1 - e(ll)/e(ll_0)
estimates restore MNL_Main
display as text "Wald chi2(`=wald_df') = " %9.2f wald_chi2 ", p = " %6.4f wald_p
display as text "Pseudo R2 = " %5.2f pseudo_r2

* Key RRRs for appendix (equation names from value labels; base = Payment)
display as text _n ">>> KEY RELATIVE RISK RATIOS <<<"
lincom [Escape_or_Release]4.Historical_Period
lincom [Escape_or_Release]4.Victim_Profile
lincom [Rescue]2.Duration_Categ
lincom [Rescue]3.Duration_Categ

* Margins: rescue by zone, death by group
margins Geographic_Zone,    predict(outcome(3))
margins Captor_Group,       predict(outcome(2))
margins Duration_Categ,     predict(outcome(3))
margins Victim_Profile,     predict(outcome(0))
margins GAULA_Intervention, over(Duration_Categ) predict(outcome(3))

* Export margins
margins Geographic_Zone, predict(outcome(3)) saving("`outdir'/margins_rescue_zone",     replace)
margins Captor_Group,    predict(outcome(2)) saving("`outdir'/margins_death_group",      replace)
margins Duration_Categ,  predict(outcome(3)) saving("`outdir'/margins_rescue_duration",  replace)
margins Victim_Profile,  predict(outcome(0)) saving("`outdir'/margins_escape_profile",   replace)

* ------------------------------------------------------------------------------
* 5. ROBUSTNESS AND IIA
* ------------------------------------------------------------------------------
display as text _n ">>> ROBUSTNESS: NO CLUSTER <<<"
quietly mlogit Y_Outcome i.Captor_Group i.Victim_Profile i.Geographic_Zone ///
       i.Modus_Operandi i.Kidnap_Structure i.GAULA_Intervention ///
       i.Victim_Sex i.Duration_Categ ln_Victims i.Historical_Period, ///
       vce(robust) base(1)
estimates store MNL_Robust

display as text _n ">>> ROBUSTNESS: NO DURATION <<<"
quietly mlogit Y_Outcome i.Captor_Group i.Victim_Profile i.Geographic_Zone ///
       i.Modus_Operandi i.Kidnap_Structure i.GAULA_Intervention ///
       i.Victim_Sex ln_Victims i.Historical_Period, ///
       vce(cluster Municipality_ID) base(1)
estimates store MNL_NoDur

display as text _n ">>> ROBUSTNESS TABLE (Death equation — β and p-value) <<<"
file open fh using "`outdir'/mnl_robustness_death.txt", write replace
foreach mdl in MNL_Main MNL_Robust MNL_NoDur {
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
estimates restore MNL_Main

* Death rate by group + proportion test
gen Death_Dummy = (Y_Outcome == 2)
display as text _n ">>> DEATH RATES BY GROUP <<<"
tabstat Death_Dummy, by(Captor_Group) stat(mean count)
prtest Death_Dummy if inlist(Captor_Group, 1, 4), by(Captor_Group)

* SUEST IIA test
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

* ------------------------------------------------------------------------------
* 6. COX CAUSE-SPECIFIC MODELS
* ------------------------------------------------------------------------------
local causes "Payment Rescue Death Escape"
local codes  "1 3 2 0"

forvalues k = 1/4 {
    local cause : word `k' of `causes'
    local code  : word `k' of `codes'
    display as text _n ">>> COX MODEL: `cause' (event code `code') <<<"
    stset Days_Captivity, failure(Y_Outcome==`code')
    stcox i.Captor_Group i.Victim_Profile i.Geographic_Zone ///
          i.Modus_Operandi i.Kidnap_Structure i.GAULA_Intervention ///
          i.Victim_Sex ln_Victims i.Historical_Period, robust
    estimates store Cox_`cause'
}

* Export key HR table via coef table
estimates restore Cox_Payment
estimates table Cox_Payment Cox_Rescue Cox_Death Cox_Escape, ///
    keep(3.Captor_Group 4.Captor_Group 4.Victim_Profile 4.Historical_Period) ///
    b(%9.3f) star(.05 .01 .001)

* Cumulative rescue at day 100 by group
display as text _n ">>> CUMULATIVE RESCUE AT DAY 100 <<<"
estimates restore Cox_Rescue
stset Days_Captivity, failure(Y_Outcome==3)
stcurve, survival at1(Captor_Group=1) at2(Captor_Group=2) ///
    at3(Captor_Group=3) at4(Captor_Group=4) range(0 100) ///
    outfile("`outdir'/cumul_rescue_d100", replace) nodraw

* Export cumulative rescue probabilities at the terminal grid point
use "`outdir'/cumul_rescue_d100.dta", clear
quietly summarize _t
local tday = r(max)
keep if _t == `tday'
local groups "FARC ELN Paramilitaries CommonDel"
file open fh using "`outdir'/cox_cumul_day100.txt", write replace
file write fh "day `tday'" _n
forvalues g = 1/4 {
    local lab : word `g' of `groups'
    scalar surv  = surv`g'[1]
    scalar cumul = 1 - surv
    file write fh "`lab' " %6.4f (cumul) _n
    display as text "`lab' Pr(Rescue by day `tday') = " %6.4f (cumul)
}
file close fh

display as text _n ">>> Updating Datos_Graficas_Cox.xlsx from cumul_rescue_d100.dta <<<"
capture noisily shell python3 "`pkgroot'/export_cox_xlsx.py" "`outdir'"

display as text _n "=== REPLICATION COMPLETE ==="
display as text "Outputs saved in: `outdir'"
