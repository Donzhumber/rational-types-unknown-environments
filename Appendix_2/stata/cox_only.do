* Cox-only replication for Appendix 2 (four cause-specific equations)
version 19
clear all
set more off
set linesize 255

capture confirm file "_setup_paths.do"
if _rc capture cd stata
capture confirm file "_setup_paths.do"
if _rc {
    display as error "Set the working directory to Appendix_2/ or Appendix_2/stata/."
    exit 601
}
include _setup_paths.do

use "`datadir'/Data_merge.dta", clear

* Rename raw variables to English
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

destring Year Month Day, replace force
gen Event_Date = mdy(Month, Day, Year)
drop if missing(Event_Date)
drop if Year == 0

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
gen Kidnap_Structure   = (Kidnap_Modality != "INDIVIDUAL") + 1
gen GAULA_Intervention = (GAULA_Group == "GAULA")
gen Victim_Sex = 1
replace Victim_Sex = 2 if Sex == "MUJER"
replace Victim_Sex = 3 if Sex == "SIN INFORMACION"
gen ln_Victims = ln(Total_Victims)
gen Historical_Period = .
replace Historical_Period = 1 if Year <= 1997
replace Historical_Period = 2 if Year >= 1998 & Year <= 2002
replace Historical_Period = 3 if Year >= 2003 & Year <= 2010
replace Historical_Period = 4 if Year >= 2011

keep if Kidnap_Type == "EXTORSIVO"
drop if inlist(Release_Type, "ND", "OTRO", "NA") | missing(Release_Type)
gen Y_Outcome = .
replace Y_Outcome = 1 if inlist(Release_Type, "PAGO", "CANJE", "PAGO - POR PRESIÓN/INTERMEDIACIÓN", "CANJE - PAGO")
replace Y_Outcome = 3 if Release_Type == "RESCATE"
replace Y_Outcome = 0 if inlist(Release_Type, "FUGA", "POR PRESIÓN/INTERMEDIACIÓN")
capture confirm variable Victim_Status
if _rc == 0 {
    replace Y_Outcome = 2 if inlist(Victim_Status, "MUERTO EN CAUTIVERIO", "ASESINADO")
}
gen Duration_Categ = 4
replace Duration_Categ = 1 if Days_Captivity <= 2
replace Duration_Categ = 2 if Days_Captivity > 2 & Days_Captivity <= 30
replace Duration_Categ = 3 if Days_Captivity > 30 & Days_Captivity != .
drop if missing(Geographic_Zone)  | missing(Y_Outcome)       | missing(Captor_Group) ///
    | missing(Victim_Profile)     | missing(Modus_Operandi)  | missing(Kidnap_Structure) ///
    | missing(GAULA_Intervention) | missing(Victim_Sex)      | missing(Duration_Categ) ///
    | missing(ln_Victims)         | missing(Historical_Period)

drop if missing(Days_Captivity) | Days_Captivity <= 0

* Value labels for factor variables
label define Lab_Group     1 "FARC" 2 "ELN" 3 "Paramilitaries" 4 "Common_delinquency", replace
label define Lab_Profile   1 "High_income" 2 "Middle_class" 3 "Vulnerable" 4 "Public_sector" 5 "Unknown", replace
label define Lab_Zone      1 "Metropolis" 2 "Andean" 3 "Caribbean" 4 "Pacific" 5 "East_Jungle", replace
label define Lab_Modus     1 "Direct_attack" 2 "Checkpoint" 3 "Military_incursion" 4 "Deception" 5 "Unknown", replace
label define Lab_Structure 1 "Individual" 2 "Collective", replace
label define Lab_GAULA     0 "No_GAULA" 1 "GAULA", replace
label define Lab_Sex       1 "Male" 2 "Female" 3 "Unknown", replace
label define Lab_Period    1 "Pre_1998" 2 "1998_2002" 3 "2003_2010" 4 "Post_2011", replace
label define Lab_Y         0 "Escape_or_Release" 1 "Payment" 2 "Death" 3 "Rescue", replace

label values Captor_Group       Lab_Group
label values Victim_Profile     Lab_Profile
label values Geographic_Zone    Lab_Zone
label values Modus_Operandi     Lab_Modus
label values Kidnap_Structure   Lab_Structure
label values GAULA_Intervention Lab_GAULA
label values Victim_Sex         Lab_Sex
label values Historical_Period  Lab_Period
label values Y_Outcome          Lab_Y

quietly count
local N_cox = r(N)
display "Cox sample N = `N_cox'"

local causes Payment Rescue Death Escape
local codes  1 3 2 0

forvalues k = 1/4 {
    local cause : word `k' of `causes'
    local code  : word `k' of `codes'
    display _n "=== COX `cause' ==="
    stset Days_Captivity, failure(Y_Outcome==`code')
    stcox i.Captor_Group i.Victim_Profile i.Geographic_Zone ///
          i.Modus_Operandi i.Kidnap_Structure i.GAULA_Intervention ///
          i.Victim_Sex ln_Victims i.Historical_Period, robust
    estimates store Cox_`cause'
}

* Export appendix rows
file open fh using "`outdir'/cox_appendix_rows.txt", write replace
local coefs ///
    "2.Captor_Group ELN" ///
    "3.Captor_Group Paramilitaries" ///
    "4.Captor_Group CommonDel" ///
    "2.Victim_Profile MiddleClass" ///
    "3.Victim_Profile Vulnerable" ///
    "4.Victim_Profile PublicSector" ///
    "2.Geographic_Zone Andean" ///
    "3.Geographic_Zone Caribbean" ///
    "5.Geographic_Zone EastJungle" ///
    "2.Modus_Operandi Roadblock" ///
    "4.Modus_Operandi Deception" ///
    "2.Kidnap_Structure GroupStruct" ///
    "1.GAULA_Intervention GAULA" ///
    "ln_Victims lnTotal" ///
    "2.Historical_Period Crisis" ///
    "3.Historical_Period DemSec" ///
    "4.Historical_Period Post2011"

forvalues k = 1/4 {
    local cause : word `k' of `causes'
    quietly estimates restore Cox_`cause'
    file write fh _n "EQUATION `cause'" _n
    foreach item of local coefs {
        local coefname : word 1 of `item'
        local label    : word 2 of `item'
        capture scalar test = _b[`coefname']
        if _rc == 0 {
            scalar hr = exp(_b[`coefname'])
            scalar pv = 2*normal(-abs(_b[`coefname']/_se[`coefname']))
            file write fh "`label' " %9.4f (hr) " " %9.4f (pv) _n
        }
        else {
            file write fh "`label' NA NA" _n
        }
    }
    file write fh "N `N_cox'" _n
    file write fh "Wald " %9.2f (e(chi2)) _n
}
file close fh
display _n "DONE: `outdir'/cox_appendix_rows.txt"
