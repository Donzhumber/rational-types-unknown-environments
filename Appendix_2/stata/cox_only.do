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
destring Año Mes Día, replace force
gen Fecha = mdy(Mes, Día, Año)
drop if missing(Fecha)
drop if Año == 0

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
gen ln_TotalVictimas = ln(TotaldeVíctimasdelCaso)
gen Periodo_Historico = .
replace Periodo_Historico = 1 if Año <= 1997
replace Periodo_Historico = 2 if Año >= 1998 & Año <= 2002
replace Periodo_Historico = 3 if Año >= 2003 & Año <= 2010
replace Periodo_Historico = 4 if Año >= 2011

keep if TipodeSecuestro == "EXTORSIVO"
drop if inlist(TipodeLiberación, "ND", "OTRO", "NA") | missing(TipodeLiberación)
gen Y_Resultado = .
replace Y_Resultado = 1 if inlist(TipodeLiberación, "PAGO", "CANJE", "PAGO - POR PRESIÓN/INTERMEDIACIÓN", "CANJE - PAGO")
replace Y_Resultado = 3 if TipodeLiberación == "RESCATE"
replace Y_Resultado = 0 if inlist(TipodeLiberación, "FUGA", "POR PRESIÓN/INTERMEDIACIÓN")
capture confirm variable SituaciónActualdelaVíctima
if _rc == 0 {
    replace Y_Resultado = 2 if inlist(SituaciónActualdelaVíctima, "MUERTO EN CAUTIVERIO", "ASESINADO")
}
gen Duracion_Categ = 4
replace Duracion_Categ = 1 if DíasdeCautiverio <= 2
replace Duracion_Categ = 2 if DíasdeCautiverio > 2 & DíasdeCautiverio <= 30
replace Duracion_Categ = 3 if DíasdeCautiverio > 30 & DíasdeCautiverio != .
drop if missing(Zona_Geografica) | missing(Y_Resultado) | missing(Grupo_Responsable) ///
    | missing(Perfil_Victima) | missing(Modus_Operandi) | missing(Estructura_Secuestro) ///
    | missing(Intervencion_GAULA) | missing(Sexo_Victima) | missing(Duracion_Categ) ///
    | missing(ln_TotalVictimas) | missing(Periodo_Historico)

drop if missing(DíasdeCautiverio) | DíasdeCautiverio <= 0
quietly count
local N_cox = r(N)
display "Cox sample N = `N_cox'"

local causes Payment Rescue Death Escape
local codes  1 3 2 0

forvalues k = 1/4 {
    local cause : word `k' of `causes'
    local code  : word `k' of `codes'
    display _n "=== COX `cause' ==="
    stset DíasdeCautiverio, failure(Y_Resultado==`code')
    stcox i.Grupo_Responsable i.Perfil_Victima i.Zona_Geografica ///
          i.Modus_Operandi i.Estructura_Secuestro i.Intervencion_GAULA ///
          i.Sexo_Victima ln_TotalVictimas i.Periodo_Historico, robust
    estimates store Cox_`cause'
}

* Export appendix rows
file open fh using "`outdir'/cox_appendix_rows.txt", write replace
local coefs ///
    "2.Grupo_Responsable ELN" ///
    "3.Grupo_Responsable Paramilitaries" ///
    "4.Grupo_Responsable CommonDel" ///
    "2.Perfil_Victima MiddleClass" ///
    "3.Perfil_Victima Vulnerable" ///
    "4.Perfil_Victima PublicSector" ///
    "2.Zona_Geografica Andean" ///
    "3.Zona_Geografica Caribbean" ///
    "5.Zona_Geografica EastJungle" ///
    "2.Modus_Operandi Roadblock" ///
    "4.Modus_Operandi Deception" ///
    "2.Estructura_Secuestro GroupStruct" ///
    "1.Intervencion_GAULA GAULA" ///
    "ln_TotalVictimas lnTotal" ///
    "2.Periodo_Historico Crisis" ///
    "3.Periodo_Historico DemSec" ///
    "4.Periodo_Historico Post2011"

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
