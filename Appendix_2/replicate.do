* ==============================================================================
* APPENDIX 2 -- ONE-CLICK REPLICATION ENTRY POINT
*
* Run from Stata with any of these (no manual cd required if using full path):
*   do "/path/to/Appendix_2/replicate.do"
*   do replicate.do     [cwd = Appendix_2/]
*   do run_all.do        [cwd = Appendix_2/stata/]
* ==============================================================================
version 19
clear all
cls
set more off

local dir "`c(pwd)'"
local dir : subinstr local dir "\" "/", all
local candidates "`dir'"

* Search relative to this do-file when Stata provides its path.
local file "`c(filename)'"
if "`file'" != "" {
    local file : subinstr local file "\" "/", all
    local fslash = strrpos("`file'", "/")
    if `fslash' > 0 {
        local filedir = substr("`file'", 1, `fslash' - 1)
        local candidates "`candidates' `filedir'"
        local parent = substr("`filedir'", 1, strrpos("`filedir'", "/") - 1)
        if "`parent'" != "`filedir'" local candidates "`candidates' `parent'"
    }
}

local app2root ""
local runfile ""

tokenize `candidates'
local n : word count `candidates'
forvalues i = 1/`n' {
    local try "`dir`i''"
    if "`try'" == "" continue

    capture confirm file "`try'/stata/run_all.do"
    if !_rc {
        local app2root "`try'"
        local runfile "`try'/stata/run_all.do"
        continue, break
    }

    capture confirm file "`try'/run_all.do"
    if !_rc {
        local runfile "`try'/run_all.do"
        continue, break
    }
}

if "`runfile'" == "" {
    local walk "`dir'"
    forvalues step = 1/12 {
        if "`walk'" == "" continue, break

        capture confirm file "`walk'/stata/run_all.do"
        if !_rc {
            local app2root "`walk'"
            local runfile "`walk'/stata/run_all.do"
            continue, break
        }

        capture confirm file "`walk'/run_all.do"
        if !_rc {
            local runfile "`walk'/run_all.do"
            continue, break
        }

        local slash = strrpos("`walk'", "/")
        if `slash' < 1 continue, break
        local walk = substr("`walk'", 1, `slash' - 1)
    }
}

if "`runfile'" == "" {
    display as error _n "Could not find stata/run_all.do."
    display as error "Ensure you downloaded the full Appendix_2/ folder."
    display as error "Then run either:"
    display as error ` "  do "/full/path/to/Appendix_2/replicate.do""'
    display as error "  or set cwd to Appendix_2/ and run:  do replicate.do"
    display as error "  or set cwd to Appendix_2/stata/ and run:  do run_all.do"
    display as error _n "Current working directory: `c(pwd)'"
    exit 601
}

display as text "Replication script: `runfile'"
cd "`=substr("`runfile'", 1, strrpos("`runfile'", "/") - 1)'"
do run_all.do
