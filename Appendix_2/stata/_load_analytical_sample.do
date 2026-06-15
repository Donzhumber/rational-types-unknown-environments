* ==============================================================================
* Load the analytical estimation sample (n = 1,125).
* Requires pkgroot, datadir, outdir from _setup_paths.do.
* ==============================================================================

capture confirm file "`outdir'/analytical_sample.dta"
if !_rc {
    use "`outdir'/analytical_sample.dta", clear
}
else {
    capture confirm file "`datadir'/analytical_sample.dta"
    if !_rc {
        use "`datadir'/analytical_sample.dta", clear
    }
    else {
        display as text "Building analytical sample from Data_merge.dta..."
        do "`pkgroot'/microstructure_replication.do"
        use "`outdir'/analytical_sample.dta", clear
    }
}
