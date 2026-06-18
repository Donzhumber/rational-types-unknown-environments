# Appendix 4 Streamlit replication package

This folder contains the Streamlit implementation of the calibrated model used in
Appendix 4.

## Contents

- `app.py`: main Streamlit application.
- `model_logic.py`: structural hazard and payoff logic.
- `rational_behavior.py`: behavioral and mechanism calculations.
- `dynamic_report.py`, `export_calibration_results.py`: report and export helpers.
- `Data_CMH.csv`: empirical input data used by the calibration interface.
- `co_2018_MGN_MPIO_POLITICO.geojson`,
  `san_andres_providencia_arcgis.geojson`, `muni_mapping.json`: geographic inputs.
- `requirements.txt`: Python dependencies.

## Run

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m streamlit run app.py
```

The application opens locally in the browser. With the same inputs and random
seed, the simulated cycles, Table 5.2 outputs, and Tab 6 diagnostics are
reproducible.

