# Identifying Rational Types in Unknown Environments

**Author:** Humberto Bernal  
**Submission Package for *Econometrica***

This repository contains the main manuscript and three self-contained online appendices, including the full replication package (data, code, and formal verification proofs) for the paper *"Identifying Rational Types in Unknown Environments"*.

---

## Repository Structure

The repository is organized as follows:

```text
├── README.md               # This file
├── README.txt              # Text-only version of the main README
├── Main/                   # Main manuscript LaTeX source & figures
│   ├── Bernal_H.tex        # Main manuscript LaTeX source
│   ├── Bernal_H.pdf        # Pre-compiled PDF of the main manuscript
│   ├── references.bib      # Bibliography file
│   ├── econsocart.cls/cfg  # Econometrica document class & config
│   ├── figuras_esp_ordenadas/     # Empirical figures (Fig 1a - Fig 10b)
│   └── figuras_calibracion_journal/ # Calibration figures (ELN / PAR scenarios)
│
├── Appendix_1/             # Online Appendix 1: Macro Dynamics (VEC and LLT)
│   ├── tex/                # LaTeX source for Appendix 1
│   ├── matlab/             # MATLAB replication scripts and data (VEC & Kalman filter)
│   └── README.txt          # Detailed replication instructions for Appendix 1
│
├── Appendix_2/             # Online Appendix 2: Microstructure & Competing Hazards
│   ├── tex/                # LaTeX source and tables for Appendix 2
│   ├── stata/              # Stata replication pipeline (data & scripts)
│   ├── replicate.do        # Stata main execution entry point
│   └── README.txt          # Detailed replication instructions for Appendix 2
│
└── Appendix_3/             # Online Appendix 3: Formal Proofs & Lean 4 Verification
    ├── tex/                # LaTeX source for Appendix 3
    ├── lean/               # Lean 4 formalization files (Appendix3Proofs.lean)
    ├── replicate.sh        # Bash script to run Lean 4 verification
    └── README.txt          # Detailed verification instructions for Appendix 3
```

---

## Getting Started & Compilation

### 1. Compiling the Main Manuscript

To compile the LaTeX source of the main manuscript, navigate to the `Main` directory and run:

```bash
cd Main
pdflatex Bernal_H
bibtex Bernal_H
pdflatex Bernal_H
pdflatex Bernal_H
```

This will generate `Bernal_H.pdf`.

### 2. Compiling the Appendices (LaTeX)

Each appendix contains its own self-contained LaTeX document. To compile them:

*   **Appendix 1:**
    ```bash
    cd Appendix_1/tex
    pdflatex Appendix_1
    bibtex Appendix_1
    pdflatex Appendix_1
    pdflatex Appendix_1
    ```
*   **Appendix 2:**
    ```bash
    cd Appendix_2/tex
    pdflatex Appendix_2
    bibtex Appendix_2
    pdflatex Appendix_2
    pdflatex Appendix_2
    ```
*   **Appendix 3:**
    ```bash
    cd Appendix_3/tex
    pdflatex Appendix_3
    bibtex Appendix_3
    pdflatex Appendix_3
    pdflatex Appendix_3
    ```

---

## Replication Pipelines

Refer to the individual `README.txt` files inside each Appendix folder for comprehensive instructions on how to replicate the quantitative analysis and formal proofs:

*   **Appendix 1 (MATLAB):** Replicates the VEC (Vector Error Correction) and LLT (Local Linear Trend) macro dynamics models. See [Appendix 1 README](Appendix_1/README.txt).
*   **Appendix 2 (Stata & Python):** Replicates the microstructure and competing hazards models. See [Appendix 2 README](Appendix_2/README.txt).
*   **Appendix 3 (Lean 4):** Contains the formal Lean 4 verification codes for the mathematical proofs. See [Appendix 3 README](Appendix_3/README.txt).
